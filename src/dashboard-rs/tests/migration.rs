use mai_dashboard::{
    config::{Config, Manager},
    db,
};
use rusqlite::params;
use serde_json::json;
fn config() -> Config {
    Config {
        url_base: String::new(),
        managers: vec![Manager {
            label: "Fixture".into(),
            url: "http://fixture".into(),
            traj_root: Some("oss://fixture/root".into()),
            capacity_mode: None,
            k8s_inventory: None,
        }],
    }
}
fn legacy(c: &rusqlite::Connection) {
    c.execute_batch("CREATE TABLE attempts(mgr TEXT,rel_id TEXT,task_id TEXT,attempt TEXT,train_id TEXT,rollout_id INTEGER,is_eval INTEGER,score REAL,reason TEXT,steps INTEGER,mtime REAL,created_at REAL,model TEXT,error TEXT,suite TEXT,benchmark_model TEXT,benchmark_run_id TEXT,judge_reward REAL,last_write REAL,error_mtime REAL,eval_pkg TEXT,PRIMARY KEY(mgr,rel_id));CREATE TABLE trained(mgr TEXT,train_id TEXT,seg TEXT,manifest TEXT,PRIMARY KEY(mgr,train_id,seg));").unwrap();
}
fn insert(c: &rusqlite::Connection, id: &str, score: f64) {
    c.execute("INSERT OR REPLACE INTO attempts(mgr,rel_id,task_id,attempt,train_id,rollout_id,is_eval,score,mtime,created_at)VALUES('http://fixture',?,'task','attempt','run',1,0,?,?,?)",params![id,score,mai_dashboard::now(),mai_dashboard::now()]).unwrap();
}
#[test]
fn concurrent_import_replay_deletion_and_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("legacy.db");
    let mut src = db::connect(&p).unwrap();
    legacy(&src);
    for i in 0..2100 {
        insert(&src, &format!("run/r00001/task/{i:05}"), 0.);
    }
    db::install_journal(&mut src).unwrap();
    let side = tmp.path().join("index.db");
    let mut dst = db::connect(&side).unwrap();
    db::schema(&dst).unwrap();
    assert_eq!(db::import_batch(&mut dst, &src, &config()).unwrap(), 2000);
    insert(&src, "run/r00001/task/00001", 1.);
    src.execute(
        "DELETE FROM attempts WHERE rel_id='run/r00001/task/00002'",
        [],
    )
    .unwrap();
    insert(&src, "aaa", 0.8);
    while db::import_batch(&mut dst, &src, &config()).unwrap() > 0 {}
    drop(dst);
    let mut dst = db::connect(&side).unwrap();
    while db::replay(&mut dst, &src, &config()).unwrap() > 0 {}
    let n: i64 = dst
        .query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 2100);
    let score: f64 = dst
        .query_row(
            "SELECT json_extract(payload,'$.score') FROM attempts WHERE id='run/r00001/task/00001'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(score, 1.);
    let before = db::get_meta(&dst, "journal_seq").unwrap();
    assert_eq!(db::replay(&mut dst, &src, &config()).unwrap(), 0);
    assert_eq!(before, db::get_meta(&dst, "journal_seq").unwrap());
    while db::rebuild_one(&mut dst, &config()).unwrap() {}
    let views = db::values(&dst, "SELECT data FROM views WHERE kind='run'", &[]).unwrap();
    assert_eq!(views.len(), 1);
    assert_eq!(views[0]["count"], 2100);
    assert_eq!(views[0]["passed"], 2);
}
#[test]
fn journal_rollback_is_atomic() {
    let mut src = rusqlite::Connection::open_in_memory().unwrap();
    legacy(&src);
    db::install_journal(&mut src).unwrap();
    {
        let tx = src.transaction().unwrap();
        insert(&tx, "rolled_back", 1.);
    }
    assert_eq!(
        src.query_row("SELECT COUNT(*) FROM dashboard_rs_journal", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
#[test]
fn moving_run_rebuilds_old_and_new() {
    let mut c = rusqlite::Connection::open_in_memory().unwrap();
    db::schema(&c).unwrap();
    let e = json!({"id":"task/attempt","task_id":"task","attempt":"attempt","train_id":"one","score":1.,"mtime":mai_dashboard::now()});
    db::upsert(&c, "0", e.clone()).unwrap();
    while db::rebuild_one(&mut c, &config()).unwrap() {}
    let mut e = e;
    e["train_id"] = json!("two");
    db::upsert(&c, "0", e).unwrap();
    while db::rebuild_one(&mut c, &config()).unwrap() {}
    let v = db::values(&c, "SELECT data FROM views WHERE kind='run'", &[]).unwrap();
    assert_eq!(v.len(), 1);
    assert_eq!(v[0]["run_key"], "two");
}
#[test]
fn legacy_runtime_mixes_plain_json_and_compressed_sessions() {
    use std::io::Write;
    let src = rusqlite::Connection::open_in_memory().unwrap();
    src.execute_batch("CREATE TABLE dashboard_runtime(mgr TEXT,updated_at REAL,health TEXT,sessions BLOB);CREATE TABLE samples(mgr TEXT,t REAL,total INTEGER);CREATE TABLE dashboard_views(mgr TEXT,status TEXT);").unwrap();
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    z.write_all(b"{\"sessions\":[],\"summary\":{\"total\":5}}")
        .unwrap();
    let blob = z.finish().unwrap();
    src.execute(
        "INSERT INTO dashboard_runtime VALUES('http://fixture',1,'{\"ok\":true}',?)",
        [blob],
    )
    .unwrap();
    let mut dst = rusqlite::Connection::open_in_memory().unwrap();
    db::schema(&dst).unwrap();
    db::sync_runtime(&mut dst, &src, &config()).unwrap();
    let v = db::values(&dst, "SELECT sessions FROM runtime", &[]).unwrap();
    assert_eq!(v[0]["summary"]["total"], 5);
}
#[test]
fn cross_run_task_deltas_preserve_means_and_recency_after_removal() {
    let mut c = rusqlite::Connection::open_in_memory().unwrap();
    db::schema(&c).unwrap();
    let t = mai_dashboard::now();
    for (i, score) in [(1, 1.), (2, 0.4)] {
        db::upsert(&c,"0",json!({"id":format!("run{i}/r00001/shared/a"),"task_id":"shared","attempt":"a","train_id":format!("run{i}"),"rollout_id":1,"score":score,"mtime":t+i as f64})).unwrap();
    }
    while db::rebuild_one(&mut c, &config()).unwrap() {}
    let task = || {
        db::values(&c, "SELECT data FROM views WHERE kind='task_all'", &[])
            .unwrap()
            .pop()
            .unwrap()
    };
    assert!((task()["mean_score"].as_f64().unwrap() - 0.7).abs() < 1e-9);
    assert_eq!(task()["count"], 2);
    db::remove(&c, "0", "run2/r00001/shared/a").unwrap();
    while db::rebuild_one(&mut c, &config()).unwrap() {}
    let task = db::values(&c, "SELECT data FROM views WHERE kind='task_all'", &[])
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(task["mean_score"], 1.);
    assert_eq!(task["last_mtime"], t + 1.);
    assert_eq!(task["count"], 1);
}
