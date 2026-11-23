use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use mai_dashboard::{
    config::{Config, Manager},
    db, indexer,
    storage::{Object, Page},
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};
#[derive(Default)]
struct Mock {
    objects: BTreeMap<String, (String, String)>,
    gets: usize,
    listed: usize,
    fail_get: bool,
    fail_list: bool,
    list_at: Option<f64>,
    list_requests: Vec<(String, bool, String)>,
    blocked_prefix: Option<String>,
    list_started: Arc<tokio::sync::Notify>,
    list_release: Arc<tokio::sync::Notify>,
    list_calls: HashMap<String, usize>,
}
type Shared = Arc<Mutex<Mock>>;
async fn object(State(s): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Response {
    let mut s = s.lock().unwrap();
    s.gets += 1;
    if s.fail_get {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    match s.objects.get(&q["path"]) {
        Some((body, _)) => body.clone().into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
async fn list(State(s): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Response {
    let gate = {
        let mut state = s.lock().unwrap();
        let prefix = q.get("path").cloned().unwrap_or_default();
        *state.list_calls.entry(prefix.clone()).or_default() += 1;
        (state.blocked_prefix.as_deref() == Some(prefix.as_str()))
            .then(|| (state.list_started.clone(), state.list_release.clone()))
    };
    if let Some((started, release)) = gate {
        started.notify_one();
        release.notified().await;
    }
    let mut s = s.lock().unwrap();
    if s.fail_list {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let prefix = q.get("path").map(String::as_str).unwrap_or("");
    let cursor = q.get("cursor").map(String::as_str).unwrap_or("");
    let recursive = q.get("recursive").is_some_and(|v| v == "true");
    s.list_requests
        .push((prefix.to_owned(), recursive, cursor.to_owned()));
    let mut entries = BTreeMap::new();
    for (path, (body, etag)) in &s.objects {
        if !path.starts_with(prefix) {
            continue;
        }
        let tail = &path[prefix.len()..];
        if !recursive && let Some((dir, _)) = tail.split_once('/') {
            let key = format!("{prefix}{dir}/");
            entries.insert(
                key.clone(),
                Object {
                    path: key,
                    etag: None,
                    modified: 0.,
                    size: 0,
                    dir: true,
                },
            );
            continue;
        }
        entries.insert(
            path.clone(),
            Object {
                path: path.clone(),
                etag: Some(etag.clone()),
                modified: 1780000000.,
                size: body.len() as u64,
                dir: false,
            },
        );
    }
    let objects: Vec<_> = entries
        .into_values()
        .filter(|o| o.path.as_str() > cursor)
        .take(1000)
        .collect();
    let cursor = if objects.len() == 1000 {
        objects.last().map(|o| o.path.clone())
    } else {
        None
    };
    s.listed += objects.len();
    Json(Page {
        objects,
        cursor,
        listed_at: Some(s.list_at.unwrap_or_else(mai_dashboard::now)),
    })
    .into_response()
}
fn cfg() -> Config {
    Config {
        url_base: String::new(),
        managers: vec![Manager {
            label: "fixture".into(),
            url: "http://fixture".into(),
            traj_root: Some("oss://fixture/root".into()),
            capacity_mode: None,
            k8s_inventory: None,
        }],
    }
}
async fn gateway() -> (Shared, String, tokio::task::JoinHandle<()>) {
    let state = Arc::new(Mutex::new(Mock::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/get", get(object))
        .route("/list", get(list))
        .with_state(state.clone());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (state, url, task)
}
fn force(c: &rusqlite::Connection) {
    c.execute("UPDATE scan_jobs SET due=0", []).unwrap();
}
#[tokio::test]
async fn late_results_overwrites_failures_and_safe_pruning() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let name = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let id = format!("model/run/r00001/task/{name}_fixture");
    let root = "model/run";
    {
        let mut s = state.lock().unwrap();
        s.objects.insert(
            format!("{id}/meta.json"),
            (
                serde_json::json!({"model":"actual-model","task_id":"suite/task","train_id":"unrelated"}).to_string(),
                "meta1".into(),
            ),
        );
        s.objects
            .insert(format!("{id}/traj.json"), ("{}".into(), "traj1".into()));
        s.objects.insert(
            format!("{id}/screenshots/frame-0.png"),
            ("image".into(), "img1".into()),
        );
    }
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    let c = db::connect(&path).unwrap();
    let rows = db::values(&c, "SELECT payload FROM attempts", &[]).unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0]["score"].is_null());
    assert_eq!(rows[0]["model"], "actual-model");
    assert_eq!(rows[0]["task_id"], "suite/task");
    assert_eq!(rows[0]["train_id"], "run");
    state.lock().unwrap().objects.insert(
        format!("{id}/result.txt"),
        ("1\nbuilt_in".into(), "result1".into()),
    );
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    let rows = db::values(&c, "SELECT payload FROM attempts", &[]).unwrap();
    assert_eq!(rows[0]["score"], 1.);
    {
        let mut s = state.lock().unwrap();
        s.objects.insert(
            format!("{id}/result.txt"),
            ("0\nbuilt_in".into(), "result2".into()),
        );
        s.fail_get = true;
    }
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["score"],
        1.
    );
    state.lock().unwrap().fail_get = false;
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["score"],
        0.
    );
    {
        let mut s = state.lock().unwrap();
        s.objects.clear();
        s.fail_list = true;
    }
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[])
            .unwrap()
            .len(),
        1
    );
    state.lock().unwrap().fail_list = false;
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    assert!(
        db::values(&c, "SELECT payload FROM attempts", &[])
            .unwrap()
            .is_empty()
    );
    task.abort();
}
#[tokio::test]
async fn multi_page_attempt_resumes_without_losing_terminal_artifact() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let id = "run/r00001/task/20260930_120000_fixture";
    {
        let mut s = state.lock().unwrap();
        s.objects.insert(
            format!("{id}/result.txt"),
            ("1\nfixture".into(), "result1".into()),
        );
        for i in 0..1050 {
            s.objects.insert(
                format!("{id}/screenshots/frame-{i:04}.png"),
                ("image".into(), format!("img{i}")),
            );
        }
    }
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        id.into(),
        1,
    )
    .await
    .unwrap();
    let c = db::connect(&path).unwrap();
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        id.into(),
        1,
    )
    .await
    .unwrap();
    let rows = db::values(&c, "SELECT payload FROM attempts", &[]).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["steps"], 1050);
    assert_eq!(rows[0]["score"], 1.);
    assert_eq!(state.lock().unwrap().gets, 1);
    task.abort();
}

#[tokio::test]
async fn legacy_task_discovery_finds_new_runs_without_waiting_for_history() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let first = "Task/20260930_120000_first";
    let second = "Task/20260930_120001_second";
    let add = |id: &str, train: &str| {
        let mut s = state.lock().unwrap();
        s.objects.insert(format!("{id}/meta.json"),(serde_json::json!({"train_id":train,"created_at":mai_dashboard::now(),"evaluation":false,"rollout_id":1}).to_string(),format!("meta-{train}")));
        s.objects.insert(
            format!("{id}/result.txt"),
            ("1\nfixture".into(), format!("result-{train}")),
        );
    };
    add(first, "first");
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "Task".into(),
        10,
    )
    .await
    .unwrap();
    let mut c = db::connect(&path).unwrap();
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let before = state.lock().unwrap().listed;
    c.execute("UPDATE scan_jobs SET due=0 WHERE kind='attempts'", [])
        .unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "attempts",
        1,
    )
    .await
    .unwrap();
    assert_eq!(
        state.lock().unwrap().listed,
        before,
        "unchanged legacy prefixes should list only beyond the durable tail"
    );
    add(second, "second");
    c.execute("UPDATE scan_jobs SET due=0 WHERE kind='attempts'", [])
        .unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "priority",
        10,
    )
    .await
    .unwrap();
    while db::rebuild_one(&mut c, &cfg()).unwrap() {}
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        c.query_row(
            "SELECT COUNT(*) FROM freshness_events WHERE published_at IS NOT NULL AND kind='new_run_oss'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    task.abort();
}
#[tokio::test]
async fn completed_enumeration_removes_deleted_benchmark_and_trained_sidecars() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let id = "run/r00001/task/20260930_120000_fixture";
    {
        let mut s = state.lock().unwrap();
        s.objects.insert(
            format!("{id}/result.txt"),
            ("1\nfixture".into(), "result".into()),
        );
        s.objects.insert("run/run_config.json".into(),(serde_json::json!({"model_name":"served","benchmark":{"suite":"fixture","model":null,"run_id":"run"}}).to_string(),"config".into()));
        s.objects.insert(
            "run/r00001/trained.json".into(),
            ("{\"tasks\":[]}".into(), "trained".into()),
        );
    }
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "run".into(),
        10,
    )
    .await
    .unwrap();
    let c = db::connect(&path).unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["suite"],
        "fixture"
    );
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["benchmark_model"],
        "served"
    );
    state.lock().unwrap().objects.insert(
        "run/run_config.json".into(),
        (serde_json::json!({"model_name":"served","benchmark":{"suite":"fixture","model":"explicit","run_id":"run"}}).to_string(),"config2".into()),
    );
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "run".into(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["benchmark_model"],
        "explicit"
    );
    {
        let mut s = state.lock().unwrap();
        s.objects.remove("run/run_config.json");
        s.objects.remove("run/r00001/trained.json");
    }
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "run".into(),
        10,
    )
    .await
    .unwrap();
    assert!(db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["suite"].is_null());
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM trained", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    task.abort();
}
#[tokio::test]
async fn older_listing_cannot_replace_a_newer_observation() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let id = "run/r00001/task/20260930_120000_fixture";
    state.lock().unwrap().objects.insert(
        format!("{id}/result.txt"),
        ("1\nfixture".into(), "one".into()),
    );
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "run".into(),
        10,
    )
    .await
    .unwrap();
    let c = db::connect(&path).unwrap();
    {
        let mut s = state.lock().unwrap();
        s.list_at = Some(mai_dashboard::now() - 60.);
        s.objects.insert(
            format!("{id}/result.txt"),
            ("0\nfixture".into(), "zero".into()),
        );
    }
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "run".into(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["score"],
        1.
    );
    state.lock().unwrap().list_at = None;
    force(&c);
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "run".into(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[]).unwrap()[0]["score"],
        0.
    );
    task.abort();
}

#[tokio::test]
async fn run_root_metadata_does_not_create_an_active_namespace() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let root = "model/20260930_run";
    let id = "model/20260930_run/r00001/task/20260930_120000_fixture";
    {
        let mut s = state.lock().unwrap();
        s.objects
            .insert(format!("{root}/meta.json"), ("{}".into(), "root".into()));
        s.objects.insert(
            format!("{id}/result.txt"),
            ("1\nfixture".into(), "result".into()),
        );
    }
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        root.into(),
        10,
    )
    .await
    .unwrap();
    let c = db::connect(&path).unwrap();
    assert_eq!(
        db::values(&c, "SELECT payload FROM attempts", &[])
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        c.query_row(
            "SELECT COUNT(*) FROM scan_jobs WHERE prefix='model/20260930_run/' AND kind='active'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0','model/20260930_run/','active',80)",
        [],
    )
    .unwrap();
    indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "active", 1)
        .await
        .unwrap();
    assert_eq!(
        c.query_row(
            "SELECT COUNT(*) FROM scan_jobs WHERE prefix='model/20260930_run/' AND kind='active'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    task.abort();
}

#[tokio::test]
async fn discovery_does_not_poll_unsupported_artifact_subtrees() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    {
        let mut s = state.lock().unwrap();
        s.objects.insert(
            "model/unsupported-task/20260930_120000_x/screenshots/1.png".into(),
            ("image".into(), "image".into()),
        );
        s.objects.insert(
            "model/valid-run/r00001/task/20260930_120000_x/result.txt".into(),
            ("1\nfixture".into(), "result".into()),
        );
    }
    indexer::scan(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "".into(),
        24,
    )
    .await
    .unwrap();
    let c = db::connect(&path).unwrap();
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(c.query_row("SELECT COUNT(*) FROM scan_jobs WHERE prefix LIKE 'model/unsupported-task/%' AND prefix <> 'model/unsupported-task/'", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
    c.execute("INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0','model/unsupported-task/20260930_120000_x/screenshots/','discover',90)", []).unwrap();
    indexer::seed(&c, &cfg()).unwrap();
    assert_eq!(
        c.query_row(
            "SELECT COUNT(*) FROM scan_jobs WHERE prefix LIKE '%screenshots%'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    task.abort();
}

#[tokio::test]
async fn new_namespaces_use_discovery_lane_and_oss_time_despite_producer_skew() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let producer_date = chrono::Local::now() + chrono::Duration::seconds(162);
    let producer_time = producer_date.timestamp() as f64;
    let id = format!(
        "model/fresh-run/r00001/task/{}_fixture",
        producer_date.format("%Y%m%d_%H%M%S")
    );
    state.lock().unwrap().objects.insert(
        format!("{id}/meta.json"),
        (
            serde_json::json!({"created_at":producer_time,"task_id":"task"}).to_string(),
            "meta".into(),
        ),
    );
    let mut c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url,
        "token".into(),
        "0".into(),
        "discovery",
        24,
    )
    .await
    .unwrap();
    while db::rebuild_one(&mut c, &cfg()).unwrap() {}
    let entry = db::values(&c, "SELECT payload FROM attempts", &[])
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(entry["created_at"], producer_time);
    let (source, observed, published): (f64,f64,f64) = c.query_row(
        "SELECT source_at,observed_at,published_at FROM freshness_events WHERE kind='new_run_oss'",[],
        |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
    assert_eq!(source, 1780000000.);
    assert!(published >= observed && observed >= source);
    let priority: i64 = c
        .query_row(
            "SELECT priority FROM scan_jobs WHERE prefix='model/fresh-run/' AND kind='reconcile'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        priority, 10,
        "completed first scan returns to the historical lane"
    );
    task.abort();
}

#[tokio::test]
async fn parent_watches_and_new_directories_bypass_old_discovery_backlog() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    state.lock().unwrap().objects.insert(
        "model/new-run/r00001/task/20260930_120000_x/meta.json".into(),
        (r#"{"task_id":"task"}"#.into(), "meta".into()),
    );
    let mut c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    for i in 0..1000 {
        let prefix = format!("old/child-{i:04}/");
        c.execute(
            "INSERT INTO scan_jobs(m,prefix,kind,priority,due) VALUES('0',?,'discover',90,0)",
            [&prefix],
        )
        .unwrap();
        c.execute(
            "INSERT INTO scan_completed VALUES('0',?,'discover',1)",
            [&prefix],
        )
        .unwrap();
    }
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,priority,due) VALUES('0','model/','discover',90,1)",
        [],
    )
    .unwrap();
    db::set_meta(&c, "discovery_seeded", "true").unwrap();
    indexer::seed(&c, &cfg()).unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url,
        "token".into(),
        "0".into(),
        "discovery",
        36,
    )
    .await
    .unwrap();
    while db::rebuild_one(&mut c, &cfg()).unwrap() {}
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert!(
        c.query_row(
            "SELECT COUNT(*) FROM scan_jobs WHERE prefix LIKE 'old/%' AND due=0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap()
            > 900
    );
    assert_eq!(
        c.query_row(
            "SELECT priority FROM scan_jobs WHERE prefix='model/'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        95
    );
    assert_eq!(
        c.query_row(
            "SELECT priority FROM scan_jobs WHERE prefix='old/child-0000/'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        90
    );
    task.abort();
}

#[tokio::test]
async fn slow_active_poll_does_not_hold_other_shards_or_duplicate_reads() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,priority,due) VALUES('0','','discover',100,1e20)",
        [],
    )
    .unwrap();
    let (started, release) = {
        let mut s = state.lock().unwrap();
        for i in 0..12 {
            let id = format!("model/run/r00001/task-{i:02}/20260930_120000_x");
            let prefix = format!("{id}/");
            c.execute(
                "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0',?,'active',80)",
                [&prefix],
            )
            .unwrap();
            s.objects.insert(
                format!("{id}/result.txt"),
                ("1\nfixture".into(), "result".into()),
            );
            if i == 0 {
                s.blocked_prefix = Some(prefix);
            }
        }
        (s.list_started.clone(), s.list_release.clone())
    };
    let scans = indexer::ACTIVE_LANES.into_iter().map(|lane| {
        indexer::scan_lane(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            lane,
            3,
        )
    });
    let all = futures::future::join_all(scans);
    let check = async {
        started.notified().await;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let count: i64 = c
                    .query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get(0))
                    .unwrap();
                if count >= 9 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("one slow poll blocked the other active shards");
        release.notify_one();
    };
    let (results, ()) = tokio::join!(all, check);
    for result in results {
        result.unwrap();
    }
    assert_eq!(
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        12
    );
    let s = state.lock().unwrap();
    assert_eq!(s.list_calls.len(), 12);
    assert!(s.list_calls.values().all(|n| *n == 1));
    assert_eq!(s.gets, 12);
    task.abort();
}

#[tokio::test]
async fn known_runs_discover_new_attempts_without_historical_screenshots() {
    for root in ["train", "model/train"] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.db");
        let (state, url, task) = gateway().await;
        let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let old = format!("{root}/r00000/task/{stamp}_a");
        state.lock().unwrap().objects.insert(
            format!("{old}/result.txt"),
            ("1\nbuilt_in".into(), "old".into()),
        );
        indexer::scan(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            root.into(),
            8,
        )
        .await
        .unwrap();
        let mut c = db::connect(&path).unwrap();
        c.execute("UPDATE scan_jobs SET due=1e20,cursor='preserved-history-cursor' WHERE kind='reconcile'",[]).unwrap();
        indexer::seed(&c, &cfg()).unwrap();
        let same_task = format!("{root}/r00000/task/{stamp}_b");
        let new_rollout = format!("{root}/r00001/new-task/{stamp}_a");
        {
            let mut s = state.lock().unwrap();
            for id in [&same_task, &new_rollout] {
                s.objects.insert(
                    format!("{id}/result.txt"),
                    ("0.8\nbuilt_in".into(), "new".into()),
                );
            }
            for i in 0..5000 {
                s.objects.insert(
                    format!("{old}/screenshots/{i:05}.png"),
                    ("image".into(), "shot".into()),
                );
            }
            s.list_requests.clear();
        }
        indexer::scan_lane(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            "watch",
            12,
        )
        .await
        .unwrap();
        {
            let s = state.lock().unwrap();
            assert!(
                s.list_requests
                    .iter()
                    .all(|(p, recursive, _)| !recursive && !p.starts_with(&format!("{old}/")))
            );
            assert!(s.list_requests.len() <= 6);
        }
        assert_eq!(
            c.query_row(
                "SELECT COUNT(*) FROM scan_jobs WHERE kind='active'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            2
        );
        indexer::scan_lane(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            "active",
            12,
        )
        .await
        .unwrap();
        while db::rebuild_one(&mut c, &cfg()).unwrap() {}
        for id in [&same_task, &new_rollout] {
            let score: f64 = c
                .query_row(
                    "SELECT json_extract(payload,'$.score') FROM attempts WHERE id=?",
                    [id],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(score, 0.8);
        }
        assert_eq!(
            c.query_row(
                "SELECT cursor FROM scan_jobs WHERE kind='reconcile' LIMIT 1",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
            "preserved-history-cursor"
        );
        let task_prefix = format!("{root}/r00000/task/");
        let cursor: String = c
            .query_row(
                "SELECT cursor FROM scan_jobs WHERE kind='watch' AND prefix=?",
                [&task_prefix],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cursor, format!("{same_task}/"));
        drop(c);
        let c = db::connect(&path).unwrap();
        indexer::seed(&c, &cfg()).unwrap();
        let after_restart = format!("{root}/r00000/task/{stamp}_c");
        state.lock().unwrap().objects.insert(
            format!("{after_restart}/result.txt"),
            ("1\nbuilt_in".into(), "later".into()),
        );
        c.execute("UPDATE scan_jobs SET due=0 WHERE kind='watch'", [])
            .unwrap();
        indexer::scan_lane(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            "watch",
            12,
        )
        .await
        .unwrap();
        assert_eq!(
            c.query_row(
                "SELECT COUNT(*) FROM scan_jobs WHERE kind='active' AND prefix=?",
                [format!("{after_restart}/")],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            1
        );
        let calls = state.lock().unwrap().list_requests.len();
        c.execute("UPDATE live_run_watches SET expires=0", [])
            .unwrap();
        c.execute("UPDATE scan_jobs SET due=0 WHERE kind='watch'", [])
            .unwrap();
        indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "watch", 12)
            .await
            .unwrap();
        assert_eq!(
            state.lock().unwrap().list_requests.len(),
            calls,
            "expired watches must not issue origin LISTs"
        );
        assert_eq!(
            c.query_row(
                "SELECT COUNT(*) FROM scan_jobs WHERE kind='watch'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
        task.abort();
    }
}

#[tokio::test]
async fn directory_watch_pagination_retains_all_unknown_attempts_and_cursor() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO live_run_watches VALUES('0','model/run/',?)",
        [mai_dashboard::now() + 3600.],
    )
    .unwrap();
    c.execute("INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0','model/run/r00000/task/','watch',85)",[]).unwrap();
    for i in 0..1001 {
        state.lock().unwrap().objects.insert(
            format!("model/run/r00000/task/20261001_120000_{i:05}/result.txt"),
            ("1".into(), "v1".into()),
        );
    }
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "watch",
        3,
    )
    .await
    .unwrap();
    assert_eq!(
        c.query_row(
            "SELECT COUNT(*) FROM scan_jobs WHERE kind='active'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1001
    );
    let cursor: String = c
        .query_row("SELECT cursor FROM scan_jobs WHERE kind='watch'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(cursor.ends_with("01000/"));
    c.execute("UPDATE scan_jobs SET due=0 WHERE kind='watch'", [])
        .unwrap();
    indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "watch", 1)
        .await
        .unwrap();
    assert_eq!(
        c.query_row("SELECT cursor FROM scan_jobs WHERE kind='watch'", [], |r| r
            .get::<_, String>(0))
            .unwrap(),
        cursor
    );
    assert_eq!(
        state.lock().unwrap().gets,
        0,
        "directory discovery never GETs artifacts"
    );
    task.abort();
}

#[tokio::test]
async fn directory_watch_backpressure_preserves_unqueued_ids_without_relisting() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO live_run_watches VALUES('0','model/run/',?)",
        [mai_dashboard::now() + 3600.],
    )
    .unwrap();
    c.execute("INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0','model/run/r00000/task/','watch',85)",[]).unwrap();
    for i in 0..1025 {
        state.lock().unwrap().objects.insert(
            format!("model/run/r00000/task/20261001_120000_{i:05}/result.txt"),
            ("1".into(), "v1".into()),
        );
    }
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "watch",
        3,
    )
    .await
    .unwrap();
    assert_eq!(
        c.query_row(
            "SELECT count(*) FROM scan_jobs WHERE kind='active'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1024
    );
    let cursor: String = c
        .query_row("SELECT cursor FROM scan_jobs WHERE kind='watch'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(cursor.ends_with("01023/"));
    let calls = state.lock().unwrap().list_requests.len();
    c.execute("UPDATE scan_jobs SET due=0 WHERE kind='watch'", [])
        .unwrap();
    let version: i64 = c
        .query_row("PRAGMA data_version", [], |r| r.get(0))
        .unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "watch",
        1,
    )
    .await
    .unwrap();
    assert_eq!(state.lock().unwrap().list_requests.len(), calls);
    assert_eq!(
        c.query_row("PRAGMA data_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        version,
        "blocked watches do not write retry deadlines"
    );
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "active",
        3,
    )
    .await
    .unwrap();
    c.execute("UPDATE scan_jobs SET due=0 WHERE kind='watch'", [])
        .unwrap();
    indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "watch", 1)
        .await
        .unwrap();
    assert_eq!(
        c.query_row("SELECT cursor FROM scan_jobs WHERE kind='watch'", [], |r| r
            .get::<_, String>(0))
            .unwrap(),
        "model/run/r00000/task/20261001_120000_01024/"
    );
    assert_eq!(c.query_row("SELECT (SELECT count(*) FROM attempts)+(SELECT count(*) FROM scan_jobs WHERE kind='active')",[],|r|r.get::<_,i64>(0)).unwrap(),1025);
    task.abort();
}

#[tokio::test]
async fn blocked_task_watches_cannot_hide_parent_discovery() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO live_run_watches VALUES('0','model/run/',?)",
        [mai_dashboard::now() + 3600.],
    )
    .unwrap();
    for i in 0..1024 {
        c.execute(
            "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0',?,'active',80)",
            [format!("model/run/r00000/task/attempt{i}/")],
        )
        .unwrap();
    }
    for i in 0..30 {
        c.execute(
            "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0',?,'watch',85)",
            [format!("model/run/r00000/task{i}/")],
        )
        .unwrap();
    }
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,due,priority) VALUES('0','model/run/', 'watch',1,85)",
        [],
    )
    .unwrap();
    indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "watch", 1)
        .await
        .unwrap();
    let s = state.lock().unwrap();
    assert_eq!(s.list_requests.len(), 1);
    assert_eq!(s.list_requests[0].0, "model/run/");
    assert!(!s.list_requests[0].1);
    assert_eq!(
        c.query_row(
            "SELECT count(*) FROM scan_jobs WHERE kind='watch' AND due=0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        30
    );
    task.abort();
}

#[tokio::test]
async fn old_unindexed_backfill_cannot_starve_known_active_results() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    for i in 0..40 {
        let id = format!("model/run/r00000/task/20261001_120000_{i:05}");
        state
            .lock()
            .unwrap()
            .objects
            .insert(format!("{id}/result.txt"), ("1".into(), "v1".into()));
        if i >= 38 {
            db::upsert(&c,"0",serde_json::json!({"id":id,"task_id":"task","score":null,"train_id":"run","rollout_id":0,"created_at":mai_dashboard::now(),"last_write":mai_dashboard::now()})).unwrap();
        }
        c.execute(
            "INSERT INTO scan_jobs(m,prefix,kind,due,priority) VALUES('0',?,'active',?,80)",
            rusqlite::params![format!("{id}/"), if i >= 38 { 1. } else { 0. }],
        )
        .unwrap();
    }
    indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "active", 3)
        .await
        .unwrap();
    for i in 38..40 {
        let id = format!("model/run/r00000/task/20261001_120000_{i:05}");
        assert_eq!(
            c.query_row(
                "SELECT json_extract(payload,'$.score') FROM attempts WHERE m='0' AND id=?",
                [id],
                |r| r.get::<_, f64>(0)
            )
            .unwrap(),
            1.
        );
    }
    assert_eq!(
        c.query_row("SELECT count(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        3,
        "first observations also make progress"
    );
    let s = state.lock().unwrap();
    assert_eq!(s.list_requests.len(), 3);
    assert_eq!(s.gets, 3);
    assert!(s.list_calls.values().all(|n| *n == 1));
    task.abort();
}

#[tokio::test]
async fn live_sessions_rescue_empty_run_and_discover_later_attempts() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,due,priority) VALUES('0','','discover',?,100)",
        [mai_dashboard::now() + 3600.],
    )
    .unwrap();
    for i in 0..1000 {
        c.execute(
            "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0',?,'discover',90)",
            [format!("model/old{i:04}/")],
        )
        .unwrap();
    }
    c.execute("INSERT INTO scan_jobs(m,prefix,kind,cursor,priority) VALUES('0','model/run/','discover','',90)",[]).unwrap();
    let config = cfg();
    indexer::observe_session_runs(&c, &config.managers[0], "0",
        &serde_json::json!({"traj_log_dir":"oss://fixture/root/model"}),
        &serde_json::json!({"sessions":[{"leased":true,"train_id":"run"},{"leased":true,"train_id":"run"}]})).unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "discovery",
        1,
    )
    .await
    .unwrap();
    assert_eq!(state.lock().unwrap().list_requests[0].0, "model/run/");
    let id = "model/run/r00000/task/20261001_120000_a";
    state
        .lock()
        .unwrap()
        .objects
        .insert(format!("{id}/result.txt"), ("1".into(), "v1".into()));
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "watch",
        3,
    )
    .await
    .unwrap();
    indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "active", 1)
        .await
        .unwrap();
    assert_eq!(
        c.query_row(
            "SELECT json_extract(payload,'$.score') FROM attempts WHERE id=?",
            [id],
            |r| r.get::<_, f64>(0)
        )
        .unwrap(),
        1.
    );
    assert_eq!(
        c.query_row(
            "SELECT count(*) FROM scan_jobs WHERE kind='discover' AND prefix LIKE 'model/old%'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1000
    );
    assert_eq!(
        c.query_row("SELECT count(*) FROM live_run_watches", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    task.abort();
}

#[test]
fn session_run_hints_cannot_escape_configured_root() {
    let tmp = tempfile::tempdir().unwrap();
    let c = db::connect(&tmp.path().join("index.db")).unwrap();
    db::schema(&c).unwrap();
    let config = cfg();
    for root in [
        "oss://fixture/root-elsewhere/model",
        "oss://fixture/root/../model",
        "oss://other/root/model",
    ] {
        indexer::observe_session_runs(
            &c,
            &config.managers[0],
            "0",
            &serde_json::json!({"traj_log_dir":root}),
            &serde_json::json!({"sessions":[{"leased":true,"train_id":"run"}]}),
        )
        .unwrap();
    }
    indexer::observe_session_runs(&c,&config.managers[0],"0",&serde_json::json!({"traj_log_dir":"oss://fixture/root/model"}),&serde_json::json!({"sessions":[{"leased":true,"train_id":"../run"},{"leased":false,"train_id":"idle"}]})).unwrap();
    assert_eq!(
        c.query_row("SELECT count(*) FROM scan_jobs", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn idle_discovery_backs_off_but_changes_and_live_sessions_wake_it() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,due,priority) VALUES('0','','discover',?,100)",
        [mai_dashboard::now() + 3600.],
    )
    .unwrap();
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0','model/idle/','discover',90)",
        [],
    )
    .unwrap();
    for expected in [30., 60., 120., 240., 480., 900., 900.] {
        c.execute("UPDATE scan_jobs SET due=0 WHERE prefix='model/idle/'", [])
            .unwrap();
        indexer::scan_lane(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            "discovery",
            1,
        )
        .await
        .unwrap();
        let due: f64 = c
            .query_row(
                "SELECT due FROM scan_jobs WHERE prefix='model/idle/'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((due - mai_dashboard::now() - expected).abs() < 2.);
    }
    let previous: i64 = c
        .query_row("SELECT unchanged FROM discovery_activity", [], |r| r.get(0))
        .unwrap();
    state.lock().unwrap().fail_list = true;
    c.execute("UPDATE scan_jobs SET due=0 WHERE prefix='model/idle/'", [])
        .unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "discovery",
        1,
    )
    .await
    .unwrap();
    assert_eq!(
        c.query_row("SELECT unchanged FROM discovery_activity", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        previous
    );
    state.lock().unwrap().fail_list = false;
    state.lock().unwrap().objects.insert(
        "model/idle/note.txt".into(),
        ("changed".into(), "v2".into()),
    );
    c.execute("UPDATE scan_jobs SET due=0 WHERE prefix='model/idle/'", [])
        .unwrap();
    indexer::scan_lane(
        path.clone(),
        cfg(),
        url.clone(),
        "token".into(),
        "0".into(),
        "discovery",
        1,
    )
    .await
    .unwrap();
    assert_eq!(
        c.query_row("SELECT unchanged FROM discovery_activity", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    c.execute(
        "UPDATE scan_jobs SET due=? WHERE prefix='model/idle/'",
        [mai_dashboard::now() + 900.],
    )
    .unwrap();
    indexer::observe_session_runs(
        &c,
        &cfg().managers[0],
        "0",
        &serde_json::json!({"traj_log_dir":"oss://fixture/root/model"}),
        &serde_json::json!({"sessions":[{"leased":true,"train_id":"idle"}]}),
    )
    .unwrap();
    for _ in 0..3 {
        let due: f64 = c
            .query_row(
                "SELECT due FROM scan_jobs WHERE prefix='model/idle/' AND kind='discover'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(due <= mai_dashboard::now() + 30.);
        c.execute(
            "UPDATE scan_jobs SET due=0 WHERE prefix='model/idle/' AND kind='discover'",
            [],
        )
        .unwrap();
        indexer::scan_lane(
            path.clone(),
            cfg(),
            url.clone(),
            "token".into(),
            "0".into(),
            "discovery",
            1,
        )
        .await
        .unwrap();
        assert_eq!(
            c.query_row("SELECT count(*) FROM discovery_activity", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    task.abort();
}

#[tokio::test]
async fn idle_discovery_fingerprints_resume_across_pages() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,due,priority) VALUES('0','','discover',?,100)",
        [mai_dashboard::now() + 3600.],
    )
    .unwrap();
    c.execute(
        "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0','model/idle/','discover',90)",
        [],
    )
    .unwrap();
    for i in 0..1001 {
        state.lock().unwrap().objects.insert(
            format!("model/idle/file{i:04}.txt"),
            ("x".into(), "v1".into()),
        );
    }
    for expected in [0, 1] {
        c.execute("UPDATE scan_jobs SET due=0 WHERE prefix='model/idle/'", [])
            .unwrap();
        for _ in 0..2 {
            indexer::scan_lane(
                path.clone(),
                cfg(),
                url.clone(),
                "token".into(),
                "0".into(),
                "discovery",
                1,
            )
            .await
            .unwrap();
        }
        assert_eq!(
            c.query_row("SELECT unchanged FROM discovery_activity", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            expected
        );
    }
    assert_eq!(state.lock().unwrap().list_requests.len(), 4);
    task.abort();
}

#[tokio::test]
async fn active_pages_finish_while_a_list_in_the_same_batch_is_blocked() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let (state, url, task) = gateway().await;
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    let (started, release) = {
        let mut s = state.lock().unwrap();
        for i in 0..3 {
            let prefix = format!("model/run/r00000/task-{i}/20261001_120000_a/");
            c.execute(
                "INSERT INTO scan_jobs(m,prefix,kind,priority) VALUES('0',?,'active',80)",
                [&prefix],
            )
            .unwrap();
            s.objects
                .insert(format!("{prefix}result.txt"), ("1".into(), "v1".into()));
            if i == 0 {
                s.blocked_prefix = Some(prefix);
            }
        }
        (s.list_started.clone(), s.list_release.clone())
    };
    let scan = indexer::scan_lane(path, cfg(), url, "token".into(), "0".into(), "active", 3);
    let verify = async {
        started.notified().await;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if c.query_row("SELECT count(*) FROM attempts", [], |r| r.get::<_, i64>(0))
                    .unwrap()
                    == 2
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("same-batch slow LIST blocked completed attempts");
        release.notify_one();
    };
    let (result, ()) = tokio::join!(scan, verify);
    result.unwrap();
    assert_eq!(
        c.query_row("SELECT count(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        3
    );
    let s = state.lock().unwrap();
    assert_eq!(s.gets, 3);
    assert_eq!(s.list_calls.len(), 3);
    assert!(s.list_calls.values().all(|n| *n == 1));
    task.abort();
}
