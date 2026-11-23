use crate::{
    aggregate::{self, num, text},
    config::Config,
    now,
};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::Semaphore;
static INDEX_MUTATIONS: std::sync::LazyLock<Arc<tokio::sync::RwLock<()>>> =
    std::sync::LazyLock::new(|| Arc::new(tokio::sync::RwLock::new(())));
pub async fn index_mutation() -> tokio::sync::OwnedRwLockReadGuard<()> {
    let _timer = crate::timing::Timer::new("*", "index", "mutation_gate_wait");
    INDEX_MUTATIONS.clone().read_owned().await
}
pub async fn index_publication() -> tokio::sync::OwnedRwLockWriteGuard<()> {
    let _timer = crate::timing::Timer::new("*", "index", "publication_gate_wait");
    INDEX_MUTATIONS.clone().write_owned().await
}
pub fn connect(path: &Path) -> Result<Connection> {
    let c = Connection::open(path)?;
    c.busy_timeout(std::time::Duration::from_secs(5))?;
    c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON; PRAGMA cache_size=-16384;")?;
    Ok(c)
}
pub fn schema(c: &Connection) -> Result<()> {
    c.execute_batch(include_str!("schema.sql"))?;
    if get_meta(c, "identity_schema")?.as_deref() != Some("2") {
        c.execute_batch("DROP TRIGGER IF EXISTS version_trained_insert; DROP TRIGGER IF EXISTS version_trained_update; DROP TRIGGER IF EXISTS version_trained_delete;")?;
        c.execute_batch(include_str!("schema.sql"))?;
        c.execute("INSERT OR IGNORE INTO run_identity SELECT a.m,a.run,json_extract(a.payload,'$.train_id') FROM attempts a JOIN(SELECT m,run,MIN(seq) AS seq FROM attempts GROUP BY m,run) x ON a.seq=x.seq",[])?;
        set_meta(c, "identity_schema", "2")?;
    }
    Ok(())
}
#[derive(Clone)]
pub struct Db {
    path: PathBuf,
    slots: Arc<Semaphore>,
}
impl Db {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            slots: Arc::new(Semaphore::new(8)),
        }
    }
    pub async fn read<T: Send + 'static>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let permit = self.slots.clone().acquire_owned().await?;
        let p = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let _p = permit;
            let c = Connection::open_with_flags(
                p,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            c.busy_timeout(std::time::Duration::from_secs(2))?;
            c.execute_batch("PRAGMA query_only=ON; PRAGMA cache_size=-4096; BEGIN;")?;
            f(&c)
        })
        .await?
    }
}
pub fn get_meta(c: &Connection, key: &str) -> Result<Option<String>> {
    Ok(
        c.query_row("SELECT value FROM meta WHERE key=?", [key], |r| r.get(0))
            .optional()?,
    )
}
pub fn set_meta(c: &Connection, key: &str, value: &str) -> Result<()> {
    c.execute(
        "INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![key, value],
    )?;
    Ok(())
}
pub fn upsert(c: &Connection, m: &str, e: Value) -> Result<()> {
    let e = aggregate::normalize(e);
    let id = text(&e, "id");
    let rk = aggregate::run_key(&e);
    let old: Option<(String, String)> = c
        .query_row(
            "SELECT run,payload FROM attempts WHERE m=? AND id=?",
            params![m, id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let payload = serde_json::to_string(&e)?;
    if old.as_ref().is_some_and(|(_, v)| v == &payload) {
        return Ok(());
    }
    if let Some((run, _)) = old {
        dirty(c, m, &run)?;
    }
    c.execute("INSERT INTO attempts(m,id,run,task,mtime,payload) VALUES(?,?,?,?,?,?) ON CONFLICT(m,id) DO UPDATE SET run=excluded.run,task=excluded.task,mtime=excluded.mtime,payload=excluded.payload",params![m,id,rk,text(&e,"task_id"),num(&e,"mtime"),payload])?;
    c.execute("INSERT INTO legacy_updates(m,id,payload) SELECT ?,?,? WHERE EXISTS(SELECT 1 FROM meta WHERE key='compatibility_publish' AND value='true')",params![m,id,payload])?;
    dirty(c, m, &rk)?;
    Ok(())
}
pub fn remove(c: &Connection, m: &str, id: &str) -> Result<()> {
    let run: Option<String> = c
        .query_row(
            "SELECT run FROM attempts WHERE m=? AND id=?",
            params![m, id],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(run) = run {
        dirty(c, m, &run)?;
        c.execute("INSERT INTO legacy_updates(m,id,payload) SELECT ?,?,NULL WHERE EXISTS(SELECT 1 FROM meta WHERE key='compatibility_publish' AND value='true')",params![m,id])?;
        c.execute("DELETE FROM attempts WHERE m=? AND id=?", params![m, id])?;
    }
    Ok(())
}
fn dirty(c: &Connection, m: &str, run: &str) -> Result<()> {
    c.execute(
        "INSERT OR IGNORE INTO dirty_runs VALUES(?,?)",
        params![m, run],
    )?;
    Ok(())
}
pub fn values(c: &Connection, sql: &str, p: &[&dyn rusqlite::ToSql]) -> Result<Vec<Value>> {
    let mut s = c.prepare(sql)?;
    let rows = s.query_map(p, |r| r.get::<_, String>(0))?;
    rows.map(|r| Ok(serde_json::from_str(&r?)?)).collect()
}
#[allow(clippy::too_many_arguments)]
fn view(
    c: &Connection,
    m: &str,
    kind: &str,
    run: &str,
    rollout: &str,
    id: &str,
    sort: f64,
    data: &Value,
) -> Result<()> {
    c.execute(
        "INSERT INTO views VALUES(?,?,?,?,?,?,?)",
        params![
            m,
            kind,
            run,
            rollout,
            id,
            sort,
            serde_json::to_string(data)?
        ],
    )?;
    Ok(())
}
pub fn rebuild_one(c: &mut Connection, config: &Config) -> Result<bool> {
    let _timer = crate::timing::Timer::new("*", "index", "rebuild");
    rebuild_with_hook(c, config, || Ok(()))
}
fn rebuild_with_hook(
    c: &mut Connection,
    config: &Config,
    before_publish: impl FnOnce() -> Result<()>,
) -> Result<bool> {
    static AGGREGATION: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = AGGREGATION
        .lock()
        .map_err(|_| anyhow::anyhow!("aggregation lock poisoned"))?;
    let urgent:Option<(String,String)>=c.query_row("SELECT d.m,d.run FROM freshness_events f JOIN attempts a ON a.m=f.m AND a.id=f.id JOIN dirty_runs d ON d.m=a.m AND d.run=a.run WHERE f.published_at IS NULL AND f.source_at>? ORDER BY f.source_at LIMIT 1",[now()-86400.],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
    let next: Option<(String, String)> = if urgent.is_some() {
        urgent
    } else {
        c.query_row(
            "SELECT m,run FROM dirty_runs ORDER BY rowid LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
    };
    let Some((m, rk)) = next else {
        return Ok(false);
    };
    let revision = run_version(c, &m, &rk)?;
    let max_age = std::env::var("MAI_GYM_TRAJ_RUN_MAX_AGE_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(30. * 86400.);
    let newest: f64 = c.query_row(
        "SELECT COALESCE(MAX(mtime),0) FROM attempts WHERE m=? AND run=?",
        params![m, rk],
        |r| r.get(0),
    )?;
    let rows = if max_age > 0. && newest < now() - max_age {
        Vec::new()
    } else {
        values(
            c,
            "SELECT payload FROM attempts WHERE m=? AND run=? ORDER BY seq",
            &[&m, &rk],
        )?
    };
    let mgr = config.manager(&m)?;
    let mut trained = BTreeMap::new();
    let train = rows.first().map(|e| text(e, "train_id")).unwrap_or("");
    {
        let mut st = c.prepare("SELECT seg,payload FROM trained WHERE m=? AND train=?")?;
        for r in st.query_map(params![m, train], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (k, v) = r?;
            trained.insert(k, serde_json::from_str(&v)?);
        }
    }
    let old_tasks: BTreeMap<String, Value> = values(
        c,
        "SELECT data FROM views WHERE m=? AND kind='task' AND run=? AND rollout=''",
        &[&m, &rk],
    )?
    .into_iter()
    .map(|v| (text(&v, "task_id").to_owned(), v))
    .collect();
    let max_age = std::env::var("MAI_GYM_TRAJ_RUN_MAX_AGE_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(30. * 86400.);
    let visible = !rows.is_empty()
        && (max_age <= 0. || rows.iter().any(|e| num(e, "mtime") >= now() - max_age));
    let result = visible
        .then(|| aggregate::build_run(rows, mgr.traj_root.as_deref().unwrap_or(""), &trained));
    before_publish()?;
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if run_version(&tx, &m, &rk)? != revision {
        return Ok(true);
    }
    tx.execute("DELETE FROM views WHERE m=? AND run=?", params![m, rk])?;
    tx.execute("DELETE FROM membership WHERE m=? AND run=?", params![m, rk])?;
    let mut affected: BTreeSet<String> = old_tasks.keys().cloned().collect();
    let mut new_tasks = BTreeMap::new();
    if let Some(v) = result {
        view(
            &tx,
            &m,
            "run",
            &rk,
            "",
            &rk,
            num(&v.run, "last_mtime"),
            &v.run,
        )?;
        for b in &v.rollouts {
            view(
                &tx,
                &m,
                "rollout",
                &rk,
                "",
                text(b, "rollout_key"),
                num(b, "t_end"),
                b,
            )?;
        }
        for t in &v.tasks {
            affected.insert(text(t, "task_id").into());
            new_tasks.insert(text(t, "task_id").to_owned(), t.clone());
            view(
                &tx,
                &m,
                "task",
                &rk,
                "",
                text(t, "task_id"),
                num(t, "last_mtime"),
                t,
            )?;
        }
        for (rok, tasks) in &v.tasks_by_rollout {
            for t in tasks {
                view(
                    &tx,
                    &m,
                    "task",
                    &rk,
                    rok,
                    text(t, "task_id"),
                    num(t, "last_mtime"),
                    t,
                )?;
            }
        }
        for (rok, ids) in &v.membership {
            for id in ids {
                tx.execute(
                    "INSERT INTO membership VALUES(?,?,?,?)",
                    params![m, rk, rok, id],
                )?;
            }
        }
        for mut e in v.items {
            e.as_object_mut().unwrap().remove("rollout_key");
            let api = e.to_string();
            tx.execute(
                "UPDATE attempts SET api=? WHERE m=? AND id=? AND api IS NOT ?",
                params![api, m, text(&e, "id"), api],
            )?;
        }
    }

    for task in affected {
        let empty = json!({});
        let old = old_tasks.get(&task).unwrap_or(&empty);
        let new = new_tasks.get(&task).unwrap_or(&empty);
        let previous = values(
            &tx,
            "SELECT data FROM views WHERE m=? AND kind='task_all' AND id=?",
            &[&m, &task],
        )?
        .pop()
        .unwrap_or(json!({"task_id":task}));
        let mut total = previous.clone();
        for k in [
            "count",
            "scored",
            "passed",
            "failed",
            "timeouts",
            "inferred_timeouts",
            "built_in_scored",
            "built_in_passed",
            "judge_scored",
            "judge_passed",
            "verifier_completed",
            "verifier_unscored",
            "verifier_scored",
            "verifier_passed",
            "verifier_failed",
            "verifier_insufficient",
            "verifier_errors",
        ] {
            if !previous[k].is_null() || !new[k].is_null() || !old[k].is_null() {
                let value = num(&previous, k) as i64 + num(new, k) as i64 - num(old, k) as i64;
                anyhow::ensure!(value >= 0, "negative aggregate counter");
                total[k] = json!(value);
            }
        }
        tx.execute(
            "DELETE FROM views WHERE m=? AND kind='task_all' AND id=?",
            params![m, task],
        )?;
        if num(&total, "count") == 0. {
            continue;
        }
        if num(&total, "verifier_completed") == 0. {
            for k in [
                "verifier_completed",
                "verifier_unscored",
                "verifier_scored",
                "verifier_passed",
                "verifier_failed",
                "verifier_insufficient",
                "verifier_errors",
            ] {
                total.as_object_mut().unwrap().remove(k);
            }
        }
        for (mean, n) in [
            ("mean_score", "scored"),
            ("built_in_mean_score", "built_in_scored"),
            ("judge_mean_score", "judge_scored"),
        ] {
            let count = num(&total, n);
            total[mean] = if count == 0. {
                Value::Null
            } else {
                json!(
                    (num(&previous, mean) * num(&previous, n) + num(new, mean) * num(new, n)
                        - num(old, mean) * num(old, n))
                        / count
                )
            };
        }
        let last = if num(old, "last_mtime") >= num(&previous, "last_mtime")
            && num(new, "last_mtime") < num(old, "last_mtime")
        {
            tx.query_row("SELECT COALESCE(MAX(sort),0) FROM views WHERE m=? AND kind='task' AND rollout='' AND id=?",params![m,task],|r|r.get::<_,f64>(0))?
        } else {
            num(&previous, "last_mtime").max(num(new, "last_mtime"))
        };
        total["last_mtime"] = json!(last);
        view(&tx, &m, "task_all", "", "", &task, last, &total)?;
    }
    tx.execute("DELETE FROM dirty_runs WHERE m=? AND run=?", params![m, rk])?;
    tx.execute("UPDATE freshness_events SET published_at=? WHERE m=? AND published_at IS NULL AND id IN(SELECT id FROM attempts WHERE m=? AND run=?)",params![now(),m,m,rk])?;
    set_meta(&tx, &format!("published:{m}"), &now().to_string())?;
    tx.commit()?;
    Ok(true)
}
pub fn install_journal(c: &mut Connection) -> Result<()> {
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS dashboard_rs_journal(seq INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,mgr TEXT NOT NULL,id TEXT NOT NULL,payload TEXT); CREATE TABLE IF NOT EXISTS dashboard_rs_consumers(name TEXT PRIMARY KEY,seq INTEGER NOT NULL);")?;
    for (table, keys) in [
        ("attempts", vec!["mgr", "rel_id"]),
        ("trained", vec!["mgr", "train_id", "seg"]),
    ] {
        let columns: Vec<String> = {
            let mut st = tx.prepare(&format!("PRAGMA table_info({table})"))?;
            st.query_map([], |r| r.get(1))?
                .collect::<rusqlite::Result<_>>()?
        };
        let payload = columns
            .iter()
            .map(|n| format!("'{n}',NEW.{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let id = if table == "attempts" {
            "NEW.rel_id"
        } else {
            "json_array(NEW.train_id,NEW.seg)"
        };
        let old_id = id.replace("NEW.", "OLD.");
        let _ = keys;
        for op in ["INSERT", "UPDATE"] {
            tx.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS dashboard_rs_{table}_{op} AFTER {op} ON {table} BEGIN INSERT INTO dashboard_rs_journal(kind,mgr,id,payload) VALUES('{table}',NEW.mgr,{id},json_object({payload})); END;"))?;
        }
        tx.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS dashboard_rs_{table}_DELETE AFTER DELETE ON {table} BEGIN INSERT INTO dashboard_rs_journal(kind,mgr,id,payload) VALUES('{table}',OLD.mgr,{old_id},NULL); END;"))?;
    }
    tx.commit()?;
    Ok(())
}
fn legacy_value(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let mut v = json!({});
    for (i, n) in row.as_ref().column_names().iter().enumerate() {
        let value = match row.get_ref(i)? {
            rusqlite::types::ValueRef::Null => Value::Null,
            rusqlite::types::ValueRef::Integer(n) => json!(n),
            rusqlite::types::ValueRef::Real(n) => json!(n),
            rusqlite::types::ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
            rusqlite::types::ValueRef::Blob(_) => Value::Null,
        };
        v[*n] = value;
    }
    Ok(v)
}
fn from_legacy(mut v: Value) -> Value {
    v["id"] = v["rel_id"].take();
    v["evaluation"] = v["is_eval"]
        .as_i64()
        .map(|n| json!(n != 0))
        .unwrap_or(Value::Null);
    for k in ["mgr", "rel_id", "is_eval"] {
        v.as_object_mut().unwrap().remove(k);
    }
    v
}
pub fn import_batch(dst: &mut Connection, src: &Connection, cfg: &Config) -> Result<usize> {
    let cursor = get_meta(dst, "import_cursor")?.unwrap_or_default();
    let keys: Vec<String> = if cursor.is_empty() {
        vec![String::new(), String::new()]
    } else {
        serde_json::from_str(&cursor)?
    };
    let mut s = src.prepare(
        "SELECT * FROM attempts WHERE (mgr,rel_id)>(?,?) ORDER BY mgr,rel_id LIMIT 2000",
    )?;
    let rows: Vec<Value> = s
        .query_map(params![keys[0], keys[1]], legacy_value)?
        .collect::<rusqlite::Result<_>>()?;
    let tx = dst.transaction()?;
    for v in &rows {
        if let Some(i) = cfg.managers.iter().position(|m| m.url == text(v, "mgr")) {
            upsert(&tx, &i.to_string(), from_legacy(v.clone()))?;
        }
    }
    if let Some(last) = rows.last() {
        set_meta(
            &tx,
            "import_cursor",
            &json!([last["mgr"], last["rel_id"]]).to_string(),
        )?;
    } else {
        set_meta(&tx, "import_done", "true")?;
    }
    tx.commit()?;
    Ok(rows.len())
}
pub fn replay(dst: &mut Connection, src: &Connection, cfg: &Config) -> Result<usize> {
    let cursor = get_meta(dst, "journal_seq")?
        .unwrap_or_else(|| "0".into())
        .parse::<i64>()?;
    let mut s=src.prepare("SELECT seq,kind,mgr,id,payload FROM dashboard_rs_journal WHERE seq>? ORDER BY seq LIMIT 2000")?;
    let rows: Vec<(i64, String, String, String, Option<String>)> = s
        .query_map([cursor], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let tx = dst.transaction()?;
    for (seq, kind, url, id, payload) in &rows {
        if let Some(m) = cfg.managers.iter().position(|m| &m.url == url) {
            let m = m.to_string();
            if kind == "attempts" {
                if let Some(p) = payload {
                    upsert(&tx, &m, from_legacy(serde_json::from_str(p)?))?;
                } else {
                    remove(&tx, &m, id)?;
                }
            } else {
                let ids: Vec<String> = serde_json::from_str(id)?;
                tx.execute(
                    "DELETE FROM trained WHERE m=? AND train=? AND seg=?",
                    params![m, ids[0], ids[1]],
                )?;
                if let Some(p) = payload {
                    let v: Value = serde_json::from_str(p)?;
                    tx.execute(
                        "INSERT INTO trained VALUES(?,?,?,?)",
                        params![m, ids[0], ids[1], text(&v, "manifest")],
                    )?;
                }
                tx.execute("INSERT OR IGNORE INTO dirty_runs SELECT m,run FROM run_identity WHERE m=? AND train=?",params![m,ids[0]])?;
            }
        }
        set_meta(&tx, "journal_seq", &seq.to_string())?;
    }
    tx.commit()?;
    Ok(rows.len())
}
pub fn import_aux(dst: &mut Connection, src: &Connection, cfg: &Config) -> Result<()> {
    let tx = dst.transaction()?;
    for (i, m) in cfg.managers.iter().enumerate() {
        let key = i.to_string();
        let mut s = src.prepare("SELECT train_id,seg,manifest FROM trained WHERE mgr=?")?;
        for r in s.query_map([&m.url], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (t, s, v) = r?;
            tx.execute(
                "INSERT OR REPLACE INTO trained VALUES(?,?,?,?)",
                params![key, t, s, v],
            )?;
        }
    }
    tx.commit()?;
    sync_runtime(dst, src, cfg)?;
    Ok(())
}
pub fn sync_runtime(dst: &mut Connection, src: &Connection, cfg: &Config) -> Result<()> {
    use std::io::Read;
    let tx = dst.transaction()?;
    for (i, m) in cfg.managers.iter().enumerate() {
        let key = i.to_string();
        let row: Option<(f64, String, Option<Vec<u8>>)> = src
            .query_row(
                "SELECT updated_at,health,sessions FROM dashboard_runtime WHERE mgr=?",
                [&m.url],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        if let Some((t, h, s)) = row {
            let prev: Option<f64> = tx
                .query_row("SELECT updated FROM runtime WHERE m=?", [&key], |r| {
                    r.get(0)
                })
                .optional()?;
            if prev != Some(t) {
                let decode = |b: &[u8]| -> Result<String> {
                    let mut o = String::new();
                    flate2::read::ZlibDecoder::new(b).read_to_string(&mut o)?;
                    Ok(o)
                };
                tx.execute(
                    "INSERT OR REPLACE INTO runtime VALUES(?,?,?,?)",
                    params![
                        key,
                        t,
                        h,
                        s.as_deref()
                            .map(decode)
                            .transpose()?
                            .unwrap_or_else(|| "{}".into())
                    ],
                )?;
            }
        }
        let last: f64 = get_meta(&tx, &format!("history_cursor:{key}"))?
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.)
            .max(now() - 36. * 3600.);
        let mut s =
            src.prepare("SELECT * FROM samples WHERE mgr=? AND t>? ORDER BY t LIMIT 10000")?;
        for r in s.query_map(params![m.url, last], legacy_value)? {
            let mut v = r?;
            v.as_object_mut().unwrap().remove("mgr");
            tx.execute(
                "INSERT OR IGNORE INTO samples VALUES(?,?,?)",
                params![key, num(&v, "t"), v.to_string()],
            )?;
        }
        let latest: f64 = tx.query_row(
            "SELECT COALESCE(MAX(t),0) FROM samples WHERE m=?",
            [&key],
            |r| r.get(0),
        )?;
        set_meta(&tx, &format!("history_cursor:{key}"), &latest.to_string())?;
        if let Some(status) = src
            .query_row(
                "SELECT status FROM dashboard_views WHERE mgr=?",
                [&m.url],
                |r| r.get::<_, String>(0),
            )
            .optional()?
        {
            set_meta(&tx, &format!("legacy_status:{key}"), &status)?;
        }
    }
    tx.execute("DELETE FROM samples WHERE t<?", [now() - 36. * 3600.])?;
    tx.commit()?;
    Ok(())
}
pub fn register_consumer(src: &mut Connection, name: &str, start: Option<i64>) -> Result<i64> {
    let tx = src.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let existing: Option<i64> = tx
        .query_row(
            "SELECT seq FROM dashboard_rs_consumers WHERE name=?",
            [name],
            |r| r.get(0),
        )
        .optional()?;
    let seq = if let Some(s) = existing {
        s
    } else {
        let s: i64 = match start {
            Some(s) => s,
            None => tx.query_row(
                "SELECT COALESCE(MAX(seq),0) FROM dashboard_rs_journal",
                [],
                |r| r.get(0),
            )?,
        };
        tx.execute(
            "INSERT INTO dashboard_rs_consumers VALUES(?,?)",
            params![name, s],
        )?;
        s
    };
    tx.commit()?;
    Ok(seq)
}
pub fn acknowledge(src: &mut Connection, name: &str, seq: i64) -> Result<()> {
    let tx = src.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute(
        "UPDATE dashboard_rs_consumers SET seq=MAX(seq,?) WHERE name=?",
        params![seq, name],
    )?;
    tx.execute("DELETE FROM dashboard_rs_journal WHERE seq IN(SELECT seq FROM dashboard_rs_journal WHERE seq<=(SELECT MIN(seq) FROM dashboard_rs_consumers) ORDER BY seq LIMIT 10000)",[])?;
    tx.commit()?;
    Ok(())
}
pub fn expire_runs(c: &Connection) -> Result<()> {
    c.execute(
        "DELETE FROM freshness_events WHERE observed_at<?",
        [now() - 36. * 3600.],
    )?;
    let age = std::env::var("MAI_GYM_TRAJ_RUN_MAX_AGE_S")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(30. * 86400.);
    if age > 0. {
        c.execute(
            "INSERT OR IGNORE INTO dirty_runs SELECT m,run FROM views WHERE kind='run' AND sort<?",
            [now() - age],
        )?;
    }
    Ok(())
}

pub fn is_busy(error: &anyhow::Error) -> bool {
    error.downcast_ref::<rusqlite::Error>().is_some_and(|e| {
        matches!(e,
        rusqlite::Error::SqliteFailure(code, _) if matches!(code.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked))
    })
}

fn run_version(c: &Connection, m: &str, run: &str) -> Result<i64> {
    Ok(c.query_row(
        "SELECT version FROM run_versions WHERE m=? AND run=?",
        params![m, run],
        |r| r.get(0),
    )
    .optional()?
    .unwrap_or(0))
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    #[test]
    fn late_update_during_aggregation_is_never_erased() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        let mut c = connect(&path).unwrap();
        schema(&c).unwrap();
        let cfg = Config {
            url_base: String::new(),
            managers: vec![crate::config::Manager {
                label: "test".into(),
                url: "http://fixture".into(),
                traj_root: None,
                capacity_mode: None,
                k8s_inventory: None,
            }],
        };
        let e = json!({"id":"run/r00001/task/attempt","train_id":"run","task_id":"task","attempt":"attempt","rollout_id":1,"score":0,"mtime":now(),"created_at":now()});
        upsert(&c, "0", e.clone()).unwrap();
        rebuild_with_hook(&mut c, &cfg, || {
            let other = connect(&path)?;
            let mut e = e.clone();
            e["score"] = json!(1);
            let tx = other.unchecked_transaction()?;
            upsert(&tx, "0", e)?;
            tx.commit()?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            c.query_row("SELECT COUNT(*) FROM views", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert!(rebuild_one(&mut c, &cfg).unwrap());
        let runs = values(&c, "SELECT data FROM views WHERE kind='run'", &[]).unwrap();
        assert_eq!(runs[0]["passed"], 1);
        assert!(!rebuild_one(&mut c, &cfg).unwrap());
    }
}
