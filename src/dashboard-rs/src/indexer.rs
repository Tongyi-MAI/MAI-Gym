use crate::{
    aggregate::{self, num, text},
    config::Config,
    db, now,
    storage::{Object, Page},
};
use anyhow::{Context, Result};
use rusqlite::{OptionalExtension, params};
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, time::Duration};
#[derive(Clone)]
struct Gateway {
    http: reqwest::Client,
    url: String,
    token: String,
    purpose: &'static str,
}
impl Gateway {
    async fn list(
        &self,
        m: &str,
        path: &str,
        cursor: &str,
        recursive: bool,
        purpose: &str,
    ) -> Result<Page> {
        let _timer = crate::timing::Timer::new(m, purpose, "list");
        Ok(self
            .http
            .get(format!("{}/list", self.url))
            .bearer_auth(&self.token)
            .query(&[
                ("m", m),
                ("path", path),
                ("cursor", cursor),
                ("recursive", if recursive { "true" } else { "false" }),
                ("purpose", purpose),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    async fn get(&self, m: &str, path: &str, etag: &str) -> Result<Option<String>> {
        let _timer = crate::timing::Timer::new(m, self.purpose, "metadata_get");
        let r = self
            .http
            .get(format!("{}/get", self.url))
            .bearer_auth(&self.token)
            .query(&[
                ("m", m),
                ("path", path),
                ("purpose", self.purpose),
                ("etag", etag),
            ])
            .send()
            .await?;
        if r.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(r.error_for_status()?.text().await?))
    }
}
fn attempt_path(path: &str) -> bool {
    let p: Vec<_> = path.trim_end_matches('/').split('/').collect();
    (p.len() == 4 && aggregate::rollout_seg(p[1]))
        || (p.len() == 5 && aggregate::rollout_seg(p[2]))
        || (p.len() == 2
            && p[1].as_bytes().get(8) == Some(&b'_')
            && p[1]
                .get(..8)
                .is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit()))
            && p[1]
                .get(9..15)
                .is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit()))
            && (p[1].len() == 15 || p[1].as_bytes().get(15) == Some(&b'_')))
}
fn discovery_prefix(path: &str) -> bool {
    let p: Vec<_> = path.trim_end_matches('/').split('/').collect();
    p.len() <= 2
        || (p.len() <= 4 && p.get(1).is_some_and(|s| aggregate::rollout_seg(s)))
        || (p.len() <= 5 && p.get(2).is_some_and(|s| aggregate::rollout_seg(s)))
}
fn discovery_priority(prefix: &str, first_inspection: bool) -> i64 {
    if prefix.is_empty() {
        100
    } else if first_inspection || !prefix.trim_end_matches('/').contains('/') {
        95
    } else {
        90
    }
}
fn structured_root(id: &str) -> Option<String> {
    let p: Vec<_> = id.trim_end_matches('/').split('/').collect();
    if p.len() == 4 && aggregate::rollout_seg(p[1]) {
        Some(format!("{}/", p[0]))
    } else if p.len() == 5 && aggregate::rollout_seg(p[2]) {
        Some(format!("{}/{}/", p[0], p[1]))
    } else {
        None
    }
}
fn activate_run_watch(c: &rusqlite::Connection, m: &str, id: &str, written: f64) -> Result<()> {
    let Some(root) = structured_root(id) else {
        return Ok(());
    };
    let expires = written + 21600.;
    if !expires.is_finite() || expires <= now() {
        return Ok(());
    }
    c.execute("INSERT INTO live_run_watches VALUES(?,?,?) ON CONFLICT(m,root) DO UPDATE SET expires=excluded.expires WHERE expires<excluded.expires",params![m,root,expires])?;
    schedule(c, m, &root, "watch", 0., 85)
}

pub fn observe_session_runs(
    c: &rusqlite::Connection,
    manager: &crate::config::Manager,
    m: &str,
    health: &Value,
    sessions: &Value,
) -> Result<()> {
    let Some(base) = manager.traj_root.as_deref() else {
        return Ok(());
    };
    let base = base.trim_end_matches('/');
    let actual = text(health, "traj_log_dir").trim_end_matches('/');
    let relative = if actual == base {
        ""
    } else if let Some(relative) = actual.strip_prefix(&format!("{base}/")) {
        relative
    } else {
        return Ok(());
    };
    let component = |s: &str| {
        !s.is_empty()
            && s != "."
            && s != ".."
            && !s.contains(['/', '\\'])
            && !s.chars().any(char::is_control)
    };
    if !relative.is_empty() && !component(relative) {
        return Ok(());
    }
    let trains: std::collections::BTreeSet<_> = sessions["sessions"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| s["leased"] == true)
        .map(|s| text(s, "train_id"))
        .filter(|s| component(s))
        .collect();
    for train in trains {
        let root = if relative.is_empty() {
            format!("{train}/")
        } else {
            format!("{relative}/{train}/")
        };
        schedule(c, m, &root, "discover", 0., 100)?;
        c.execute("UPDATE scan_jobs SET priority=100,due=MIN(due,?) WHERE m=? AND prefix=? AND kind='discover'",params![now(),m,root])?;
        c.execute("INSERT INTO live_run_watches VALUES(?,?,?) ON CONFLICT(m,root) DO UPDATE SET expires=excluded.expires WHERE expires<excluded.expires",params![m,root,now()+21600.])?;
        schedule(c, m, &root, "watch", 0., 85)?;
    }
    Ok(())
}

fn pending_discoveries(c: &rusqlite::Connection, m: &str) -> Result<i64> {
    Ok(c.query_row("SELECT count(*) FROM scan_jobs s WHERE m=? AND kind='active' AND NOT EXISTS(SELECT 1 FROM attempts a WHERE a.m=s.m AND a.id=rtrim(s.prefix,'/'))", [m], |r|r.get(0))?)
}
fn task_watch(prefix: &str) -> bool {
    let parts: Vec<_> = prefix.trim_end_matches('/').split('/').collect();
    parts.len() >= 3 && aggregate::rollout_seg(parts[parts.len() - 2])
}
fn directory_watch(c: &mut rusqlite::Connection, m: &str, job: &Job, page: Page) -> Result<()> {
    let lease: Option<(String, f64)> = c.query_row(
        "SELECT root,expires FROM live_run_watches WHERE m=? AND substr(?,1,length(root))=root ORDER BY length(root) DESC LIMIT 1",
        params![m,job.scope], |r| Ok((r.get(0)?,r.get(1)?))).optional()?;
    let Some((root, _)) = lease.filter(|(_, expires)| *expires > now()) else {
        c.execute(
            "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='watch'",
            params![m, job.scope],
        )?;
        return Ok(());
    };
    let parts: Vec<_> = job.scope.trim_end_matches('/').split('/').collect();
    let task_scope = task_watch(&job.scope);
    let rollout_scope = parts.last().is_some_and(|p| aggregate::rollout_seg(p));
    let mut tail = job.cursor.clone();
    let mut pending = if task_scope {
        pending_discoveries(c, m)?
    } else {
        0
    };
    let mut blocked = false;
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    for o in &page.objects {
        if !o.dir {
            continue;
        }
        let id = o.path.trim_end_matches('/');
        if attempt_path(id) {
            let known: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM attempts WHERE m=? AND id=?)",
                params![m, id],
                |r| r.get(0),
            )?;
            if !known {
                let queued: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM scan_jobs WHERE m=? AND prefix=? AND kind='active')",params![m,o.path],|r|r.get(0))?;
                if !queued {
                    if pending >= 1024 {
                        blocked = true;
                        break;
                    }
                    schedule(&tx, m, &o.path, "active", 0., 80)?;
                    pending += 1;
                }
            }
            tail = tail.max(o.path.clone());
        } else if !task_scope
            && (rollout_scope || aggregate::rollout_seg(id.rsplit('/').next().unwrap_or("")))
        {
            if !rollout_scope {
                let last: Option<f64> = tx
                    .query_row(
                        "SELECT sort FROM views WHERE m=? AND kind='rollout' AND run=? AND id=?",
                        params![
                            m,
                            root.trim_end_matches('/'),
                            id.rsplit('/').next().unwrap_or("")
                        ],
                        |r| r.get(0),
                    )
                    .optional()?;
                if last.is_some_and(|t| t < now() - 21600.) {
                    continue;
                }
            }
            schedule(&tx, m, &o.path, "watch", 0., 85)?;
        }
    }
    let cursor = if blocked {
        &tail
    } else {
        page.cursor
            .as_deref()
            .unwrap_or(if task_scope { &tail } else { "" })
    };
    tx.execute(
        "UPDATE scan_jobs SET cursor=?,due=?,failures=0 WHERE m=? AND prefix=? AND kind='watch'",
        params![
            cursor,
            if page.cursor.is_some() && !blocked {
                now()
            } else {
                now() + 30.
            },
            m,
            job.scope
        ],
    )?;
    if page.cursor.is_none() && !blocked {
        completed(&tx, m, &job.scope, "watch")?;
    }
    tx.commit()?;
    Ok(())
}
fn descriptor(id: &str, root: &str) -> Value {
    let p: Vec<_> = id.split('/').collect();
    let n = p.len();
    let mut e = json!({"id":id,"task_id":p.get(n.saturating_sub(2)),"attempt":p.last(),"train_id":null,"rollout_id":null,"evaluation":null,"score":null,"reason":null,"judge_reward":null,"error":null,"error_mtime":null,"steps":0,"mtime":0,"created_at":null,"last_write":null,"model":root.trim_end_matches('/').rsplit('/').next(),"suite":null,"benchmark_model":null,"benchmark_run_id":null,"eval_pkg":null});
    if n >= 4 && aggregate::rollout_seg(p[n - 3]) {
        e["train_id"] = json!(p[n - 4]);
        let seg = p[n - 3];
        let body = seg.trim_start_matches("eval_").trim_start_matches('r');
        e["rollout_id"] = json!(body.parse::<i64>().ok());
        e["evaluation"] = json!(seg.starts_with("eval_"));
    }
    if let Some(name) = p.last()
        && let Some(stamp) = name.get(..15)
        && let Ok(dt) = chrono::NaiveDateTime::parse_from_str(stamp, "%Y%m%d_%H%M%S")
    {
        use chrono::TimeZone;
        if let Some(ts) = chrono::Local.from_local_datetime(&dt).earliest() {
            let t = ts.timestamp() as f64;
            e["created_at"] = json!(t);
            e["mtime"] = json!(t);
            e["last_write"] = json!(t);
        }
    }
    aggregate::normalize(e)
}
fn benchmark_metadata(config: &Value) -> Value {
    let benchmark = &config["benchmark"];
    json!({
        "suite": benchmark["suite"],
        "model": if text(benchmark, "model").is_empty() {
            &config["model_name"]
        } else {
            &benchmark["model"]
        },
        "run_id": benchmark["run_id"],
    })
}
async fn update_attempt(
    c: &rusqlite::Connection,
    g: &Gateway,
    m: &str,
    id: &str,
    root: &str,
    objects: &[Object],
    changed: &HashMap<String, bool>,
) -> Result<Value> {
    let mut e = db::values(
        c,
        "SELECT payload FROM attempts WHERE m=? AND id=?",
        &[&m, &id],
    )?
    .into_iter()
    .next()
    .unwrap_or_else(|| descriptor(id, root));
    let old_verifier = aggregate::verifier(text(&e, "reason")).map(|_| e["reason"].clone());
    let by_name: HashMap<_, _> = objects
        .iter()
        .map(|o| (o.path.strip_prefix(&format!("{id}/")).unwrap_or(&o.path), o))
        .collect();
    for filename in [
        "meta.json",
        "result.txt",
        "error.txt",
        "judge_result.txt",
        "verifier_result.json",
    ] {
        let Some(o) = by_name.get(filename) else {
            if filename == "result.txt" {
                e["score"] = Value::Null;
                e["reason"] = Value::Null;
            }
            if filename == "error.txt" {
                e["error"] = Value::Null;
                e["error_mtime"] = Value::Null;
            }
            continue;
        };
        if !changed.get(&o.path).copied().unwrap_or(false) {
            continue;
        }
        let Some(raw) = g.get(m, &o.path, o.etag.as_deref().unwrap_or("")).await? else {
            anyhow::bail!("artifact disappeared during enumeration")
        };
        match filename {
            "meta.json" => {
                let meta: Value = serde_json::from_str(&raw)?;
                if id.split('/').count() == 2 {
                    for k in ["train_id", "created_at", "rollout_id", "evaluation"] {
                        e[k] = meta[k].clone();
                    }
                    e["mtime"] = e["created_at"].clone();
                }
                for key in ["model", "task_id"] {
                    if meta[key].as_str().is_some_and(|name| !name.is_empty()) {
                        e[key] = meta[key].clone();
                    }
                }
            }
            "result.txt" => {
                let (s, r) = aggregate::parse_result(&raw);
                e["score"] = json!(s);
                e["reason"] = json!(r);
            }
            "error.txt" => {
                e["error"] = if raw.trim().is_empty() {
                    Value::Null
                } else {
                    json!(raw.trim())
                };
                e["error_mtime"] = json!(o.modified);
            }
            "judge_result.txt" => {
                e["judge_reward"] = json!(
                    raw.lines()
                        .next()
                        .and_then(|s| s.strip_prefix("verdict:"))
                        .and_then(|s| s.trim().parse::<f64>().ok())
                );
            }
            "verifier_result.json" => {
                if aggregate::verifier(&raw).is_some() {
                    e["reason"] = json!(raw);
                }
            }
            _ => {}
        }
    }
    if by_name.contains_key("verifier_result.json")
        && !changed
            .get(&by_name["verifier_result.json"].path)
            .copied()
            .unwrap_or(false)
        && let Some(reason) = old_verifier
    {
        e["reason"] = reason;
    }
    if !by_name.contains_key("judge_result.txt") {
        e["judge_reward"] = Value::Null;
    }
    let shots: Vec<_> = objects
        .iter()
        .filter(|o| {
            o.path.starts_with(&format!("{id}/screenshots/"))
                && [".png", ".jpg", ".jpeg", ".webp"]
                    .iter()
                    .any(|s| o.path.to_lowercase().ends_with(s))
        })
        .collect();
    e["steps"] = json!(shots.len());
    e["last_write"] = json!(
        shots
            .iter()
            .map(|o| o.modified)
            .reduce(f64::max)
            .unwrap_or(num(&e, "created_at"))
    );
    let rk = aggregate::run_key(&e);
    if rk != aggregate::UNGROUPED {
        let config_path = format!("{rk}/run_config.json");
        if let Some(raw) = db::get_meta(c, &format!("sidecar:{m}:{config_path}"))?
            && let Ok(v) = serde_json::from_str::<Value>(&raw)
        {
            let b = benchmark_metadata(&v);
            for (k, bk) in [
                ("suite", "suite"),
                ("benchmark_model", "model"),
                ("benchmark_run_id", "run_id"),
            ] {
                e[k] = b[bk].clone();
            }
        }
        let manifest = format!("{rk}/eval_manifest.json");
        if let Some(raw) = db::get_meta(c, &format!("sidecar:{m}:{manifest}"))?
            && let Ok(v) = serde_json::from_str::<Value>(&raw)
            && text(&v, "schema").starts_with("eb-eval/")
        {
            for p in v["packages"].as_array().into_iter().flatten() {
                if p["tasks"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|t| t["class_name"] == e["task_id"])
                {
                    e["eval_pkg"] = if p["label"].is_null() {
                        p["set_id"].clone()
                    } else {
                        p["label"].clone()
                    };
                    break;
                }
            }
        }
    }
    Ok(aggregate::normalize(e))
}
fn schedule(
    c: &rusqlite::Connection,
    m: &str,
    prefix: &str,
    kind: &str,
    due: f64,
    priority: i64,
) -> Result<()> {
    if kind == "discover" && !discovery_prefix(prefix) {
        return Ok(());
    }
    let exists: bool = c.query_row(
        "SELECT EXISTS(SELECT 1 FROM scan_jobs WHERE m=? AND prefix=? AND (kind=? OR ?='discover'))",
        params![m, prefix, kind, kind],
        |r| r.get(0),
    )?;
    if !exists {
        let priority = if kind == "discover" {
            discovery_priority(prefix, true)
        } else {
            priority
        };
        let count: i64 = c.query_row("SELECT COUNT(*) FROM scan_jobs", [], |r| r.get(0))?;
        anyhow::ensure!(count < 50000, "bounded discovery queue is full");
        c.execute(
            "INSERT INTO scan_jobs(m,prefix,kind,due,priority) VALUES(?,?,?,?,?)",
            params![m, prefix, kind, due, priority],
        )?;
    }
    if !exists && kind == "reconcile" {
        c.execute("INSERT OR IGNORE INTO scan_seen SELECT m,?,id,0 FROM attempts WHERE m=? AND id>=? AND id<?",params![prefix,m,prefix,format!("{}0",prefix.trim_end_matches('/'))])?;
    }
    Ok(())
}
fn is_metadata(path: &str) -> bool {
    [
        "meta.json",
        "result.txt",
        "error.txt",
        "judge_result.txt",
        "verifier_result.json",
    ]
    .contains(&path.rsplit('/').next().unwrap_or(""))
}
fn completed(c: &rusqlite::Connection, m: &str, prefix: &str, kind: &str) -> Result<()> {
    c.execute(
        "INSERT OR REPLACE INTO scan_completed VALUES(?,?,?,?)",
        params![m, prefix, kind, now()],
    )?;
    let key = format!("scan_cycle:{m}");
    let start = db::get_meta(c, &key)?
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.);
    let pending:bool=c.query_row("SELECT EXISTS(SELECT 1 FROM scan_jobs j WHERE j.m=? AND j.kind<>'active' AND NOT EXISTS(SELECT 1 FROM scan_completed x WHERE x.m=j.m AND x.prefix=j.prefix AND x.kind=j.kind AND x.at>=?))",params![m,start],|r|r.get(0))?;
    if !pending {
        db::set_meta(c, &format!("full_scan:{m}"), &now().to_string())?;
        db::set_meta(c, &key, &now().to_string())?;
    }
    Ok(())
}
fn file_attempt(path: &str) -> Option<String> {
    if let Some((id, _)) = path.split_once("/screenshots/") {
        return attempt_path(id).then(|| id.into());
    }
    let (id, name) = path.rsplit_once('/')?;
    [
        "meta.json",
        "result.txt",
        "error.txt",
        "judge_result.txt",
        "verifier_result.json",
        "traj.json",
    ]
    .contains(&name)
    .then_some(id)
    .filter(|id| attempt_path(id))
    .map(str::to_owned)
}
fn changed(c: &rusqlite::Connection, m: &str, o: &Object) -> Result<bool> {
    let old: Option<(Option<String>, f64, i64)> = c
        .query_row(
            "SELECT etag,mtime,size FROM objects WHERE m=? AND path=?",
            params![m, o.path],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    Ok(old.is_none_or(|v| v.0 != o.etag || v.1 != o.modified || v.2 != o.size as i64))
}
fn record_object(
    c: &rusqlite::Connection,
    m: &str,
    scope: &str,
    generation: i64,
    o: &Object,
) -> Result<()> {
    c.execute("INSERT INTO objects VALUES(?,?,?,?,?,?,?) ON CONFLICT(m,path) DO UPDATE SET etag=excluded.etag,mtime=excluded.mtime,size=excluded.size,scope=excluded.scope,generation=excluded.generation",params![m,o.path,o.etag,o.modified,o.size,scope,generation])?;
    Ok(())
}
async fn sidecar(c: &rusqlite::Connection, g: &Gateway, m: &str, o: &Object) -> Result<()> {
    if !changed(c, m, o)? {
        return Ok(());
    }
    let Some(raw) = g.get(m, &o.path, o.etag.as_deref().unwrap_or("")).await? else {
        anyhow::bail!("sidecar disappeared")
    };
    let v: Value = serde_json::from_str(&raw)?;
    let _mutation = db::index_mutation().await;
    db::set_meta(c, &format!("sidecar:{m}:{}", o.path), &raw)?;
    if o.path.ends_with("/trained.json") && v["tasks"].is_array() {
        let seg = o.path.trim_end_matches("/trained.json");
        let mut p = seg.rsplit('/');
        let rollout = p.next().unwrap_or("");
        let train = p.next().unwrap_or("");
        c.execute(
            "INSERT OR REPLACE INTO trained VALUES(?,?,?,?)",
            params![m, train, rollout, raw],
        )?;
        c.execute(
            "INSERT OR IGNORE INTO dirty_runs SELECT m,run FROM run_identity WHERE m=? AND train=?",
            params![m, train],
        )?;
    }
    let run = o.path.rsplit_once('/').map(|(r, _)| r).unwrap_or("");
    if o.path.ends_with("/run_config.json") || o.path.ends_with("/eval_manifest.json") {
        let rows = if o.path.ends_with("/run_config.json") {
            let b = benchmark_metadata(&v);
            db::values(
                c,
                "SELECT payload FROM attempts WHERE m=? AND run=? AND (json_extract(payload,'$.suite') IS NOT ? OR json_extract(payload,'$.benchmark_model') IS NOT ? OR json_extract(payload,'$.benchmark_run_id') IS NOT ?)",
                &[
                    &m,
                    &run,
                    &b["suite"].as_str(),
                    &b["model"].as_str(),
                    &b["run_id"].as_str(),
                ],
            )?
        } else {
            db::values(
                c,
                "SELECT payload FROM attempts WHERE m=? AND run=?",
                &[&m, &run],
            )?
        };
        for mut e in rows {
            if o.path.ends_with("/run_config.json") {
                let b = benchmark_metadata(&v);
                for (k, bk) in [
                    ("suite", "suite"),
                    ("benchmark_model", "model"),
                    ("benchmark_run_id", "run_id"),
                ] {
                    e[k] = b[bk].clone();
                }
            } else if text(&v, "schema").starts_with("eb-eval/") {
                e["eval_pkg"] = Value::Null;
                for p in v["packages"].as_array().into_iter().flatten() {
                    if p["tasks"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|t| t["class_name"] == e["task_id"])
                    {
                        e["eval_pkg"] = if p["label"].is_null() {
                            p["set_id"].clone()
                        } else {
                            p["label"].clone()
                        };
                        break;
                    }
                }
            }
            db::upsert(c, m, e)?;
        }
    }
    record_object(c, m, run, 0, o)?;
    Ok(())
}
fn remove_sidecar(c: &rusqlite::Connection, m: &str, path: &str) -> Result<()> {
    c.execute(
        "DELETE FROM meta WHERE key=?",
        [format!("sidecar:{m}:{path}")],
    )?;
    c.execute("DELETE FROM objects WHERE m=? AND path=?", params![m, path])?;
    let run = path.rsplit_once('/').map(|(p, _)| p).unwrap_or("");
    if path.ends_with("/trained.json") {
        let mut parts = run.rsplit('/');
        let segment = parts.next().unwrap_or("");
        let train = parts.next().unwrap_or("");
        c.execute(
            "DELETE FROM trained WHERE m=? AND train=? AND seg=?",
            params![m, train, segment],
        )?;
        c.execute(
            "INSERT OR IGNORE INTO dirty_runs SELECT m,run FROM run_identity WHERE m=? AND train=?",
            params![m, train],
        )?;
    } else {
        for mut entry in db::values(
            c,
            "SELECT payload FROM attempts WHERE m=? AND run=?",
            &[&m, &run],
        )? {
            if path.ends_with("/run_config.json") {
                for k in ["suite", "benchmark_model", "benchmark_run_id"] {
                    entry[k] = Value::Null;
                }
            } else if path.ends_with("/eval_manifest.json") {
                entry["eval_pkg"] = Value::Null;
            }
            db::upsert(c, m, entry)?;
        }
    }
    Ok(())
}
#[derive(Clone)]
struct Job {
    scope: String,
    kind: String,
    cursor: String,
    generation: i64,
    failures: i64,
    priority: i64,
}
fn discovery_delay(c: &rusqlite::Connection, m: &str, job: &Job, page: &Page) -> Result<f64> {
    use sha2::{Digest, Sha256};
    let live: bool = c.query_row(
        "SELECT EXISTS(SELECT 1 FROM live_run_watches WHERE m=? AND root=? AND expires>?)",
        params![m, job.scope, now()],
        |r| r.get(0),
    )?;
    if !job.scope.trim_end_matches('/').contains('/') || live {
        c.execute(
            "DELETE FROM discovery_activity WHERE m=? AND prefix=?",
            params![m, job.scope],
        )?;
        return Ok(if page.cursor.is_some() { 0. } else { 30. });
    }
    let (previous, pending, unchanged): (String, String, u32) = c
        .query_row(
            "SELECT fingerprint,pending,unchanged FROM discovery_activity WHERE m=? AND prefix=?",
            params![m, job.scope],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?
        .unwrap_or_default();
    let mut digest = Sha256::new();
    if !job.cursor.is_empty() {
        digest.update(pending.as_bytes());
    }
    digest.update(serde_json::to_vec(&page.objects)?);
    let fingerprint = format!("{:x}", digest.finalize());
    if page.cursor.is_some() {
        c.execute("INSERT INTO discovery_activity(m,prefix,pending) VALUES(?,?,?) ON CONFLICT(m,prefix) DO UPDATE SET pending=excluded.pending",params![m,job.scope,fingerprint])?;
        return Ok(0.);
    }
    let unchanged = if fingerprint == previous {
        (unchanged + 1).min(5)
    } else {
        0
    };
    c.execute("INSERT INTO discovery_activity(m,prefix,fingerprint,unchanged) VALUES(?,?,?,?) ON CONFLICT(m,prefix) DO UPDATE SET fingerprint=excluded.fingerprint,pending='',unchanged=excluded.unchanged",params![m,job.scope,fingerprint,unchanged])?;
    Ok((30. * 2f64.powi(unchanged as i32)).min(900.))
}
async fn page_job(
    c: &mut rusqlite::Connection,
    g: &Gateway,
    cfg: &Config,
    m: &str,
    job: &Job,
    page: Page,
) -> Result<()> {
    let _timer = crate::timing::Timer::new(m, &job.kind, "page_processing");
    if job.kind == "watch" {
        return directory_watch(c, m, job, page);
    }
    let scope = job.scope.as_str();
    let kind = job.kind.as_str();
    let generation = job.generation;
    let listed_at = page.listed_at.unwrap_or_else(now);
    let g = Gateway {
        purpose: if kind == "reconcile" && job.priority < 70 {
            "historical"
        } else if kind == "active" {
            "active"
        } else {
            "discovery"
        },
        ..g.clone()
    };
    let root = cfg
        .manager(m)?
        .traj_root
        .as_deref()
        .context("missing trajectory root")?;
    if kind == "active"
        && (!attempt_path(scope)
            || page
                .objects
                .iter()
                .any(|o| file_attempt(&o.path).is_some_and(|id| id != scope.trim_end_matches('/'))))
    {
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        schedule(&tx, m, scope, "reconcile", 0., 10)?;
        tx.execute(
            "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='active'",
            params![m, scope],
        )?;
        tx.commit()?;
        return Ok(());
    }
    for o in &page.objects {
        if ["run_config.json", "eval_manifest.json", "trained.json"]
            .contains(&o.path.rsplit('/').next().unwrap_or(""))
        {
            sidecar(c, &g, m, o).await?;
            if kind == "reconcile" {
                c.execute(
                    "INSERT OR REPLACE INTO sidecar_seen VALUES(?,?,?,?)",
                    params![m, scope, o.path, generation],
                )?;
            }
        }
    }
    if kind == "attempts" {
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let tail_key = format!("attempt_tail:{m}:{scope}");
        let mut tail = db::get_meta(&tx, &tail_key)?.unwrap_or_default();
        for object in &page.objects {
            let id = object.path.trim_end_matches('/');
            if object.dir && attempt_path(id) {
                if object.path > tail {
                    tail = object.path.clone();
                }
                let known: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM attempts WHERE m=? AND id=?)",
                    params![m, id],
                    |r| r.get(0),
                )?;
                if !known {
                    schedule(&tx, m, &object.path, "active", 0., 70)?;
                }
            }
        }
        db::set_meta(&tx, &tail_key, &tail)?;
        tx.execute(
            "UPDATE scan_jobs SET cursor=?,due=?,failures=0 WHERE m=? AND prefix=? AND kind=?",
            params![
                page.cursor.as_deref().unwrap_or(&tail),
                if page.cursor.is_some() {
                    now()
                } else {
                    now() + 30.
                },
                m,
                scope,
                kind
            ],
        )?;
        if page.cursor.is_none() {
            completed(&tx, m, scope, kind)?;
            db::set_meta(&tx, &format!("discovered:{m}"), &now().to_string())?;
        }
        tx.commit()?;
        return Ok(());
    }
    if kind == "discover" {
        let namespace = page.objects.iter().any(|o| {
            o.dir
                && (aggregate::rollout_seg(
                    o.path
                        .trim_end_matches('/')
                        .rsplit('/')
                        .next()
                        .unwrap_or(""),
                ) || attempt_path(&o.path))
        });
        let leaf = page.objects.iter().any(|o| {
            !o.dir && file_attempt(&o.path).as_deref() == Some(scope.trim_end_matches('/'))
        });
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        if namespace || leaf {
            tx.execute(
                "DELETE FROM discovery_activity WHERE m=? AND prefix=?",
                params![m, scope],
            )?;
            if namespace
                && scope.trim_end_matches('/').split('/').count() == 1
                && page.objects.iter().any(|o| o.dir && attempt_path(&o.path))
            {
                schedule(&tx, m, scope, "attempts", 0., 85)?;
            }
            let known_namespace: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM attempts WHERE m=? AND id>=? AND id<?)",
                params![m, scope, format!("{}0", scope.trim_end_matches('/'))],
                |r| r.get(0),
            )?;
            schedule(
                &tx,
                m,
                scope,
                if leaf && !namespace {
                    "active"
                } else {
                    "reconcile"
                },
                0.,
                if leaf && !namespace {
                    80
                } else if !known_namespace {
                    70
                } else {
                    10
                },
            )?;
            tx.execute(
                "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='discover'",
                params![m, scope],
            )?;
        } else {
            for o in &page.objects {
                if o.dir && o.path != scope && !o.path.contains("distributed_payloads/") {
                    let classified: bool = tx.query_row(
                        "SELECT EXISTS(SELECT 1 FROM scan_jobs WHERE m=? AND prefix=? AND kind IN ('reconcile','attempts','active'))",
                        params![m,o.path], |r| r.get(0),
                    )?;
                    if !classified {
                        schedule(&tx, m, &o.path, "discover", 0., 90)?;
                    }
                }
            }
            let delay = discovery_delay(&tx, m, job, &page)?;
            tx.execute(
                "UPDATE scan_jobs SET cursor=?,due=?,failures=0,priority=? WHERE m=? AND prefix=? AND kind=?",
                params![
                    page.cursor.as_deref().unwrap_or(""),
                    now() + delay,
                    if page.cursor.is_some() {
                        job.priority
                    } else {
                        discovery_priority(scope, false)
                    },
                    m,
                    scope,
                    kind
                ],
            )?;
        }
        if page.cursor.is_none() {
            db::set_meta(&tx, &format!("discovered:{m}"), &now().to_string())?;
            completed(&tx, m, scope, kind)?;
        }
        tx.commit()?;
        return Ok(());
    }
    c.execute(
        "INSERT OR IGNORE INTO scan_started VALUES(?,?,?,?)",
        params![m, scope, generation, listed_at],
    )?;
    {
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        for o in &page.objects {
            if !o.dir && file_attempt(&o.path).is_some() {
                tx.execute("INSERT INTO stage_observed VALUES(?,?,?,?,?) ON CONFLICT(m,scope,generation,id) DO UPDATE SET at=MIN(at,excluded.at)",params![m,scope,generation,file_attempt(&o.path).unwrap(),listed_at])?;
                tx.execute(
                    "INSERT OR REPLACE INTO scan_stage VALUES(?,?,?,?,?)",
                    params![m, scope, generation, o.path, serde_json::to_string(o)?],
                )?;
            }
        }
        tx.commit()?;
    }
    let staged: Vec<Object> = db::values(
        c,
        "SELECT data FROM scan_stage WHERE m=? AND scope=? AND generation=? ORDER BY path",
        &[&m, &scope, &generation],
    )?
    .into_iter()
    .map(serde_json::from_value)
    .collect::<std::result::Result<_, _>>()?;
    anyhow::ensure!(
        staged.len() <= 100000,
        "single attempt exceeds bounded staging capacity"
    );
    let mut groups: std::collections::BTreeMap<String, Vec<Object>> =
        std::collections::BTreeMap::new();
    for o in staged {
        if let Some(id) = file_attempt(&o.path) {
            groups.entry(id).or_default().push(o);
        }
    }
    let tail = page.cursor.as_deref().and_then(file_attempt);
    for (id, objects) in groups {
        if page.cursor.is_some() && tail.as_deref() == Some(&id) {
            continue;
        }
        let observed_at: f64 = c.query_row(
            "SELECT at FROM stage_observed WHERE m=? AND scope=? AND generation=? AND id=?",
            params![m, scope, generation, id],
            |r| r.get(0),
        )?;
        let mut delta = HashMap::new();
        for o in &objects {
            if is_metadata(&o.path) {
                delta.insert(o.path.clone(), changed(c, m, o)?);
            }
        }
        let entry = update_attempt(c, &g, m, &id, root, &objects, &delta).await?;
        let hot = entry["score"].is_null()
            && entry["evaluation_terminal"] != true
            && text(&entry, "error").is_empty()
            && (now() - num(&entry, "last_write") < 1800.
                || now() - num(&entry, "created_at") < 21600.);
        let terminal = entry["score"].is_number()
            || entry["evaluation_terminal"] == true
            || !text(&entry, "error").is_empty();
        let terminal_at = objects
            .iter()
            .filter(|o| {
                [
                    "result.txt",
                    "judge_result.txt",
                    "verifier_result.json",
                    "error.txt",
                ]
                .contains(&o.path.rsplit('/').next().unwrap_or(""))
            })
            .map(|o| o.modified)
            .fold(0., f64::max);
        let run_key = aggregate::run_key(&entry);
        let source_at = objects
            .iter()
            .map(|o| o.modified)
            .filter(|t| t.is_finite() && *t > 0.)
            .reduce(f64::min);
        let _mutation = db::index_mutation().await;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let previous: Option<f64> = tx
            .query_row(
                "SELECT at FROM attempt_observed WHERE m=? AND id=?",
                params![m, id],
                |r| r.get(0),
            )
            .optional()?;
        if previous.is_some_and(|at| at > observed_at) {
            if kind == "reconcile" {
                tx.execute(
                    "INSERT OR REPLACE INTO scan_seen VALUES(?,?,?,?)",
                    params![m, scope, id, generation],
                )?;
            }
            for o in &objects {
                tx.execute(
                    "DELETE FROM scan_stage WHERE m=? AND scope=? AND generation=? AND path=?",
                    params![m, scope, generation, o.path],
                )?;
            }
            tx.execute(
                "DELETE FROM stage_observed WHERE m=? AND scope=? AND generation=? AND id=?",
                params![m, scope, generation, id],
            )?;
            tx.commit()?;
            continue;
        }
        tx.execute(
            "INSERT OR REPLACE INTO attempt_observed VALUES(?,?,?)",
            params![m, id, observed_at],
        )?;
        tx.execute(
            "DELETE FROM stage_observed WHERE m=? AND scope=? AND generation=? AND id=?",
            params![m, scope, generation, id],
        )?;
        let known: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM attempts WHERE m=? AND run=?)",
            params![m, run_key],
            |r| r.get(0),
        )?;
        if !known && let Some(source_at) = source_at {
            tx.execute(
                "INSERT OR IGNORE INTO freshness_events VALUES(?,?,'new_run_oss',?,?,NULL)",
                params![m, id, source_at, observed_at],
            )?;
        }
        if previous.is_none()
            && let Some(source_at) = source_at
        {
            tx.execute(
                "INSERT OR IGNORE INTO freshness_events VALUES(?,?,'attempt_discovery',?,?,NULL)",
                params![m, id, source_at, observed_at],
            )?;
        }
        if kind == "active" && terminal && terminal_at > 0. {
            tx.execute(
                "INSERT OR IGNORE INTO freshness_events VALUES(?,?,'terminal',?,?,NULL)",
                params![m, id, terminal_at, now()],
            )?;
        }
        db::upsert(&tx, m, entry)?;
        activate_run_watch(
            &tx,
            m,
            &id,
            objects.iter().map(|o| o.modified).fold(0., f64::max),
        )?;
        tx.execute("DELETE FROM objects WHERE m=? AND scope=?", params![m, id])?;
        for o in &objects {
            if is_metadata(&o.path) {
                record_object(&tx, m, &id, generation, o)?;
            }
            tx.execute(
                "DELETE FROM scan_stage WHERE m=? AND scope=? AND generation=? AND path=?",
                params![m, scope, generation, o.path],
            )?;
        }
        if kind == "reconcile" {
            tx.execute("INSERT INTO scan_seen VALUES(?,?,?,?) ON CONFLICT(m,scope,id) DO UPDATE SET generation=excluded.generation",params![m,scope,id,generation])?;
        }
        if hot {
            schedule(&tx, m, &format!("{id}/"), "active", now() + 15., 80)?;
        } else {
            tx.execute(
                "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='active'",
                params![m, format!("{id}/")],
            )?;
        }
        tx.commit()?;
    }
    let _mutation = db::index_mutation().await;
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if page.cursor.is_none() {
        if kind == "reconcile" {
            let cycle_start: f64 = tx.query_row(
                "SELECT at FROM scan_started WHERE m=? AND scope=? AND generation=?",
                params![m, scope, generation],
                |r| r.get(0),
            )?;
            let missing: Vec<String> = {
                let mut s = tx.prepare(
                    "SELECT path FROM sidecar_seen WHERE m=? AND scope=? AND generation<>?",
                )?;
                s.query_map(params![m, scope, generation], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for path in missing {
                let newer: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM objects WHERE m=? AND path=? AND mtime>?)",
                    params![m, path, cycle_start],
                    |r| r.get(0),
                )?;
                if newer {
                    tx.execute(
                        "UPDATE sidecar_seen SET generation=? WHERE m=? AND scope=? AND path=?",
                        params![generation, m, scope, path],
                    )?;
                    continue;
                }
                remove_sidecar(&tx, m, &path)?;
            }
            tx.execute(
                "DELETE FROM sidecar_seen WHERE m=? AND scope=? AND generation<>?",
                params![m, scope, generation],
            )?;
            let stale: Vec<String> = {
                let mut s =
                    tx.prepare("SELECT id FROM scan_seen WHERE m=? AND scope=? AND generation<>?")?;
                s.query_map(params![m, scope, generation], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for id in stale {
                let newer: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM attempt_observed WHERE m=? AND id=? AND at>?)",
                    params![m, id, cycle_start],
                    |r| r.get(0),
                )?;
                if newer {
                    tx.execute(
                        "UPDATE scan_seen SET generation=? WHERE m=? AND scope=? AND id=?",
                        params![generation, m, scope, id],
                    )?;
                    continue;
                }
                db::remove(&tx, m, &id)?;
                tx.execute(
                    "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='active'",
                    params![m, format!("{id}/")],
                )?;
            }
            tx.execute(
                "DELETE FROM scan_seen WHERE m=? AND scope=? AND generation<>?",
                params![m, scope, generation],
            )?;
            db::set_meta(&tx, &format!("reconciled:{m}:{scope}"), &now().to_string())?;
        }
        tx.execute("UPDATE scan_jobs SET cursor='',generation=generation+1,due=?,failures=0,priority=CASE WHEN kind='reconcile' THEN 10 ELSE priority END WHERE m=? AND prefix=? AND kind=?",params![now()+if kind=="active"{15.}else{3600.},m,scope,kind])?;
        tx.execute(
            "DELETE FROM scan_started WHERE m=? AND scope=? AND generation=?",
            params![m, scope, generation],
        )?;
        if kind != "active" {
            completed(&tx, m, scope, kind)?;
        }
    } else {
        tx.execute(
            "UPDATE scan_jobs SET cursor=?,due=?,failures=0 WHERE m=? AND prefix=? AND kind=?",
            params![page.cursor, now(), m, scope, kind],
        )?;
    }
    tx.commit()?;
    Ok(())
}
pub async fn scan(
    path: PathBuf,
    cfg: Config,
    url: String,
    token: String,
    m: String,
    prefix: String,
    max_pages: usize,
) -> Result<()> {
    scan_work(
        path,
        cfg,
        url,
        token,
        m,
        prefix,
        max_pages,
        Work {
            publish: true,
            lane: "all",
        },
    )
    .await
}
struct Work {
    publish: bool,
    lane: &'static str,
}
pub const ACTIVE_LANES: [&str; 4] = ["active:0", "active:1", "active:2", "active:3"];
pub async fn scan_lane(
    path: PathBuf,
    cfg: Config,
    url: String,
    token: String,
    m: String,
    lane: &'static str,
    max_pages: usize,
) -> Result<()> {
    scan_work(
        path,
        cfg,
        url,
        token,
        m,
        String::new(),
        max_pages,
        Work {
            publish: false,
            lane,
        },
    )
    .await
}
#[allow(clippy::too_many_arguments)]
async fn scan_work(
    path: PathBuf,
    cfg: Config,
    url: String,
    token: String,
    m: String,
    prefix: String,
    max_pages: usize,
    work: Work,
) -> Result<()> {
    if !prefix.is_empty() {
        crate::storage::safe(prefix.trim_end_matches('/'))?;
    }
    let prefix = if prefix.is_empty() {
        prefix
    } else {
        format!("{}/", prefix.trim_end_matches('/'))
    };
    let mut c = db::connect(&path)?;
    if work.publish {
        db::schema(&c)?;
    }
    let g = Gateway {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(40))
            .build()?,
        url,
        token,
        purpose: "indexing",
    };
    if c.query_row(
        "SELECT COUNT(*) FROM scan_jobs WHERE m=? AND prefix=?",
        params![m, prefix],
        |r| r.get::<_, i64>(0),
    )? == 0
    {
        schedule(
            &c,
            &m,
            &prefix,
            if attempt_path(&prefix) {
                "active"
            } else {
                "discover"
            },
            0.,
            100,
        )?;
    }
    let mut consumed = 0;
    let (lane, shard) = if let Some(shard) = work.lane.strip_prefix("active:") {
        let shard: i64 = shard.parse()?;
        anyhow::ensure!((0..4).contains(&shard), "invalid active shard");
        ("active", shard)
    } else {
        (work.lane, -1)
    };
    while consumed < max_pages {
        let budget = (max_pages - consumed).min(if shard >= 0 { 3 } else { 12 });
        let mut jobs = Vec::new();
        let mut selected = std::collections::HashSet::new();
        let watch_blocked =
            matches!(lane, "watch" | "all" | "priority") && pending_discoveries(&c, &m)? >= 1024;
        let quotas = if lane == "active" {
            vec![
                ("active", (budget * 2).div_ceil(3), 1),
                ("active", (budget / 3).max(1), 0),
                ("active", budget, -1),
            ]
        } else {
            vec![
                ("discover", 2, -1),
                ("attempts", 2, -1),
                ("active", 6, -1),
                ("reconcile", 2, -1),
                ("", 12, -1),
            ]
        };
        for (kind, quota, known) in quotas {
            if jobs.len() >= budget {
                break;
            }
            let mut st=c.prepare("SELECT prefix,kind,cursor,generation,failures,priority FROM scan_jobs WHERE m=?1 AND (prefix=?2 OR substr(prefix,1,length(?2))=?2) AND due<=?3 AND (?4='' OR kind=?4) AND (?6<0 OR rowid%4=?6) AND (?5='all' OR (?5='priority' AND (kind<>'reconcile' OR priority>=70)) OR (?5='active' AND kind='active') OR (?5='discovery' AND (kind='discover' OR (kind='reconcile' AND priority>=70))) OR (?5='attempts' AND kind='attempts') OR (?5='watch' AND kind='watch') OR (?5='historical' AND kind='reconcile' AND priority<70)) AND (?7=0 OR kind<>'watch' OR EXISTS(SELECT 1 FROM live_run_watches w WHERE w.m=scan_jobs.m AND substr(scan_jobs.prefix,1,length(w.root))=w.root AND length(scan_jobs.prefix)-length(replace(scan_jobs.prefix,'/',''))<=length(w.root)-length(replace(w.root,'/',''))+1)) AND (?8<0 OR EXISTS(SELECT 1 FROM attempts a WHERE a.m=scan_jobs.m AND a.id=rtrim(scan_jobs.prefix,'/'))=?8) ORDER BY priority DESC,due,prefix LIMIT 24")?;
            let candidates = st.query_map(
                params![m, prefix, now(), kind, lane, shard, watch_blocked, known],
                |r| {
                    Ok(Job {
                        scope: r.get(0)?,
                        kind: r.get(1)?,
                        cursor: r.get(2)?,
                        generation: r.get(3)?,
                        failures: r.get(4)?,
                        priority: r.get(5)?,
                    })
                },
            )?;
            let mut n = 0;
            for job in candidates {
                let job = job?;
                if job.kind == "watch" {
                    let live: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM live_run_watches WHERE m=? AND expires>? AND substr(?,1,length(root))=root)", params![m,now(),job.scope], |r|r.get(0))?;
                    if !live {
                        c.execute(
                            "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='watch'",
                            params![m, job.scope],
                        )?;
                        continue;
                    }
                }
                if selected.insert((job.scope.clone(), job.kind.clone())) {
                    jobs.push(job);
                    n += 1;
                }
                if n >= quota || jobs.len() >= budget {
                    break;
                }
            }
        }
        if jobs.is_empty() {
            break;
        }
        let outcomes = if lane == "active" {
            futures::future::join_all(jobs.iter().map(|job| async {
                let page = g.list(&m, &job.scope, &job.cursor, true, "active").await?;
                let ready = std::time::Instant::now();
                let mut connection = db::connect(&path)?;
                connection.execute_batch("PRAGMA cache_size=-2048;")?;
                crate::timing::record(&m, "active", "listed_page_wait", ready.elapsed());
                page_job(&mut connection, &g, &cfg, &m, job, page).await
            }))
            .await
        } else {
            let pages = futures::future::join_all(jobs.iter().map(|j| async {
                let page = g
                    .list(
                        &m,
                        &j.scope,
                        &j.cursor,
                        j.kind != "discover" && j.kind != "attempts" && j.kind != "watch",
                        if j.kind == "active" {
                            "active"
                        } else if j.kind == "discover"
                            || j.kind == "attempts"
                            || j.kind == "watch"
                            || (j.kind == "reconcile" && j.priority >= 70)
                        {
                            "discovery"
                        } else {
                            "historical"
                        },
                    )
                    .await;
                (page, std::time::Instant::now())
            }))
            .await;
            let mut outcomes = Vec::with_capacity(jobs.len());
            for (job, (page, ready)) in jobs.iter().zip(pages) {
                crate::timing::record(&m, &job.kind, "listed_page_wait", ready.elapsed());
                outcomes.push(match page {
                    Ok(page) => page_job(&mut c, &g, &cfg, &m, job, page).await,
                    Err(e) => Err(e),
                });
            }
            outcomes
        };
        for (job, outcome) in jobs.into_iter().zip(outcomes) {
            consumed += 1;
            if let Err(e) = outcome {
                c.execute("UPDATE scan_jobs SET failures=failures+1,due=? WHERE m=? AND prefix=? AND kind=?",params![now()+5.*2f64.powi((job.failures as i32).min(8)),m,job.scope,job.kind])?;
                tracing::warn!(
                    manager = m,
                    kind = job.kind,
                    failures = job.failures,
                    "index work deferred: {e}"
                );
            }
        }
        if work.publish {
            let dbpath = path.clone();
            let config = cfg.clone();
            tokio::task::spawn_blocking(move || -> Result<()> {
                let mut c = db::connect(&dbpath)?;
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while std::time::Instant::now() < deadline && db::rebuild_one(&mut c, &config)? {}
                Ok(())
            })
            .await??;
        }
    }

    Ok(())
}
pub fn seed(c: &rusqlite::Connection, cfg: &Config) -> Result<()> {
    if db::get_meta(c, "live_watch_seeded")?.as_deref() != Some("1") {
        for (i, manager) in cfg.managers.iter().enumerate() {
            if manager.traj_root.is_none() {
                continue;
            }
            let m = i.to_string();
            let mut st = c.prepare(
                "SELECT MIN(id),MAX(mtime) FROM attempts WHERE m=? AND mtime>? GROUP BY run",
            )?;
            let roots = st
                .query_map(params![m, now() - 21600.], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for (id, written) in roots {
                activate_run_watch(c, &m, &id, written)?;
            }
        }
        db::set_meta(c, "live_watch_seeded", "1")?;
    }
    if db::get_meta(c, "discovery_priority_version")?.as_deref() != Some("1") {
        let tx = c.unchecked_transaction()?;
        tx.execute("UPDATE scan_jobs SET priority=CASE WHEN prefix='' THEN 100 WHEN instr(rtrim(prefix,'/'),'/')=0 OR NOT EXISTS(SELECT 1 FROM scan_completed x WHERE x.m=scan_jobs.m AND x.prefix=scan_jobs.prefix AND x.kind='discover') THEN 95 ELSE 90 END WHERE kind='discover'", [])?;
        db::set_meta(&tx, "discovery_priority_version", "1")?;
        tx.commit()?;
    }
    if db::get_meta(c, "namespace_discovery_version")?.as_deref() != Some("1") {
        let jobs: Vec<(String, String)> = {
            let mut s = c.prepare("SELECT m,prefix FROM scan_jobs WHERE kind='discover'")?;
            s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?
        };
        let tx = c.unchecked_transaction()?;
        for (m, prefix) in jobs {
            if !discovery_prefix(&prefix) {
                tx.execute(
                    "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='discover'",
                    params![m, prefix],
                )?;
            }
        }
        db::set_meta(&tx, "namespace_discovery_version", "1")?;
        tx.commit()?;
    }
    if db::get_meta(c, "active_layout_version")?.as_deref() != Some("2") {
        let jobs: Vec<(String, String)> = {
            let mut st = c.prepare("SELECT m,prefix FROM scan_jobs WHERE kind='active'")?;
            st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?
        };
        for (m, prefix) in jobs {
            if !attempt_path(&prefix) {
                schedule(c, &m, &prefix, "reconcile", 0., 10)?;
                c.execute(
                    "DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind='active'",
                    params![m, prefix],
                )?;
            }
        }
        c.execute("UPDATE scan_jobs SET priority=CASE WHEN EXISTS(SELECT 1 FROM attempts a WHERE a.m=scan_jobs.m AND a.id=rtrim(scan_jobs.prefix,'/')) THEN 80 ELSE 70 END WHERE kind='active' AND priority=95",[])?;
        db::set_meta(c, "active_layout_version", "2")?;
    }
    if db::get_meta(c, "attempt_watch_version")?.as_deref() != Some("2") {
        let jobs: Vec<(String, String)> = {
            let mut st = c.prepare("SELECT m,prefix FROM scan_jobs WHERE kind='attempts'")?;
            st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?
        };
        for (m, prefix) in jobs {
            let model_parent: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM scan_jobs WHERE m=? AND kind='reconcile' AND prefix>? AND prefix<?)",params![m,prefix,format!("{}0",prefix.trim_end_matches('/'))],|r|r.get(0))?;
            if model_parent {
                c.execute("DELETE FROM scan_jobs WHERE m=? AND prefix=? AND kind IN ('attempts','reconcile')",params![m,prefix])?;
                for table in [
                    "scan_stage",
                    "stage_observed",
                    "scan_started",
                    "scan_seen",
                    "sidecar_seen",
                ] {
                    c.execute(
                        &format!("DELETE FROM {table} WHERE m=? AND scope=?"),
                        params![m, prefix],
                    )?;
                }
                schedule(c, &m, &prefix, "discover", 0., 90)?;
                continue;
            }
            let last: Option<String> = c.query_row(
                "SELECT MAX(id) FROM attempts WHERE m=? AND id>=? AND id<?",
                params![m, prefix, format!("{}0", prefix.trim_end_matches('/'))],
                |r| r.get(0),
            )?;
            let tail = last.map(|id| format!("{id}/")).unwrap_or_default();
            db::set_meta(c, &format!("attempt_tail:{m}:{prefix}"), &tail)?;
            c.execute(
                "UPDATE scan_jobs SET cursor=?,due=0 WHERE m=? AND prefix=? AND kind='attempts'",
                params![tail, m, prefix],
            )?;
        }
        db::set_meta(c, "attempt_watch_version", "2")?;
    }
    if db::get_meta(c, "discovery_seeded")?.as_deref() == Some("true") {
        return Ok(());
    }
    let mut namespaces = std::collections::BTreeSet::new();
    {
        let mut s = c.prepare("SELECT m,MIN(id) FROM attempts GROUP BY m,run")?;
        for row in s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))? {
            let (m, id) = row?;
            let p: Vec<_> = id.split('/').collect();
            let prefix = if p.len() == 2 || (p.len() == 4 && aggregate::rollout_seg(p[1])) {
                format!("{}/", p[0])
            } else if p.len() == 5 && aggregate::rollout_seg(p[2]) {
                format!("{}/{}/", p[0], p[1])
            } else {
                continue;
            };
            namespaces.insert((m, prefix, p.len() == 2));
        }
    }
    for (i, (m, prefix, old_layout)) in namespaces.into_iter().enumerate() {
        if old_layout {
            schedule(c, &m, &prefix, "attempts", 0., 85)?;
        }
        schedule(
            c,
            &m,
            &prefix,
            "reconcile",
            now() + 60. + (i % 3600) as f64,
            10,
        )?;
    }
    for (i, m) in cfg.managers.iter().enumerate() {
        if m.traj_root.is_some() {
            schedule(c, &i.to_string(), "", "discover", 0., 100)?;
        }
    }
    let mut s=c.prepare("SELECT m,id FROM attempts WHERE json_extract(payload,'$.score') IS NULL AND COALESCE(json_extract(payload,'$.error'),'')='' AND COALESCE(json_extract(payload,'$.evaluation_terminal'),0)=0 AND (json_extract(payload,'$.last_write')>? OR json_extract(payload,'$.created_at')>?) LIMIT 50001")?;
    let rows: Vec<(String, String)> = s
        .query_map(params![now() - 1800., now() - 21600.], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    anyhow::ensure!(rows.len() <= 40000, "active queue exceeds validated bound");
    for (m, id) in rows {
        schedule(c, &m, &format!("{id}/"), "active", 0., 80)?;
    }
    db::set_meta(c, "discovery_seeded", "true")?;
    db::set_meta(c, "attempt_watch_version", "0")?;
    seed(c, cfg)
}
