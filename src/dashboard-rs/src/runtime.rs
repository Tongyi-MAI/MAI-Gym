use crate::{
    aggregate::{num, text},
    config::{Config, Manager},
    db, now,
};
use anyhow::{Context, Result};
use rusqlite::{OptionalExtension, params};
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, time::Duration};
use tokio::sync::mpsc;
struct Snapshot {
    m: String,
    health: Value,
    sessions: Option<Value>,
    sample: Option<Value>,
}
async fn manager_snapshot(
    client: &reqwest::Client,
    url: &str,
    health_timeout: Duration,
    sessions_timeout: Duration,
) -> (Value, Option<Value>) {
    async fn read(client: &reqwest::Client, url: String, timeout: Duration) -> Result<Value> {
        Ok(client
            .get(url)
            .timeout(timeout)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    let (health, sessions) = tokio::join!(
        read(client, format!("{url}/health"), health_timeout),
        read(client, format!("{url}/sessions"), sessions_timeout)
    );
    (
        health.unwrap_or(json!({"ok":false,"error":"manager unavailable"})),
        sessions.ok(),
    )
}
pub fn capacity(manager: &Manager, mut data: Value) -> Value {
    if manager.capacity_mode.as_deref() != Some("on_demand") {
        return data;
    }
    if data["summary"]["availability_mode"] == "on_demand"
        && !data["summary"]["on_demand_capacity"].is_null()
    {
        return data;
    }
    let sessions = data["sessions"].as_array().cloned().unwrap_or_default();
    let idle = sessions
        .iter()
        .filter(|s| s["status"] == "ready" && s["leased"] != true)
        .count();
    if !data["summary"].is_object() {
        data["summary"] = json!({});
    }
    let summary = &mut data["summary"];
    let target = num(summary, "target").max(0.);
    let total = summary["total"]
        .as_f64()
        .unwrap_or(sessions.len() as f64)
        .max(0.);
    let pending = num(summary, "pending_provisions").max(0.);
    let remaining = (target - total - pending).max(0.);
    for (k, v) in [
        ("acquirable", idle as f64 + remaining),
        ("manager_idle", idle as f64),
        ("on_demand_capacity", remaining),
        ("pending_provisions", pending),
    ] {
        summary[k] = json!(v as u64);
    }
    summary["availability_mode"] = json!("on_demand");
    data
}
async fn poll(
    manager: Manager,
    m: String,
    tx: mpsc::Sender<Snapshot>,
    path: PathBuf,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(45))
        .build()?;
    let mut prev: HashMap<String, f64> = HashMap::new();
    let c = db::connect(&path)?;
    let mut cumulative=c.query_row("SELECT json_extract(data,'$.cum_steps') FROM samples WHERE m=? ORDER BY t DESC LIMIT 1",[&m],|r|r.get::<_,f64>(0)).optional()?.unwrap_or(0.);
    drop(c);
    loop {
        let (health, mut sessions) = manager_snapshot(
            &client,
            manager.url.trim_end_matches('/'),
            Duration::from_secs(10),
            Duration::from_secs(45),
        )
        .await;
        let mut sample = None;
        if let Some(v) = sessions.take() {
            let data = capacity(&manager, v);
            let rows = data["sessions"].as_array().cloned().unwrap_or_default();
            let mut next = HashMap::new();
            let mut counts = HashMap::from([("ready", 0), ("busy", 0), ("invalid", 0)]);
            for row in &rows {
                if let Some(n) = counts.get_mut(text(row, "status")) {
                    *n += 1;
                }
                let sid = text(row, "session_id").to_owned();
                let step = num(row, "step_count");
                if let Some(old) = prev.get(&sid) {
                    cumulative += if step >= *old { step - old } else { step };
                }
                next.insert(sid, step);
            }
            prev = next;
            sample = Some(
                json!({"t":(now()*10.).round()/10.,"total":data["summary"]["total"].as_u64().unwrap_or(rows.len()as u64),"leased":num(&data["summary"],"leased")as u64,"acquirable":num(&data["summary"],"acquirable")as u64,"ready":counts["ready"],"busy":counts["busy"],"invalid":counts["invalid"],"cum_steps":cumulative}),
            );
            sessions = Some(data);
        }
        tx.send(Snapshot {
            m: m.clone(),
            health,
            sessions,
            sample,
        })
        .await
        .context("runtime writer stopped")?;
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
pub async fn run(path: PathBuf, cfg: Config) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<Snapshot>(24);
    for (i, m) in cfg.managers.iter().enumerate() {
        let (tx, path, m) = (tx.clone(), path.clone(), m.clone());
        tokio::spawn(async move {
            while !tx.is_closed() {
                if let Err(e) = poll(m.clone(), i.to_string(), tx.clone(), path.clone()).await {
                    tracing::warn!("runtime poller retrying: {e}");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        });
    }
    drop(tx);
    while let Some(mut s) = rx.recv().await {
        let path = path.clone();
        let manager = cfg.manager(&s.m)?.clone();
        let outcome = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut c = db::connect(&path)?;
            let inventory = db::get_meta(&c, &format!("inventory:{}", s.m))?
                .and_then(|s| serde_json::from_str::<Value>(&s).ok());
            if let (Some(v), Some(inv)) = (&mut s.sessions, inventory) {
                if let Some(summary) = v["summary"].as_object_mut() {
                    summary.extend(inv.as_object().unwrap().clone());
                }
                if let Some(rows) = v["fleet_autoscale"].as_object_mut()
                    && rows.len() == 1
                    && let Some(row) = rows.values_mut().next().and_then(Value::as_object_mut)
                {
                    row.extend(inv.as_object().unwrap().clone());
                }
            }
            let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            let sessions = if let Some(v) = &s.sessions {
                v.to_string()
            } else {
                tx.query_row("SELECT sessions FROM runtime WHERE m=?", [&s.m], |r| {
                    r.get::<_, String>(0)
                })
                .optional()?
                .unwrap_or_else(|| "{}".into())
            };
            tx.execute(
                "INSERT OR REPLACE INTO runtime VALUES(?,?,?,?)",
                params![s.m, now(), s.health.to_string(), sessions],
            )?;
            if let Some(v) = s.sample {
                tx.execute(
                    "INSERT OR REPLACE INTO samples VALUES(?,?,?)",
                    params![s.m, num(&v, "t"), v.to_string()],
                )?;
            }
            tx.execute("DELETE FROM samples WHERE t<?", [now() - 36. * 3600.])?;
            tx.commit()?;
            if let Some(sessions) = &s.sessions {
                let hints = (|| -> Result<()> {
                    let tx =
                        c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    crate::indexer::observe_session_runs(&tx, &manager, &s.m, &s.health, sessions)?;
                    tx.commit()?;
                    Ok(())
                })();
                if hints.is_err() {
                    tracing::warn!(
                        manager = s.m,
                        "live run hints deferred; retained manager snapshot"
                    );
                }
            }
            Ok(())
        })
        .await;
        if !matches!(outcome, Ok(Ok(()))) {
            tracing::warn!("runtime database publication deferred: {outcome:?}");
        }
    }
    Ok(())
}
async fn inventory(config: &Value) -> Result<Value> {
    let valid = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
            && !s.starts_with('-')
    };
    let host = text(config, "ssh_host");
    let ns = text(config, "namespace");
    let pool = text(config, "sandbox_pool");
    anyhow::ensure!(
        valid(host) && valid(ns) && (pool.is_empty() || valid(pool)),
        "invalid inventory target"
    );
    let selector = if pool.is_empty() {
        String::new()
    } else {
        format!(" -l agents.kruise.io/sandbox-pool={pool}")
    };
    let command = format!(
        "timeout --signal=TERM --kill-after=5s 50s kubectl --request-timeout=20s -n {ns} get pods{selector} --chunk-size=500 --no-headers -o custom-columns=PHASE:.status.phase,DELETING:.metadata.deletionTimestamp"
    );
    let out = tokio::time::timeout(
        Duration::from_secs(75),
        tokio::process::Command::new("ssh")
            .args([
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                &command,
            ])
            .kill_on_drop(true)
            .output(),
    )
    .await??;
    anyhow::ensure!(out.status.success(), "inventory command failed");
    let raw = String::from_utf8(out.stdout)?;
    let mut phases: HashMap<String, u64> = HashMap::new();
    let mut terminating = 0;
    for line in raw.lines() {
        let p: Vec<_> = line.split_whitespace().collect();
        anyhow::ensure!(
            p.len() == 2
                && [
                    "Pending",
                    "Running",
                    "Succeeded",
                    "Failed",
                    "Unknown",
                    "<none>"
                ]
                .contains(&p[0]),
            "unexpected inventory response"
        );
        let phase = if p[0] == "<none>" { "Unknown" } else { p[0] };
        *phases.entry(phase.into()).or_default() += 1;
        terminating += u64::from(p[1] != "<none>");
    }
    let live: u64 = phases
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "Failed" | "Succeeded"))
        .map(|(_, v)| v)
        .sum();
    let running = phases.get("Running").copied().unwrap_or(0);
    Ok(
        json!({"live_pod_envs":live,"running_pod_envs":running,"provisioning_pod_envs":live.saturating_sub(running),"terminating_pod_envs":terminating,"pod_env_phase_counts":phases,"pod_env_inventory_observed_at":now(),"pod_env_inventory_error":null,"pod_env_inventory_source":"kubernetes"}),
    )
}
pub async fn inventory_loop(path: PathBuf, cfg: Config) -> Result<()> {
    let mut groups: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for (i, m) in cfg.managers.iter().enumerate() {
        if let Some(c) = &m.k8s_inventory {
            groups.entry(c.to_string()).or_default().push(i.to_string());
        }
    }
    let mut targets: Vec<_> = groups
        .into_iter()
        .map(|(config, keys)| (config, keys, 0.0_f64, 0i32))
        .collect();
    if targets.is_empty() {
        return Ok(());
    }
    loop {
        targets.sort_by(|a, b| a.2.total_cmp(&b.2));
        let (config, keys, due, failures) = &mut targets[0];
        tokio::time::sleep(Duration::from_secs_f64((*due - now()).max(0.))).await;
        let result = inventory(&serde_json::from_str::<Value>(config)?).await;
        let ok = result.is_ok();
        if ok {
            *failures = 0
        } else {
            *failures += 1;
        }
        let (path, keys) = (path.clone(), keys.clone());
        let outcome = tokio::task::spawn_blocking(move || -> Result<()> {
            let c = db::connect(&path)?;
            for key in keys {
                let v = match &result {
                    Ok(v) => v.clone(),
                    Err(_) => {
                        let mut v = db::get_meta(&c, &format!("inventory:{key}"))?
                            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                            .unwrap_or(json!({}));
                        v["pod_env_inventory_error"] = json!("inventory refresh failed");
                        v
                    }
                };
                db::set_meta(&c, &format!("inventory:{key}"), &v.to_string())?;
            }
            Ok(())
        })
        .await;
        if !matches!(outcome, Ok(Ok(()))) {
            tracing::warn!("runtime database publication deferred: {outcome:?}");
        }
        *due = now() + (300. * 2f64.powi((*failures - 1).clamp(0, 10))).min(1800.);
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn slow_sessions_do_not_expire_a_completed_health_response() {
        use axum::{Json, Router, routing::get};
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({"ok":true})) }))
            .route(
                "/sessions",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    Json(json!({"sessions":[{"train_id":"live"}]}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (health, sessions) = manager_snapshot(
            &reqwest::Client::new(),
            &url,
            Duration::from_millis(100),
            Duration::from_secs(2),
        )
        .await;
        assert_eq!(health["ok"], true);
        assert_eq!(sessions.unwrap()["sessions"][0]["train_id"], "live");
        server.abort();
    }
}
