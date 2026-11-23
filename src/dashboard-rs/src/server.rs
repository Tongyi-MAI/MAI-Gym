macro_rules! ensure_api {
    ($condition:expr, $message:expr) => {
        if !$condition {
            return Err(ApiError(anyhow::anyhow!($message)));
        }
    };
}
use crate::{
    aggregate::{self, num, text},
    config::Config,
    db::{self, Db},
    now, shards,
    storage::Storage,
};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use rusqlite::{OptionalExtension, params};
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Arc};
#[derive(Clone)]
pub struct App {
    pub db: Db,
    pub cfg: Arc<Config>,
    pub storage: Arc<Storage>,
    pub http: reqwest::Client,
    pub allow_admin: bool,
}
pub struct ApiError(anyhow::Error);
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        tracing::warn!(error=%self.0,"request failed");
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"request failed","detail":self.0.to_string()})),
        )
            .into_response()
    }
}
type Q = HashMap<String, String>;
fn key(a: &App, q: &Q) -> Result<String> {
    let m = q.get("m").cloned().unwrap_or_else(|| "0".into());
    a.cfg.manager(&m)?;
    Ok(m)
}
fn arg<'a>(q: &'a Q, k: &str) -> &'a str {
    q.get(k)
        .map(String::as_str)
        .filter(|s| *s != "__all__")
        .unwrap_or("")
}
fn number(q: &Q, k: &str, d: usize) -> usize {
    arg(q, k).parse().unwrap_or(d)
}
fn info(c: &rusqlite::Connection, cfg: &Config, m: &str) -> Result<Value> {
    let manager = cfg.manager(m)?;
    let mut status = db::get_meta(c, &format!("legacy_status:{m}"))?
        .and_then(|v| serde_json::from_str::<Value>(&v).ok())
        .unwrap_or(json!({}));
    status["root"] = json!(manager.traj_root);
    status["mode"] = json!(if manager
        .traj_root
        .as_deref()
        .is_some_and(|r| r.starts_with("oss://"))
    {
        "oss"
    } else if manager.traj_root.is_some() {
        "local"
    } else {
        "unavailable"
    });
    status["num_attempts"] = json!(c.query_row(
        "SELECT COUNT(*) FROM attempts WHERE m=?",
        [m],
        |r| r.get::<_, i64>(0)
    )?);
    status["num_runs"] = json!(c.query_row(
        "SELECT COUNT(*) FROM views WHERE m=? AND kind='run'",
        [m],
        |r| r.get::<_, i64>(0)
    )?);
    for (k, prefix) in [
        ("last_publication", "published"),
        ("last_discovery", "discovered"),
        ("full_scan_completed", "full_scan"),
    ] {
        status[k] = db::get_meta(c, &format!("{prefix}:{m}"))?
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(Value::Null, |n| json!(n));
    }
    Ok(status)
}
fn runtime(c: &rusqlite::Connection, m: &str) -> Result<(Value, Value, f64)> {
    let r: Option<(String, String, f64)> = c
        .query_row(
            "SELECT health,sessions,updated FROM runtime WHERE m=?",
            [m],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    if let Some((h, s, t)) = r {
        let mut h: Value = serde_json::from_str(&h)?;
        if now() - t > 30. {
            h["stale"] = json!(true);
            h["snapshot_age_seconds"] = json!(now() - t);
        }
        Ok((h, serde_json::from_str(&s)?, t))
    } else {
        Ok((
            json!({"ok":false,"stale":true}),
            json!({"summary":{},"sessions":[],"by_device":{}}),
            0.,
        ))
    }
}
fn health_only(c: &rusqlite::Connection, m: &str) -> Result<Value> {
    let row: Option<(String, f64)> = c
        .query_row("SELECT health,updated FROM runtime WHERE m=?", [m], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    let Some((raw, t)) = row else {
        return Ok(json!({"ok":false,"stale":true}));
    };
    let mut h: Value = serde_json::from_str(&raw)?;
    if now() - t > 30. {
        h["stale"] = json!(true);
        h["snapshot_age_seconds"] = json!(now() - t);
    }
    Ok(h)
}
fn history(c: &rusqlite::Connection, m: &str, since: f64) -> Result<Value> {
    Ok(
        json!({"samples":db::values(c,"SELECT data FROM samples WHERE m=? AND t>? ORDER BY t",&[&m,&since])?,"interval":5,"now":now()}),
    )
}
fn paginate_array(mut data: Value, field: &str, q: &Q) -> Result<Value> {
    if !q.contains_key("limit") {
        return Ok(data);
    }
    let limit = number(q, "limit", 50).clamp(1, 200);
    let offset = number(q, "cursor", 0);
    let items = data[field].as_array().cloned().unwrap_or_default();
    data["total"] = json!(items.len());
    data["next_cursor"] = if offset + limit < items.len() {
        json!((offset + limit).to_string())
    } else {
        Value::Null
    };
    data[field] = json!(
        items
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>()
    );
    Ok(data)
}
pub fn router(a: App) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/ready", get(ready))
        .route("/api/managers", get(managers))
        .route("/api/overview", get(overview))
        .route("/api/health", get(health))
        .route("/api/sessions", get(sessions))
        .route("/api/history", get(hist))
        .route("/api/traj/info", get(traj_info))
        .route("/api/traj/runs", get(runs))
        .route("/api/traj/rollouts", get(rollouts))
        .route("/api/traj/tasks", get(tasks))
        .route("/api/traj/list", get(attempts))
        .route("/api/traj/detail", get(detail))
        .route("/api/traj/screenshot", get(screenshot))
        .route("/api/traj/results", get(results))
        .route("/api/benchmarks", get(benchmarks))
        .route("/api/benchmarks/suite", get(suite))
        .route("/api/admin/{*operation}", post(admin))
        .layer(tower_http::compression::CompressionLayer::new())
        .with_state(a)
}
async fn home(State(a): State<App>) -> Html<String> {
    let base = serde_json::to_string(&a.cfg.url_base).unwrap();
    Html(include_str!("../static/index.html").replacen(
        "<head>",
        &format!("<head><script>window.__DASH_BASE__={base};</script>"),
        1,
    ))
}
async fn ready(State(a): State<App>) -> Result<Response, ApiError> {
    let v=a.db.read(|c|Ok(json!({"ready":db::get_meta(c,"initial_publication_done")?.as_deref()==Some("true"),"pending_runs":c.query_row("SELECT COUNT(*) FROM dirty_runs",[],|r|r.get::<_,i64>(0))?}))).await?;
    Ok((
        if v["ready"] == true {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(v),
    )
        .into_response())
}
async fn managers(State(a): State<App>) -> Json<Value> {
    Json(json!(
        a.cfg
            .managers
            .iter()
            .enumerate()
            .map(|(i, m)| json!({"key":i.to_string(),"label":m.label,"url":m.url}))
            .collect::<Vec<_>>()
    ))
}
async fn health(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    Ok(Json(a.db.read(move |c| Ok(runtime(c, &m)?.0)).await?))
}
async fn sessions(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    Ok(Json(
        a.db.read(move |c| paginate_array(runtime(c, &m)?.1, "sessions", &q))
            .await?,
    ))
}
async fn hist(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    let since = arg(&q, "since").parse().unwrap_or(0.);
    Ok(Json(a.db.read(move |c| history(c, &m, since)).await?))
}
async fn overview(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    let cfg = a.cfg.clone();
    let since = arg(&q, "since").parse().unwrap_or(0.);
    Ok(Json(a.db.read(move|c|{let(h,s,t)=runtime(c,&m)?;let mut health=json!({});for i in 0..cfg.managers.len(){health[i.to_string()]=health_only(c,&i.to_string())?;}let mut pq=q.clone();pq.entry("limit".into()).or_insert("50".into());Ok(json!({"manager":m,"health":h,"managers":health,"sessions":paginate_array(s,"sessions",&pq)?,"history":history(c,&m,since)?,"snapshot_at":t,"freshness":{"last_publication":db::get_meta(c,&format!("published:{m}"))?,"last_discovery":db::get_meta(c,&format!("discovered:{m}"))?,"full_scan_completed":db::get_meta(c,&format!("full_scan:{m}"))?}}))}).await?))
}
async fn traj_info(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    let cfg = a.cfg.clone();
    Ok(Json(a.db.read(move |c| info(c, &cfg, &m)).await?))
}
fn view_page(c: &rusqlite::Connection, m: &str, kind: &str, q: &Q, field: &str) -> Result<Value> {
    if kind == "run" && arg(q, "merge_shards") == "1" {
        let mut page = virtual_page(shards::runs(c, m)?, m, kind, q, field)?;
        for run in page["runs"].as_array_mut().unwrap() {
            shards::hydrate_pass_metrics(c, m, run)?;
        }
        return Ok(page);
    }
    if kind != "run"
        && let Some(members) = shards::resolve(c, m, arg(q, "run"))?
    {
        let rollout = if kind == "task" {
            arg(q, "rollout")
        } else {
            ""
        };
        let mut value = virtual_page(
            shards::views(c, m, kind, rollout, &members)?,
            m,
            kind,
            q,
            field,
        )?;
        value["shards"] = json!(members);
        return Ok(value);
    }
    let run = if kind == "run" { "" } else { arg(q, "run") };
    let rollout = if kind == "task" {
        arg(q, "rollout")
    } else {
        ""
    };
    let filter = arg(q, "task").to_lowercase();
    let scope = json!([m, kind, run, rollout, filter]).to_string();
    let mut sort = f64::MAX;
    let mut id = String::new();
    if !arg(q, "cursor").is_empty() {
        let v: Value = serde_json::from_str(arg(q, "cursor"))?;
        anyhow::ensure!(v[0] == scope, "cursor does not match query");
        sort = v[1].as_f64().ok_or_else(|| anyhow::anyhow!("bad cursor"))?;
        id = v[2].as_str().unwrap_or("").into();
    }
    let paged = q.contains_key("limit");
    let limit = if paged {
        number(q, "limit", 50).clamp(1, 200) as i64
    } else {
        -1
    };
    let selector = if kind == "run" {
        "m=?1 AND kind=?2"
    } else {
        "m=?1 AND kind=?2 AND run=?3 AND rollout=?4"
    };
    let sql = format!(
        "SELECT data FROM views WHERE {selector} AND (?5='' OR instr(lower(id),?5)>0) AND (sort<?6 OR (sort=?6 AND id>?7)) ORDER BY sort DESC,id LIMIT ?8"
    );
    let mut s = c.prepare(&sql)?;
    let entries: Vec<Value> = s
        .query_map(
            params![
                m,
                kind,
                run,
                rollout,
                filter,
                sort,
                id,
                if paged { limit + 1 } else { -1 }
            ],
            |r| r.get::<_, String>(0),
        )?
        .map(|r| Ok(serde_json::from_str(&r?)?))
        .collect::<Result<_>>()?;
    let has_more = paged && entries.len() > limit as usize;
    let mut entries = entries;
    if has_more {
        entries.pop();
    }
    let total:i64=c.query_row(&format!("SELECT COALESCE(SUM(json_extract(data,'$.count')),0) FROM views WHERE {selector} AND (?5='' OR instr(lower(id),?5)>0)"),params![m,kind,run,rollout,filter],|r|r.get(0))?;
    let cursor = if has_more {
        let last = entries.last().unwrap();
        let sk = if kind == "rollout" {
            "t_end"
        } else {
            "last_mtime"
        };
        let ik = if kind == "run" {
            "run_key"
        } else if kind == "rollout" {
            "rollout_key"
        } else {
            "task_id"
        };
        json!(json!([scope, num(last, sk), text(last, ik)]).to_string())
    } else {
        Value::Null
    };
    let mut v = json!({field:entries});
    if field != "runs" {
        v["total_attempts"] = json!(total);
    }
    if paged {
        v["next_cursor"] = cursor;
    }
    Ok(v)
}
fn virtual_page(mut rows: Vec<Value>, m: &str, kind: &str, q: &Q, field: &str) -> Result<Value> {
    let id_field = match kind {
        "run" => "run_key",
        "rollout" => "rollout_key",
        _ => "task_id",
    };
    let sort_field = if kind == "rollout" {
        "t_end"
    } else {
        "last_mtime"
    };
    let filter = arg(q, "task").to_lowercase();
    rows.retain(|r| {
        filter.is_empty()
            || text(r, id_field).to_lowercase().contains(&filter)
            || r["shards"].as_array().is_some_and(|ss| {
                ss.iter()
                    .any(|s| text(s, "run_key").to_lowercase().contains(&filter))
            })
    });
    rows.sort_by(|a, b| {
        num(b, sort_field)
            .total_cmp(&num(a, sort_field))
            .then_with(|| text(a, id_field).cmp(text(b, id_field)))
    });
    let total: u64 = rows.iter().map(|r| r["count"].as_u64().unwrap_or(0)).sum();
    let total_runs = rows.len();
    let scope = json!([m, kind, arg(q, "run"), arg(q, "rollout"), filter, "merged"]).to_string();
    if !arg(q, "cursor").is_empty() {
        let cursor: Value = serde_json::from_str(arg(q, "cursor"))?;
        anyhow::ensure!(cursor[0] == scope, "cursor does not match query");
        let sort = cursor[1]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("bad cursor"))?;
        let id = cursor[2]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("bad cursor"))?;
        rows.retain(|r| {
            num(r, sort_field) < sort || (num(r, sort_field) == sort && text(r, id_field) > id)
        });
    }
    let mut value = json!({});
    if q.contains_key("limit") {
        let limit = number(q, "limit", 50).clamp(1, 200);
        let more = rows.len() > limit;
        rows.truncate(limit);
        value["next_cursor"] = if more {
            let last = rows.last().unwrap();
            json!(json!([scope, num(last, sort_field), text(last, id_field)]).to_string())
        } else {
            Value::Null
        };
    }
    value[field] = json!(rows);
    if field == "runs" {
        value["total_runs"] = json!(total_runs);
        value["total_attempts"] = json!(total);
    } else {
        value["total_attempts"] = json!(total);
    }
    Ok(value)
}
async fn runs(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    Ok(Json(
        a.db.read(move |c| view_page(c, &m, "run", &q, "runs"))
            .await?,
    ))
}
async fn rollouts(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    Ok(Json(
        a.db.read(move |c| view_page(c, &m, "rollout", &q, "rollouts"))
            .await?,
    ))
}
async fn tasks(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    Ok(Json(
        a.db.read(move |c| {
            view_page(
                c,
                &m,
                if arg(&q, "run").is_empty() {
                    "task_all"
                } else {
                    "task"
                },
                &q,
                "tasks",
            )
        })
        .await?,
    ))
}
fn item_query(c: &rusqlite::Connection, m: &str, q: &Q, all: bool) -> Result<Value> {
    let run = arg(q, "run");
    let members = shards::resolve(c, m, run)?;
    let keys = members.as_ref().map(|rows| shards::member_keys(rows));
    let selector = keys.as_deref().unwrap_or(run);
    let rollout = arg(q, "rollout");
    let task = arg(q, "task").to_lowercase();
    let exact = q.get("task_exact");
    let page = number(q, "page", 0);
    let size = number(q, "page_size", 50).clamp(1, 200);
    let mut pred = String::from(
        "a.m=?1 AND a.api IS NOT NULL AND EXISTS(SELECT 1 FROM views v WHERE v.m=a.m AND v.kind='run' AND v.run=a.run)",
    );
    pred.push_str(if members.is_some() {
        " AND a.run IN(SELECT value FROM json_each(?2))"
    } else if run.is_empty() {
        " AND ?2=''"
    } else {
        " AND a.run=?2"
    });
    pred.push_str(if rollout.is_empty() {" AND ?3=''"} else {" AND EXISTS(SELECT 1 FROM membership x WHERE x.m=a.m AND x.run=a.run AND x.rollout=?3 AND x.id=a.id)"});
    pred.push_str(if exact.is_some() {
        " AND a.task=?4 AND ?5 IS NOT NULL"
    } else {
        " AND ?4 IS NULL AND (?5='' OR instr(lower(a.task),?5)>0)"
    });
    let total: i64 = if members.is_some()
        && task.is_empty()
        && exact.is_none()
        && rollout.is_empty()
    {
        c.query_row("SELECT COALESCE(SUM(json_extract(data,'$.count')),0) FROM views WHERE m=? AND kind='run' AND run IN(SELECT value FROM json_each(?))", params![m, selector], |r| r.get(0))?
    } else if task.is_empty() && exact.is_none() && rollout.is_empty() {
        c.query_row("SELECT COALESCE(SUM(json_extract(data,'$.count')),0) FROM views WHERE m=? AND kind='run' AND (?='' OR run=?)",params![m,run,run],|r|r.get(0))?
    } else {
        c.query_row(
            &format!("SELECT COUNT(*) FROM attempts a WHERE {pred}"),
            params![m, selector, rollout, exact, task],
            |r| r.get(0),
        )?
    };
    let mut s=c.prepare(&format!("SELECT COALESCE(a.api,a.payload) FROM attempts a WHERE {pred} ORDER BY a.mtime DESC,a.id LIMIT ?6 OFFSET ?7"))?;
    let mut items: Vec<Value> = s
        .query_map(
            params![
                m,
                selector,
                rollout,
                exact,
                task,
                if all { -1 } else { size as i64 },
                if all {
                    0
                } else {
                    page.saturating_mul(size) as i64
                }
            ],
            |r| r.get::<_, String>(0),
        )?
        .map(|r| Ok(serde_json::from_str(&r?)?))
        .collect::<Result<_>>()?;
    if members.is_some() {
        for item in &mut items {
            item["source_run"] = json!(aggregate::run_key(item));
            if let Some((_, index)) = shards::shard_name(text(item, "train_id")) {
                item["source_shard"] = json!(index);
            }
        }
    }
    Ok(json!({"items":items,"total":total,"page":page,"page_size":size}))
}
async fn attempts(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    Ok(Json(
        a.db.read(move |c| item_query(c, &m, &q, false)).await?,
    ))
}
async fn benchmark_value(a: &App) -> Result<Value> {
    let cfg = a.cfg.clone();
    a.db.read(move |c| {
        let mut rows = Vec::new();
        for (i, m) in cfg.managers.iter().enumerate() {
            for v in db::values(
                c,
                "SELECT data FROM views WHERE m=? AND kind='run'",
                &[&i.to_string()],
            )? {
                rows.push((i.to_string(), m.label.clone(), v));
            }
        }
        Ok(aggregate::benchmarks(rows))
    })
    .await
}
async fn benchmarks(State(a): State<App>) -> Result<Json<Value>, ApiError> {
    let v = benchmark_value(&a).await?;
    let mut suites:Vec<_>=v.as_object().unwrap().iter().map(|(s,v)|json!({"suite":s,"n_models":v["models"].as_array().unwrap().len(),"last_mtime":v["models"].as_array().unwrap().iter().map(|m|num(m,"last_mtime")).fold(0.,f64::max)})).collect();
    suites.sort_by(|a, b| num(b, "last_mtime").total_cmp(&num(a, "last_mtime")));
    Ok(Json(json!({"suites":suites})))
}
async fn suite(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let v = benchmark_value(&a).await?;
    let s = arg(&q, "s");
    Ok(Json(if v[s].is_null() {
        json!({"suite":s,"models":[]})
    } else {
        v[s].clone()
    }))
}
async fn detail(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    let id = arg(&q, "id").to_string();
    crate::storage::safe(&id)?;
    let mm = m.clone();
    let ii = id.clone();
    let e =
        a.db.read(move |c| {
            let mut entry = db::values(
                c,
                "SELECT COALESCE(api,payload) FROM attempts WHERE m=? AND id=?",
                &[&mm, &ii],
            )?
            .into_iter()
            .next()
            .unwrap_or(json!({}));
            let mut etags = serde_json::Map::new();
            let mut st = c.prepare(
                "SELECT path,etag FROM objects WHERE m=? AND scope=? AND etag IS NOT NULL",
            )?;
            for row in st.query_map(rusqlite::params![mm, ii], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })? {
                let (path, etag) = row?;
                if let Some(name) = path.rsplit('/').next() {
                    etags.insert(name.into(), json!(etag));
                }
            }
            entry["_artifact_etags"] = Value::Object(etags);
            Ok(entry)
        })
        .await?;
    Ok(Json(a.storage.detail_cached(&m, &id, e).await?))
}
async fn screenshot(State(a): State<App>, Query(q): Query<Q>) -> Result<Response, ApiError> {
    let m = key(&a, &q)?;
    let id = arg(&q, "id");
    let name = arg(&q, "name");
    crate::storage::safe(id)?;
    crate::storage::safe(name)?;
    ensure_api!(
        !name.contains('/')
            && ["png", "jpg", "jpeg", "webp"]
                .iter()
                .any(|ext| name.to_lowercase().ends_with(&format!(".{ext}"))),
        "bad screenshot name"
    );
    let blob = a
        .storage
        .get(&m, &format!("{id}/screenshots/{name}"), "interactive")
        .await?;
    let Some(blob) = blob else {
        return Ok(StatusCode::NOT_FOUND.into_response());
    };
    let b = blob.bytes;
    let media = if b.starts_with(b"\x89PNG") {
        "image/png"
    } else if b.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if b.starts_with(b"RIFF") && b.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else {
        "application/octet-stream"
    };
    Ok((
        [
            ("content-type", media),
            ("cache-control", "private, max-age=5"),
        ],
        b,
    )
        .into_response())
}
async fn results(State(a): State<App>, Query(q): Query<Q>) -> Result<Json<Value>, ApiError> {
    let m = key(&a, &q)?;
    let root = a.cfg.manager(&m)?.traj_root.clone().unwrap_or_default();
    Ok(Json(a.db.read(move|c|{let wanted=arg(&q,"run");let runs=if wanted.starts_with(shards::PREFIX){shards::runs(c,&m)?}else{db::values(c,"SELECT data FROM views WHERE m=? AND kind='run' ORDER BY sort DESC,id",&[&m])?};let mut output=Vec::new();for r in runs{let rk=text(&r,"run_key");if !wanted.is_empty() && if wanted.starts_with(shards::PREFIX) {wanted!=rk} else {!wanted.split(',').any(|x|x.trim()==rk)} {continue;}let mut rq=q.clone();rq.insert("run".into(),rk.into());rq.remove("limit");let ro=view_page(c,&m,"rollout",&rq,"rollouts")?;let mut buckets=Vec::new();for b in ro["rollouts"].as_array().unwrap(){let rok=text(b,"rollout_key");if !arg(&q,"rollout").is_empty()&&arg(&q,"rollout")!=rok{continue;}rq.insert("rollout".into(),rok.into());let at=item_query(c,&m,&rq,true)?;let mut grouped:std::collections::BTreeMap<String,Vec<&Value>>=std::collections::BTreeMap::new();for e in at["items"].as_array().unwrap(){grouped.entry(text(e,"task_id").into()).or_default().push(e);}let tasks:Vec<_>=grouped.into_iter().map(|(t,es)|{let winner=es.iter().find(|e|e["score"].as_f64().is_some_and(aggregate::success));let scored=es.iter().any(|e|e["score"].is_number());let rep=winner.copied().unwrap_or(es[0]);json!({"task_id":t,"success":if winner.is_some(){json!(true)}else if scored{json!(false)}else{Value::Null},"traj_path":format!("{}/{}",root.trim_end_matches('/'),text(rep,"id"))})}).collect();buckets.push(json!({"rollout_key":rok,"label":b["label"],"is_eval":b["is_eval"],"tasks":tasks}));}output.push(json!({"run_key":rk,"train_id":r["train_id"],"model_slug":r["model_slug"],"rollouts":buckets}));}Ok(json!({"manager":m,"runs":output}))}).await?))
}
async fn admin(
    State(a): State<App>,
    Path(operation): Path<String>,
    Query(mut q): Query<Q>,
) -> Result<Response, ApiError> {
    if !a.allow_admin {
        return Ok((
            StatusCode::FORBIDDEN,
            "admin operations disabled in shadow service",
        )
            .into_response());
    }
    let m = key(&a, &q)?;
    let parts: Vec<_> = operation.split('/').collect();
    ensure_api!(
        (parts.len() == 1 && parts[0] == "scale")
            || (parts.len() == 2
                && ["force_release", "force_invalidate"].contains(&parts[0])
                && parts[1]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')),
        "unknown admin operation"
    );
    q.remove("m");
    let r = a
        .http
        .post(format!(
            "{}/admin/{operation}",
            a.cfg.manager(&m)?.url.trim_end_matches('/')
        ))
        .query(&q)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await?;
    let status = StatusCode::from_u16(r.status().as_u16())?;
    Ok((
        status,
        [("content-type", "application/json")],
        r.bytes().await?,
    )
        .into_response())
}
