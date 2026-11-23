use crate::{
    aggregate::{self, text},
    config::Config,
    now,
};
use anyhow::{Context, Result};
use bytes::Bytes;
use opendal::{
    ErrorKind, Operator,
    raw::{
        Access, OpList, OpRead,
        oio::{List as _, Read as _},
    },
};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, Weak},
    time::Duration,
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
#[derive(Clone)]
pub struct Blob {
    pub bytes: Bytes,
    pub etag: Option<String>,
    pub modified: Option<f64>,
}
#[derive(Clone)]
struct Cached {
    blob: Option<Blob>,
    expires: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    pub path: String,
    pub etag: Option<String>,
    pub modified: f64,
    pub size: u64,
    pub dir: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Page {
    #[serde(default)]
    pub listed_at: Option<f64>,
    pub objects: Vec<Object>,
    pub cursor: Option<String>,
}
#[derive(Default, Clone, Serialize)]
pub struct Metric {
    pub requests: u64,
    pub bytes: u64,
    pub not_found: u64,
    pub errors: u64,
    pub not_modified: u64,
}
#[derive(Clone)]
struct DetailCached {
    value: Value,
    checked_at: f64,
}
type DiskRow = (Option<Vec<u8>>, Option<String>, Option<f64>, f64);
pub struct Storage {
    ops: HashMap<String, Operator>,
    memory: moka::future::Cache<String, Cached>,
    disk: Arc<Mutex<rusqlite::Connection>>,
    locks: Mutex<HashMap<String, Weak<AsyncMutex<()>>>>,
    slots: Semaphore,
    background_slots: Semaphore,
    discovery_slots: Semaphore,
    metrics: Mutex<HashMap<String, Metric>>,
    lists: moka::future::Cache<String, (f64, Page)>,
    failures: moka::future::Cache<String, ()>,
    finalized: moka::future::Cache<String, bool>,
    details: moka::future::Cache<String, DetailCached>,
    detail_refresh: Arc<Semaphore>,
}
pub fn safe(id: &str) -> Result<()> {
    anyhow::ensure!(
        !id.is_empty()
            && !id.starts_with('/')
            && !id.contains('\\')
            && !id.contains('\0')
            && id
                .split('/')
                .all(|p| !p.is_empty() && p != ".." && p != "."),
        "invalid storage path"
    );
    Ok(())
}
impl Storage {
    pub fn new(cfg: &Config, cache: PathBuf) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut ops = HashMap::new();
        for (i, m) in cfg.managers.iter().enumerate() {
            if let Some(root) = &m.traj_root {
                let op = if let Some(r) = root.strip_prefix("oss://") {
                    let (bucket, prefix) = r.split_once('/').unwrap_or((r, ""));
                    let b = opendal::services::Oss::default()
                        .bucket(bucket)
                        .root(prefix)
                        .endpoint(
                            &std::env::var("MAI_GYM_OSS_ENDPOINT")
                                .context("OSS endpoint required")?,
                        )
                        .access_key_id(
                            &std::env::var("MAI_GYM_OSS_ACCESS_KEY_ID")
                                .context("OSS access key required")?,
                        )
                        .access_key_secret(
                            &std::env::var("MAI_GYM_OSS_ACCESS_KEY_SECRET")
                                .context("OSS secret required")?,
                        );
                    Operator::new(b)?.finish()
                } else {
                    Operator::new(opendal::services::Fs::default().root(root))?.finish()
                };
                ops.insert(i.to_string(), op);
            }
        }
        Self::with_operators(ops, cache)
    }
    pub fn with_operators(ops: HashMap<String, Operator>, cache: PathBuf) -> Result<Self> {
        let c = crate::db::connect(&cache)?;
        c.execute_batch("CREATE TABLE IF NOT EXISTS artifacts(key TEXT PRIMARY KEY,body BLOB,etag TEXT,modified REAL,expires REAL NOT NULL,used REAL NOT NULL); CREATE INDEX IF NOT EXISTS artifacts_used ON artifacts(used); CREATE TABLE IF NOT EXISTS metrics(key TEXT PRIMARY KEY,data TEXT NOT NULL);")?;
        let metrics = {
            let mut s = c.prepare("SELECT key,data FROM metrics")?;
            s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .filter_map(|r| r.ok())
                .filter_map(|(k, v)| serde_json::from_str::<Metric>(&v).ok().map(|v| (k, v)))
                .collect()
        };
        Ok(Self {
            ops,
            memory: moka::future::Cache::builder()
                .max_capacity(256 * 1024 * 1024)
                .weigher(|_: &String, c: &Cached| {
                    c.blob
                        .as_ref()
                        .map_or(128, |b| (b.bytes.len() + 256).min(u32::MAX as usize) as u32)
                })
                .build(),
            disk: Arc::new(Mutex::new(c)),
            locks: Mutex::new(HashMap::new()),
            slots: Semaphore::new(48),
            background_slots: Semaphore::new(4),
            discovery_slots: Semaphore::new(32),
            metrics: Mutex::new(metrics),
            failures: moka::future::Cache::builder()
                .max_capacity(10000)
                .time_to_live(Duration::from_secs(2))
                .build(),
            details: moka::future::Cache::builder()
                .max_capacity(64 * 1024 * 1024)
                .time_to_live(Duration::from_secs(300))
                .weigher(|_: &String, d: &DetailCached| {
                    d.value.to_string().len().min(u32::MAX as usize) as u32
                })
                .build(),
            detail_refresh: Arc::new(Semaphore::new(4)),
            finalized: moka::future::Cache::builder()
                .max_capacity(100000)
                .time_to_live(Duration::from_secs(3600))
                .build(),
            lists: moka::future::Cache::builder()
                .max_capacity(16 * 1024 * 1024)
                .weigher(|_: &String, p: &(f64, Page)| {
                    (p.1.objects.len() * 256).min(u32::MAX as usize) as u32
                })
                .build(),
        })
    }
    fn lock(&self, key: &str) -> Arc<AsyncMutex<()>> {
        let mut locks = self.locks.lock().unwrap();
        if locks.len() > 1024 {
            locks.retain(|_, v| v.strong_count() > 0);
        }
        if let Some(v) = locks.get(key).and_then(Weak::upgrade) {
            v
        } else {
            let l = Arc::new(AsyncMutex::new(()));
            locks.insert(key.into(), Arc::downgrade(&l));
            l
        }
    }
    fn metric(&self, m: &str, op: &str, purpose: &str, status: &str, bytes: usize) {
        let mut metrics = self.metrics.lock().unwrap();
        let v = metrics.entry(format!("{m}:{op}:{purpose}")).or_default();
        v.requests += 1;
        v.bytes += bytes as u64;
        match status {
            "404" => v.not_found += 1,
            "304" => v.not_modified += 1,
            "error" => v.errors += 1,
            _ => {}
        }
    }
    pub fn metrics(&self) -> Value {
        json!(*self.metrics.lock().unwrap())
    }
    pub async fn checkpoint(&self) -> Result<()> {
        let disk = self.disk.clone();
        let metrics = self.metrics.lock().unwrap().clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut c = disk.lock().unwrap();
            let tx = c.transaction()?;
            for (k, v) in metrics {
                tx.execute(
                    "INSERT OR REPLACE INTO metrics VALUES(?,?)",
                    params![k, serde_json::to_string(&v)?],
                )?;
            }
            let mut size: i64 = tx.query_row(
                "SELECT COALESCE(SUM(COALESCE(length(body),0)+512),0) FROM artifacts",
                [],
                |r| r.get(0),
            )?;
            while size > 20 * 1024 * 1024 * 1024 {
                let (k, n): (String, i64) = tx.query_row(
                    "SELECT key,COALESCE(length(body),0)+512 FROM artifacts ORDER BY used LIMIT 1",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?;
                tx.execute("DELETE FROM artifacts WHERE key=?", [k])?;
                size -= n;
            }
            tx.commit()?;
            c.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")?;
            Ok(())
        })
        .await??;
        Ok(())
    }
    async fn disk_get(&self, key: String) -> Result<Option<Cached>> {
        let disk = self.disk.clone();
        tokio::task::spawn_blocking(move || -> Result<_> {
            let c = disk.lock().unwrap();
            let row: Option<DiskRow> = c
                .query_row(
                    "SELECT body,etag,modified,expires FROM artifacts WHERE key=?",
                    [&key],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .optional()?;
            if row.is_some() {
                c.execute(
                    "UPDATE artifacts SET used=? WHERE key=?",
                    params![now(), key],
                )?;
            }
            Ok(row.map(|(body, etag, modified, expires)| Cached {
                blob: body.map(|b| Blob {
                    bytes: b.into(),
                    etag,
                    modified,
                }),
                expires,
            }))
        })
        .await?
    }
    async fn store(&self, key: String, v: Cached) -> Result<()> {
        let disk = self.disk.clone();
        let copy = v.clone();
        let k = key.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let c = disk.lock().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?)",
                params![
                    k,
                    copy.blob.as_ref().map(|b| b.bytes.as_ref()),
                    copy.blob.as_ref().and_then(|b| b.etag.as_deref()),
                    copy.blob.as_ref().and_then(|b| b.modified),
                    copy.expires,
                    now()
                ],
            )?;
            Ok(())
        })
        .await??;
        self.memory.insert(key, v).await;
        Ok(())
    }
    pub async fn get(&self, m: &str, path: &str, purpose: &str) -> Result<Option<Blob>> {
        self.get_checked(m, path, purpose, None).await
    }
    pub async fn get_checked(
        &self,
        m: &str,
        path: &str,
        purpose: &str,
        expected: Option<&str>,
    ) -> Result<Option<Blob>> {
        safe(path)?;
        let op = self.ops.get(m).context("storage unavailable")?;
        let key = format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "{}:{}:{}{}",
                    op.info().scheme(),
                    op.info().name(),
                    op.info().root(),
                    path
                )
                .as_bytes()
            )
        );
        let attempt = path
            .split_once("/screenshots/")
            .map(|(a, _)| a)
            .or_else(|| path.rsplit_once('/').map(|(a, _)| a))
            .unwrap_or("");
        let ttl = if self.finalized.get(&format!("{m}:{attempt}")).await == Some(true) {
            60.
        } else {
            5.
        };
        let lock = self.lock(&key);
        let _guard = lock.lock().await;
        if self.failures.get(&key).await.is_some() {
            anyhow::bail!("storage request recently failed; retry deferred");
        }
        let cached = match self.memory.get(&key).await {
            Some(v) => Some(v),
            None => self.disk_get(key.clone()).await?,
        };
        if let Some(v) = &cached {
            let matches =
                expected.is_none_or(|e| v.blob.as_ref().and_then(|b| b.etag.as_deref()) == Some(e));
            let fresh_listing = purpose != "interactive" && expected.is_some() && matches;
            if fresh_listing || (matches && v.expires > now()) {
                return Ok(v.blob.clone());
            }
        }
        let _discovery = if purpose == "discovery" {
            Some(self.discovery_slots.acquire().await?)
        } else {
            None
        };
        let _background = if purpose == "historical" {
            Some(self.background_slots.acquire().await?)
        } else {
            None
        };
        let _permit = tokio::time::timeout(Duration::from_secs(10), self.slots.acquire()).await??;
        let mut args = OpRead::default();
        if let Some(etag) = cached
            .as_ref()
            .and_then(|v| v.blob.as_ref())
            .and_then(|b| b.etag.as_deref())
        {
            args = args.with_if_none_match(etag);
        }
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let (r, mut reader) = op.inner().read(path, args).await?;
            let meta = r.metadata();
            let etag = meta.and_then(|m| m.etag()).map(String::from);
            let modified = meta
                .and_then(|m| m.last_modified())
                .map(|t| t.into_inner().as_second() as f64);
            let mut bytes = Vec::new();
            loop {
                let b = reader.read().await?;
                if b.is_empty() {
                    break;
                }
                if bytes.len() + b.len() > 16 * 1024 * 1024 {
                    return Err(opendal::Error::new(
                        ErrorKind::Unexpected,
                        "artifact exceeds 16 MiB limit",
                    ));
                }
                bytes.extend_from_slice(&b.to_bytes());
            }
            Ok::<_, opendal::Error>(Blob {
                bytes: bytes.into(),
                etag,
                modified,
            })
        })
        .await;
        let value = match result {
            Ok(Ok(blob)) => {
                self.metric(m, "GET", purpose, "200", blob.bytes.len());
                Cached {
                    blob: Some(blob),
                    expires: now() + ttl,
                }
            }
            Ok(Err(e)) if e.kind() == ErrorKind::NotFound => {
                self.metric(m, "GET", purpose, "404", 0);
                Cached {
                    blob: None,
                    expires: now() + ttl,
                }
            }
            Ok(Err(e))
                if e.kind() == ErrorKind::ConditionNotMatch
                    && cached.as_ref().is_some_and(|v| v.blob.is_some()) =>
            {
                self.metric(m, "GET", purpose, "304", 0);
                Cached {
                    blob: cached.unwrap().blob,
                    expires: now() + ttl,
                }
            }
            Ok(Err(e)) => {
                self.metric(m, "GET", purpose, "error", 0);
                self.failures.insert(key.clone(), ()).await;
                return Err(anyhow::anyhow!("storage read failed ({:?})", e.kind()));
            }
            Err(_) => {
                self.metric(m, "GET", purpose, "error", 0);
                self.failures.insert(key.clone(), ()).await;
                anyhow::bail!("storage read timed out")
            }
        };
        let blob = value.blob.clone();
        self.store(key, value).await?;
        Ok(blob)
    }
    pub async fn list(
        &self,
        m: &str,
        prefix: &str,
        cursor: &str,
        purpose: &str,
        recursive: bool,
    ) -> Result<Page> {
        let prefix = if prefix == "/" { "" } else { prefix };
        if !prefix.is_empty() {
            safe(prefix.trim_end_matches('/'))?;
        }
        let op = self.ops.get(m).context("storage unavailable")?;
        let key = format!(
            "list:{}:{}:{}:{prefix}:{cursor}:{recursive}",
            op.info().scheme(),
            op.info().name(),
            op.info().root()
        );
        let lock = self.lock(&key);
        let _guard = lock.lock().await;
        if self.failures.get(&key).await.is_some() {
            anyhow::bail!("storage request recently failed; retry deferred");
        }
        if let Some((at, page)) = self.lists.get(&key).await
            && at + 5. > now()
        {
            return Ok(page);
        }
        let _discovery = if purpose == "discovery" {
            Some(self.discovery_slots.acquire().await?)
        } else {
            None
        };
        let _background = if purpose == "historical" {
            Some(self.background_slots.acquire().await?)
        } else {
            None
        };
        let _p = tokio::time::timeout(Duration::from_secs(10), self.slots.acquire()).await??;
        let args = OpList::default()
            .with_limit(1000)
            .with_recursive(recursive)
            .with_start_after(cursor);
        let listed_at = now();
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let (_, mut list) = op
                .inner()
                .list(if prefix.is_empty() { "/" } else { prefix }, args)
                .await?;
            let mut objects = Vec::new();
            while objects.len() < 1000 {
                let Some(e) = list.next().await? else { break };
                let meta = e.metadata();
                objects.push(Object {
                    path: e.path().into(),
                    etag: meta.etag().map(String::from),
                    modified: meta
                        .last_modified()
                        .map(|t| t.into_inner().as_second() as f64)
                        .unwrap_or(0.),
                    size: meta.content_length(),
                    dir: meta.is_dir(),
                });
            }
            objects.sort_by(|a, b| a.path.cmp(&b.path));
            let cursor = if objects.len() == 1000 {
                objects.last().map(|e| e.path.clone())
            } else {
                None
            };
            Ok::<_, opendal::Error>(Page {
                objects,
                cursor,
                listed_at: Some(listed_at),
            })
        })
        .await;
        match result {
            Ok(Ok(page)) => {
                self.metric(m, "LIST", purpose, "200", 0);
                self.lists.insert(key, (now(), page.clone())).await;
                Ok(page)
            }
            Ok(Err(e)) => {
                self.metric(m, "LIST", purpose, "error", 0);
                self.failures.insert(key.clone(), ()).await;
                Err(anyhow::anyhow!("storage list failed ({:?})", e.kind()))
            }
            Err(_) => {
                self.metric(m, "LIST", purpose, "error", 0);
                self.failures.insert(key.clone(), ()).await;
                anyhow::bail!("storage list timed out")
            }
        }
    }
    pub async fn detail_cached(self: &Arc<Self>, m: &str, id: &str, e: Value) -> Result<Value> {
        let terminal = e["score"].is_number()
            || e["evaluation_terminal"] == true
            || !text(&e, "error").is_empty();
        if !terminal {
            return self.detail(m, id, e).await;
        }
        let key = format!(
            "detail:{m}:{id}:{:x}",
            Sha256::digest(e.to_string().as_bytes())
        );
        let lock = self.lock(&key);
        if let Some(cached) = self.details.get(&key).await {
            if now() - cached.checked_at > 60.
                && let Ok(guard) = lock.try_lock_owned()
                && let Ok(permit) = self.detail_refresh.clone().try_acquire_owned()
            {
                let (store, m, id) = (self.clone(), m.to_string(), id.to_string());
                tokio::spawn(async move {
                    let (_guard, _permit) = (guard, permit);
                    if let Ok(value) = store.detail(&m, &id, e).await {
                        store
                            .details
                            .insert(
                                key,
                                DetailCached {
                                    value,
                                    checked_at: now(),
                                },
                            )
                            .await;
                    }
                });
            }
            return Ok(cached.value);
        }
        let _guard = lock.lock().await;
        if let Some(cached) = self.details.get(&key).await {
            return Ok(cached.value);
        }
        let value = self.detail(m, id, e).await?;
        self.details
            .insert(
                key,
                DetailCached {
                    value: value.clone(),
                    checked_at: now(),
                },
            )
            .await;
        Ok(value)
    }
    pub async fn detail(&self, m: &str, id: &str, mut e: Value) -> Result<Value> {
        self.finalized
            .insert(
                format!("{m}:{id}"),
                e["score"].is_number()
                    || e["evaluation_terminal"] == true
                    || !text(&e, "error").is_empty(),
            )
            .await;
        let etags = e["_artifact_etags"].clone();
        let get = |file: &str| {
            let path = format!("{id}/{file}");
            let expected = etags[file].as_str().map(str::to_owned);
            async move {
                self.get_checked(m, &path, "interactive", expected.as_deref())
                    .await
            }
        };
        let (raw, result, error, judge) = tokio::try_join!(
            get("traj.json"),
            get("result.txt"),
            get("error.txt"),
            get("judge_result.txt")
        )?;
        if let Some(b) = result {
            let (s, r) = aggregate::parse_result(&String::from_utf8_lossy(&b.bytes));
            e["score"] = json!(s);
            e["reason"] = json!(r);
        }
        if let Some(b) = error {
            e["error"] = json!(String::from_utf8_lossy(&b.bytes).trim());
            e["error_mtime"] = json!(b.modified);
        }
        if [
            "Railway12306V130",
            "Railway12306V140",
            "Railway12306V150",
            "Railway12306New70V150",
        ]
        .iter()
        .any(|p| text(&e, "task_id").starts_with(p))
            && let Some(b) = get("verifier_result.json").await?
        {
            let reason = String::from_utf8_lossy(&b.bytes);
            if aggregate::verifier(&reason).is_some() {
                e["reason"] = json!(reason);
            }
        }
        let mut verdict = None;
        let mut output = None;
        if let Some(b) = &judge {
            let s = String::from_utf8_lossy(&b.bytes);
            let mut lines = s.lines();
            let first = lines.next().unwrap_or("");
            verdict = first
                .strip_prefix("verdict:")
                .and_then(|v| v.trim().parse::<f64>().ok());
            let rest = lines.collect::<Vec<_>>().join("\n");
            if !rest.trim().is_empty() {
                output = Some(rest.trim().to_string());
            }
            if verdict.is_some() {
                e["judge_reward"] = json!(verdict);
            }
        }
        e = aggregate::normalize(e);
        let mut steps = json!([]);
        let mut tools = Value::Null;
        let mut writing = false;
        if let Some(b) = raw {
            match serde_json::from_slice::<Value>(&b.bytes) {
                Ok(v) => {
                    if v["0"]["traj"].is_array() {
                        steps = v["0"]["traj"].clone();
                    }
                    tools = v["0"]["tools"].clone();
                }
                Err(_) => writing = true,
            }
        }
        let mut shots = Vec::new();
        let mut cursor = String::new();
        loop {
            let page = self
                .list(
                    m,
                    &format!("{id}/screenshots/"),
                    &cursor,
                    "interactive",
                    false,
                )
                .await?;
            for o in page.objects {
                if !o.dir {
                    let s = o.path.rsplit('/').next().unwrap_or("");
                    if [".png", ".jpg", ".jpeg", ".webp"]
                        .iter()
                        .any(|ext| s.to_lowercase().ends_with(ext))
                    {
                        shots.push(s.to_string());
                    }
                }
            }
            if let Some(c) = page.cursor {
                cursor = c;
            } else {
                break;
            }
            anyhow::ensure!(shots.len() < 10000, "too many screenshots");
        }
        shots.sort_by_key(|s| {
            s.rsplit_once('.')
                .and_then(|(b, _)| b.rsplit('-').next())
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(0)
        });
        let mut out = json!({"available":true,"id":id,"steps":steps,"tools":tools,"screenshots":shots,"writing":writing,"judge_verdict":verdict,"judge_output":output,"run_key":aggregate::run_key(&e)});
        for k in [
            "score",
            "reason",
            "error",
            "timeout",
            "timeout_inferred",
            "timeout_kind",
            "evaluation_method",
            "built_in_score",
            "llm_judge_score",
            "train_id",
            "task_id",
            "model",
            "model_slug",
        ] {
            out[k] = e[k].clone();
        }
        if judge.is_some() && e["evaluation_method"] != "both" {
            out["evaluation_method"] = json!("llm_judge");
        }
        for k in [
            "evaluation_terminal",
            "verifier_status",
            "verifier_score",
            "verifier_result",
        ] {
            if !e[k].is_null() {
                out[k] = e[k].clone();
            }
        }
        Ok(out)
    }
}
impl<'de> Deserialize<'de> for Metric {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        Ok(Self {
            requests: v["requests"].as_u64().unwrap_or(0),
            bytes: v["bytes"].as_u64().unwrap_or(0),
            not_found: v["not_found"].as_u64().unwrap_or(0),
            errors: v["errors"].as_u64().unwrap_or(0),
            not_modified: v["not_modified"].as_u64().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod detail_tests {
    use super::*;
    #[tokio::test]
    async fn finalized_details_revalidate_off_path_and_changed_metadata_invalidates() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("objects");
        std::fs::create_dir_all(root.join("attempt/screenshots")).unwrap();
        std::fs::write(root.join("attempt/result.txt"), "1\nfixture").unwrap();
        std::fs::write(root.join("attempt/traj.json"), "{}").unwrap();
        let op = Operator::new(opendal::services::Fs::default().root(root.to_str().unwrap()))
            .unwrap()
            .finish();
        let store = Arc::new(
            Storage::with_operators(
                HashMap::from([("0".into(), op)]),
                tmp.path().join("cache.db"),
            )
            .unwrap(),
        );
        let e = json!({"score":1,"id":"attempt","task_id":"task"});
        let first = store
            .detail_cached("0", "attempt", e.clone())
            .await
            .unwrap();
        assert_eq!(first["score"], 1.);
        let before = store.metrics();
        for _ in 0..50 {
            assert_eq!(
                store
                    .detail_cached("0", "attempt", e.clone())
                    .await
                    .unwrap(),
                first
            );
        }
        assert_eq!(before, store.metrics());
        let key = format!(
            "detail:0:attempt:{:x}",
            Sha256::digest(e.to_string().as_bytes())
        );
        let mut cached = store.details.get(&key).await.unwrap();
        cached.checked_at -= 61.;
        store.details.insert(key.clone(), cached).await;
        let guard = store.lock(&key).lock_owned().await;
        assert_eq!(
            tokio::time::timeout(
                Duration::from_secs(1),
                store.detail_cached("0", "attempt", e.clone())
            )
            .await
            .unwrap()
            .unwrap(),
            first
        );
        drop(guard);
        std::fs::write(root.join("attempt/result.txt"), "0\nfixture").unwrap();
        let changed = json!({"score":0,"id":"attempt","task_id":"task","_artifact_etags":{"result.txt":"changed"}});
        assert_eq!(
            store.detail_cached("0", "attempt", changed).await.unwrap()["score"],
            0.
        );
    }
}
