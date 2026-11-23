use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
};
const LIMITS_MS: [u64; 15] = [
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    30000,
    60000,
    u64::MAX,
];
#[derive(Default, Clone, Serialize)]
struct Histogram {
    count: u64,
    total_ms: f64,
    max_ms: f64,
    buckets: [u64; 15],
}
static METRICS: LazyLock<Mutex<BTreeMap<String, Histogram>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
pub fn record(manager: &str, lane: &str, phase: &str, elapsed: Duration) {
    let key = format!("{manager}:{lane}:{phase}");
    let Ok(mut metrics) = METRICS.lock() else {
        return;
    };
    if metrics.len() >= 256 && !metrics.contains_key(&key) {
        return;
    }
    let h = metrics.entry(key).or_default();
    let ms = elapsed.as_secs_f64() * 1000.;
    h.count += 1;
    h.total_ms += ms;
    h.max_ms = h.max_ms.max(ms);
    h.buckets[LIMITS_MS
        .iter()
        .position(|limit| ms <= *limit as f64)
        .unwrap_or(14)] += 1;
}
pub struct Timer {
    manager: String,
    lane: String,
    phase: &'static str,
    start: Instant,
}
impl Timer {
    pub fn new(manager: &str, lane: &str, phase: &'static str) -> Self {
        Self {
            manager: manager.into(),
            lane: lane.into(),
            phase,
            start: Instant::now(),
        }
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        record(&self.manager, &self.lane, self.phase, self.start.elapsed());
    }
}
pub fn persist_if_due(c: &rusqlite::Connection) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::now() as u64;
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 60
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return Ok(());
    }
    let snapshot = METRICS.lock().map(|m| m.clone()).unwrap_or_default();
    crate::db::set_meta(c,"indexer_phase_metrics",&serde_json::json!({"time":now,"pid":std::process::id(),"bucket_upper_ms":LIMITS_MS,"phases":snapshot}).to_string())
}
