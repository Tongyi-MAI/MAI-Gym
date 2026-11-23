pub mod aggregate;
pub mod config;
pub mod db;
pub mod indexer;
pub mod runtime;
pub mod server;
pub mod shards;
pub mod storage;
pub mod timing;
pub fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
