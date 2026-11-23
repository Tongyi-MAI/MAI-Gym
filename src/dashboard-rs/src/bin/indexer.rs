use clap::{Parser, Subcommand};
use futures::TryStreamExt;
use mai_dashboard::{config::Config, db};
use std::path::PathBuf;

fn prepare_publication(c: &mut rusqlite::Connection, cfg: &Config) -> anyhow::Result<()> {
    db::schema(c)?;
    mai_dashboard::indexer::seed(c, cfg)?;
    while db::rebuild_one(c, cfg)? {}
    db::set_meta(c, "initial_publication_done", "true")
}

async fn join_lanes<F>(lanes: impl IntoIterator<Item = F>) -> anyhow::Result<Vec<()>>
where
    F: std::future::Future<Output = anyhow::Result<()>>,
{
    lanes
        .into_iter()
        .collect::<futures::stream::FuturesUnordered<_>>()
        .try_collect()
        .await
}

async fn supervise_lane<F, Fut>(manager: usize, lane: &str, cycle: F) -> anyhow::Result<()>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    loop {
        let delay = match cycle().await {
            Ok(()) => std::time::Duration::from_millis(if lane.starts_with("active:") {
                250
            } else {
                100
            }),
            Err(error) => {
                tracing::warn!(manager, lane, "indexing lane retrying: {error}");
                std::time::Duration::from_secs(1)
            }
        };
        tokio::time::sleep(delay).await;
    }
}
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    db: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Prepare,
    Run {
        #[arg(long, requires = "legacy_db")]
        legacy_bridge: bool,
        #[arg(long)]
        legacy_db: Option<PathBuf>,
        #[arg(long)]
        broker: String,
        #[arg(long)]
        broker_token_file: PathBuf,
    },
    Journal {
        #[arg(long)]
        legacy_db: PathBuf,
    },
    Import {
        #[arg(long)]
        legacy_db: PathBuf,
    },
    Follow {
        #[arg(long)]
        legacy_db: PathBuf,
        #[arg(long)]
        once: bool,
    },
    Scan {
        #[arg(long)]
        broker: String,
        #[arg(long)]
        broker_token_file: PathBuf,
        #[arg(long)]
        manager: String,
        #[arg(long)]
        prefix: String,
        #[arg(long, default_value_t = 100)]
        max_pages: usize,
    },
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let cfg = Config::load(&args.config)?;
    let db_parent = args
        .db
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(db_parent)?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(args.db.with_extension("writer.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock)?;
    let consumer = format!(
        "dashboard-rs:{}",
        std::fs::canonicalize(db_parent)?
            .join(args.db.file_name().unwrap())
            .display()
    );
    match args.command {
        Command::Prepare => {
            let mut c = db::connect(&args.db)?;
            prepare_publication(&mut c, &cfg)?;
            println!("current read model and discovery queues prepared without OSS requests");
        }
        Command::Run {
            legacy_bridge,
            legacy_db,
            broker,
            broker_token_file,
        } => {
            let scanner_db = legacy_db.as_ref().unwrap_or(&args.db);
            let ownership = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(format!("{}.scanner.lock", scanner_db.display()))?;
            fs2::FileExt::try_lock_exclusive(&ownership).map_err(|_| {
                anyhow::anyhow!("another scanner owns indexing; refusing duplicate scans")
            })?;
            let mut c = db::connect(&args.db)?;
            db::schema(&c)?;
            if let Some(legacy_db) = legacy_db {
                anyhow::ensure!(
                    db::get_meta(&c, "initial_publication_done")?.as_deref() == Some("true")
                        && db::get_meta(&c, "import_done")?.as_deref() == Some("true"),
                    "initial import/publication not complete"
                );
                let src = rusqlite::Connection::open_with_flags(
                    &legacy_db,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
                )?;
                while db::replay(&mut c, &src, &cfg)? > 0 {}
                db::sync_runtime(&mut c, &src, &cfg)?;
                mai_dashboard::indexer::seed(&c, &cfg)?;
            } else {
                db::set_meta(&c, "compatibility_publish", "false")?;
                prepare_publication(&mut c, &cfg)?;
            }
            db::set_meta(
                &c,
                "compatibility_publish",
                if legacy_bridge { "true" } else { "false" },
            )?;
            db::set_meta(&c, "indexing_owner", "rust")?;
            db::set_meta(&c, "indexing_pid", &std::process::id().to_string())?;
            #[cfg(target_os = "linux")]
            {
                let stat = std::fs::read_to_string("/proc/self/stat")?;
                let ticks = stat
                    .rsplit_once(')')
                    .ok_or_else(|| anyhow::anyhow!("process stat"))?
                    .1
                    .split_whitespace()
                    .nth(19)
                    .ok_or_else(|| anyhow::anyhow!("process start time"))?;
                db::set_meta(&c, "indexing_start_ticks", ticks)?;
            }
            let token = std::fs::read_to_string(broker_token_file)?
                .trim()
                .to_string();
            let (runtime_path, runtime_cfg) = (args.db.clone(), cfg.clone());
            tokio::spawn(async move {
                if let Err(e) = mai_dashboard::runtime::run(runtime_path, runtime_cfg).await {
                    tracing::error!("runtime sampling stopped: {e}");
                }
            });
            let (inventory_path, inventory_cfg) = (args.db.clone(), cfg.clone());
            tokio::spawn(async move {
                if let Err(e) =
                    mai_dashboard::runtime::inventory_loop(inventory_path, inventory_cfg).await
                {
                    tracing::error!("inventory stopped: {e}");
                }
            });
            let aggregate_path = args.db.clone();
            let aggregate_cfg = cfg.clone();
            tokio::spawn(async move {
                loop {
                    let path = aggregate_path.clone();
                    let cfg = aggregate_cfg.clone();
                    let publication = db::index_publication().await;
                    let outcome = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
                        let _publication = publication;
                        let mut db = db::connect(&path)?;
                        let result = db::rebuild_one(&mut db, &cfg);
                        if mai_dashboard::timing::persist_if_due(&db).is_err() {
                            tracing::warn!("phase metrics publication deferred");
                        }
                        result
                    })
                    .await;
                    match outcome {
                        Ok(Ok(true)) => {}
                        Ok(Ok(false)) => {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await
                        }
                        other => {
                            tracing::error!("publication deferred: {other:?}");
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        }
                    }
                }
            });
            let lanes = cfg
                .managers
                .iter()
                .enumerate()
                .filter(|(_, m)| m.traj_root.is_some())
                .flat_map(|(i, _)| {
                    mai_dashboard::indexer::ACTIVE_LANES
                        .into_iter()
                        .chain(["discovery", "attempts", "watch"])
                        .map(move |lane| (i, lane))
                });
            let priority = join_lanes(lanes.map(|(i, lane)| {
                let path = args.db.clone();
                let cfg = cfg.clone();
                let broker = broker.clone();
                let token = token.clone();
                async move {
                    let c = db::connect(&path)?;
                    supervise_lane(i, lane, || async {
                        mai_dashboard::indexer::scan_lane(
                            path.clone(),
                            cfg.clone(),
                            broker.clone(),
                            token.clone(),
                            i.to_string(),
                            lane,
                            if lane.starts_with("active:") { 3 } else { 12 },
                        )
                        .await?;
                        db::set_meta(
                            &c,
                            &format!("heartbeat:{i}:{lane}"),
                            &mai_dashboard::now().to_string(),
                        )?;
                        if lane.starts_with("active:") {
                            let ages: (i64, Option<f64>) = c.query_row(
                                "SELECT COUNT(*),MIN(CAST(value AS REAL)) FROM meta WHERE key GLOB ?",
                                [format!("heartbeat:{i}:active:[0-3]")],
                                |r| Ok((r.get(0)?, r.get(1)?)),
                            )?;
                            if ages.0 == 4 && let Some(at) = ages.1 {
                                db::set_meta(&c, &format!("heartbeat:{i}:active"), &at.to_string())?;
                            }
                        }
                        db::set_meta(&c, "indexer_heartbeat", &mai_dashboard::now().to_string())?;
                        Ok(())
                    })
                    .await
                }
            }));
            let historical = async {
                loop {
                    let backlog: i64 =
                        c.query_row("SELECT COUNT(*) FROM legacy_updates", [], |r| r.get(0))?;
                    if backlog > 100000 {
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                    let scans = cfg
                        .managers
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| m.traj_root.is_some())
                        .map(|(i, _)| {
                            mai_dashboard::indexer::scan_lane(
                                args.db.clone(),
                                cfg.clone(),
                                broker.clone(),
                                token.clone(),
                                i.to_string(),
                                "historical",
                                2,
                            )
                        });
                    for result in futures::future::join_all(scans).await {
                        result?;
                    }
                    db::expire_runs(&c)?;
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                #[allow(unreachable_code)]
                Ok::<(), anyhow::Error>(())
            };
            tokio::try_join!(priority, historical)?;
        }
        Command::Journal { legacy_db } => {
            let mut src = db::connect(&legacy_db)?;
            db::install_journal(&mut src)?;
            println!("journal installed");
        }
        Command::Import { legacy_db } => {
            let mut c = db::connect(&args.db)?;
            db::schema(&c)?;
            let mut control = db::connect(&legacy_db)?;
            let prior = db::get_meta(&c, "journal_seq")?
                .map(|s| s.parse::<i64>())
                .transpose()?;
            let prior =
                prior.or_else(|| db::get_meta(&c, "import_cursor").ok().flatten().map(|_| 0));
            let start = db::register_consumer(&mut control, &consumer, prior)?;
            if prior.is_none() {
                db::set_meta(&c, "journal_seq", &start.to_string())?;
            }
            let src = rusqlite::Connection::open_with_flags(
                legacy_db,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            )?;
            src.busy_timeout(std::time::Duration::from_secs(5))?;
            let _: i64 = src.query_row("SELECT COUNT(*) FROM dashboard_rs_journal", [], |r| {
                r.get(0)
            })?;
            if db::get_meta(&c, "import_done")?.as_deref() != Some("true") {
                db::import_aux(&mut c, &src, &cfg)?;
                let mut n = 0;
                loop {
                    let batch = db::import_batch(&mut c, &src, &cfg)?;
                    n += batch;
                    if batch == 0 {
                        break;
                    }
                    if n % 100000 == 0 {
                        println!("imported {n} rows");
                    }
                }
            }
            while db::replay(&mut c, &src, &cfg)? > 0 {}
            let mut runs = 0;
            while db::rebuild_one(&mut c, &cfg)? {
                runs += 1;
                if runs % 100 == 0 {
                    println!("published {runs} affected runs");
                }
            }
            db::set_meta(&c, "initial_publication_done", "true")?;
            println!("import and initial publication complete");
        }
        Command::Follow { legacy_db, once } => {
            let mut c = db::connect(&args.db)?;
            db::schema(&c)?;
            let mut control = db::connect(&legacy_db)?;
            let prior = db::get_meta(&c, "journal_seq")?
                .map(|s| s.parse::<i64>())
                .transpose()?;
            let prior =
                prior.or_else(|| db::get_meta(&c, "import_cursor").ok().flatten().map(|_| 0));
            let start = db::register_consumer(&mut control, &consumer, prior)?;
            if prior.is_none() {
                db::set_meta(&c, "journal_seq", &start.to_string())?;
            }
            let src = rusqlite::Connection::open_with_flags(
                legacy_db,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            )?;
            src.busy_timeout(std::time::Duration::from_secs(5))?;
            loop {
                let n = db::replay(&mut c, &src, &cfg)?;
                db::expire_runs(&c)?;
                db::set_meta(&c, "indexer_heartbeat", &mai_dashboard::now().to_string())?;
                db::sync_runtime(&mut c, &src, &cfg)?;
                let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while std::time::Instant::now() < until && db::rebuild_one(&mut c, &cfg)? {}
                let seq = db::get_meta(&c, "journal_seq")?
                    .unwrap_or_else(|| "0".into())
                    .parse()?;
                if let Err(error) = db::acknowledge(&mut control, &consumer, seq) {
                    if db::is_busy(&error) {
                        tracing::warn!("journal acknowledgement deferred: {error}");
                    } else {
                        return Err(error);
                    }
                }
                if once {
                    break;
                }
                if n == 0 {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
        Command::Scan {
            broker,
            broker_token_file,
            manager,
            prefix,
            max_pages,
        } => {
            cfg.manager(&manager)?;
            mai_dashboard::indexer::scan(
                args.db,
                cfg,
                broker,
                std::fs::read_to_string(broker_token_file)?.trim().into(),
                manager,
                prefix,
                max_pages,
            )
            .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mai_dashboard::config::Manager;
    use serde_json::json;
    use std::time::Duration;

    fn fixture_config() -> Config {
        Config {
            url_base: String::new(),
            managers: vec![Manager {
                label: "Fixture".into(),
                url: "http://fixture".into(),
                traj_root: Some("oss://example-bucket/trajectories".into()),
                capacity_mode: None,
                k8s_inventory: None,
            }],
        }
    }

    #[test]
    fn fresh_publication_prepares_discovery_without_claiming_import_or_scan_coverage() {
        let mut c = rusqlite::Connection::open_in_memory().unwrap();
        prepare_publication(&mut c, &fixture_config()).unwrap();
        assert_eq!(
            db::get_meta(&c, "initial_publication_done")
                .unwrap()
                .as_deref(),
            Some("true")
        );
        assert!(db::get_meta(&c, "import_done").unwrap().is_none());
        let queued: i64 = c
            .query_row("SELECT COUNT(*) FROM scan_jobs", [], |r| r.get(0))
            .unwrap();
        let completed: i64 = c
            .query_row("SELECT COUNT(*) FROM scan_completed", [], |r| r.get(0))
            .unwrap();
        assert!(queued > 0);
        assert_eq!(completed, 0);
    }

    #[test]
    fn preparation_publishes_existing_dirty_runs_and_is_idempotent() {
        let mut c = rusqlite::Connection::open_in_memory().unwrap();
        db::schema(&c).unwrap();
        db::upsert(
            &c,
            "0",
            json!({
                "id": "run/r00000/task/attempt", "task_id": "task", "attempt": "attempt",
                "train_id": "run", "rollout_id": 0, "score": 1., "mtime": mai_dashboard::now(),
            }),
        )
        .unwrap();
        prepare_publication(&mut c, &fixture_config()).unwrap();
        prepare_publication(&mut c, &fixture_config()).unwrap();
        let dirty: i64 = c
            .query_row("SELECT COUNT(*) FROM dirty_runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(dirty, 0);
        let views = db::values(&c, "SELECT data FROM views WHERE kind='run'", &[]).unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0]["count"], 1);
    }

    #[test]
    fn run_accepts_fresh_database_but_legacy_bridge_requires_legacy_database() {
        let base = [
            "indexer",
            "--config",
            "config.yaml",
            "--db",
            "index.db",
            "run",
            "--broker",
            "http://broker",
            "--broker-token-file",
            "broker.token",
        ];
        let args = Args::try_parse_from(base).unwrap();
        assert!(matches!(
            args.command,
            Command::Run {
                legacy_db: None,
                ..
            }
        ));
        assert!(Args::try_parse_from(base.into_iter().chain(["--legacy-bridge"])).is_err());
        let args = Args::try_parse_from(base.into_iter().chain([
            "--legacy-bridge",
            "--legacy-db",
            "legacy.db",
        ]))
        .unwrap();
        assert!(matches!(
            args.command,
            Command::Run {
                legacy_db: Some(_),
                legacy_bridge: true,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn lane_recovers_after_sqlite_write_contention() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index.db");
        let c = db::connect(&path).unwrap();
        db::schema(&c).unwrap();
        c.busy_timeout(Duration::ZERO).unwrap();
        let blocker = db::connect(&path).unwrap();
        blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
        let failed = tokio::sync::Notify::new();
        let recovered = tokio::sync::Notify::new();
        let lane = supervise_lane(0, "active", || async {
            if let Err(error) = db::set_meta(&c, "heartbeat:0:active", "recovered") {
                failed.notify_one();
                return Err(error);
            }
            recovered.notify_one();
            Ok(())
        });
        let check = async {
            failed.notified().await;
            assert!(db::get_meta(&c, "heartbeat:0:active").unwrap().is_none());
            blocker.execute_batch("COMMIT").unwrap();
            recovered.notified().await;
            assert_eq!(
                db::get_meta(&c, "heartbeat:0:active").unwrap().as_deref(),
                Some("recovered")
            );
        };
        tokio::time::timeout(Duration::from_secs(3), async {
            tokio::select! {
                result = lane => panic!("lane exited instead of retrying: {result:?}"),
                _ = check => {}
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn failed_lane_is_not_hidden_by_earlier_infinite_lanes() {
        let lanes = (0..33).map(|i| async move {
            if i == 24 {
                anyhow::bail!("injected lane initialization failure");
            }
            futures::future::pending::<anyhow::Result<()>>().await
        });
        let result = tokio::time::timeout(Duration::from_secs(1), join_lanes(lanes))
            .await
            .expect("failure hidden by an infinite lane");
        assert!(result.unwrap_err().to_string().contains("injected"));
    }
}
