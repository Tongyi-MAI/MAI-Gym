use mai_dashboard::db;
use std::process::Command;

#[test]
fn prepare_creates_fresh_database_directory_without_legacy_database_or_storage_access() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.yaml");
    std::fs::write(&config, "managers:\n  - label: Fixture\n    url: http://fixture\n    traj_root: oss://example-bucket/trajectories\n").unwrap();
    let path = temp.path().join("new/state/index.db");
    let output = Command::new(env!("CARGO_BIN_EXE_indexer"))
        .arg("--config")
        .arg(&config)
        .arg("--db")
        .arg(&path)
        .arg("prepare")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let c = db::connect(&path).unwrap();
    assert_eq!(
        db::get_meta(&c, "initial_publication_done")
            .unwrap()
            .as_deref(),
        Some("true")
    );
    assert!(db::get_meta(&c, "import_done").unwrap().is_none());
    let completed: i64 = c
        .query_row("SELECT COUNT(*) FROM scan_completed", [], |row| row.get(0))
        .unwrap();
    assert_eq!(completed, 0);
    assert!(String::from_utf8_lossy(&output.stdout).contains("without OSS requests"));

    let legacy = temp.path().join("legacy.db");
    let output = Command::new(env!("CARGO_BIN_EXE_indexer"))
        .arg("--config")
        .arg(&config)
        .arg("--db")
        .arg(&path)
        .arg("run")
        .arg("--legacy-db")
        .arg(&legacy)
        .arg("--broker")
        .arg("http://fixture")
        .arg("--broker-token-file")
        .arg(temp.path().join("unused.token"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("initial import/publication not complete")
    );
    assert!(!legacy.exists());
}
