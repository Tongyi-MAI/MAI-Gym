use mai_dashboard::storage::Storage;
use std::{collections::HashMap, sync::Arc};

#[tokio::test]
async fn root_discovery_accepts_slash_without_allowing_absolute_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("objects");
    std::fs::create_dir_all(root.join("run")).unwrap();
    std::fs::write(root.join("run/meta.json"), b"{}").unwrap();
    let op = opendal::Operator::new(opendal::services::Fs::default().root(root.to_str().unwrap()))
        .unwrap()
        .finish();
    let store = Storage::with_operators(
        HashMap::from([("0".into(), op)]),
        tmp.path().join("cache.db"),
    )
    .unwrap();
    let first = store.list("0", "/", "", "discovery", false).await.unwrap();
    assert!(
        first
            .objects
            .iter()
            .any(|entry| entry.path == "run/" && entry.dir)
    );
    let second = store.list("0", "", "", "discovery", false).await.unwrap();
    assert_eq!(first.objects.len(), second.objects.len());
    assert_eq!(store.metrics()["0:LIST:discovery"]["requests"], 1);
    for prefix in ["/run/", "//", "../", "run/../"] {
        assert!(
            store
                .list("0", prefix, "", "discovery", false)
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn fifty_identical_misses_share_one_origin_read() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("objects");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("artifact"), b"fixture").unwrap();
    let op = opendal::Operator::new(opendal::services::Fs::default().root(root.to_str().unwrap()))
        .unwrap()
        .finish();
    let store = Arc::new(
        Storage::with_operators(
            HashMap::from([("0".into(), op)]),
            tmp.path().join("cache.db"),
        )
        .unwrap(),
    );
    let barrier = Arc::new(tokio::sync::Barrier::new(50));
    let mut tasks = Vec::new();
    for _ in 0..50 {
        let s = store.clone();
        let b = barrier.clone();
        tasks.push(tokio::spawn(async move {
            b.wait().await;
            s.get("0", "artifact", "test").await.unwrap().unwrap().bytes
        }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap(), "fixture");
    }
    assert_eq!(store.metrics()["0:GET:test"]["requests"], 1);
}
#[tokio::test]
async fn missing_objects_are_coalesced_and_disk_survives_restart() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("objects");
    std::fs::create_dir(&root).unwrap();
    let op = opendal::Operator::new(opendal::services::Fs::default().root(root.to_str().unwrap()))
        .unwrap()
        .finish();
    let cache = tmp.path().join("cache.db");
    let make = || {
        Storage::with_operators(HashMap::from([("0".into(), op.clone())]), cache.clone()).unwrap()
    };
    let s = make();
    for _ in 0..10 {
        assert!(s.get("0", "missing", "test").await.unwrap().is_none());
    }
    assert_eq!(s.metrics()["0:GET:test"]["requests"], 1);
    s.checkpoint().await.unwrap();
    drop(s);
    let s = make();
    assert!(s.get("0", "missing", "test").await.unwrap().is_none());
    assert_eq!(s.metrics()["0:GET:test"]["requests"], 1);
}
#[test]
fn traversal_is_rejected() {
    for p in ["../secret", "/absolute", "a/../b", "a\\b", "a//b", "a/./b"] {
        assert!(mai_dashboard::storage::safe(p).is_err());
    }
}
