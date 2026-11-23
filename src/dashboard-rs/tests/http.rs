use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use mai_dashboard::{
    config::{Config, Manager},
    db,
    server::{App, router},
    storage::Storage,
};
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Arc};
use tower::ServiceExt;
async fn get(app: &axum::Router, path: &str) -> (u16, Value) {
    let r = app
        .clone()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = r.status().as_u16();
    let bytes = to_bytes(r.into_body(), 20 * 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
#[tokio::test]
async fn metadata_routes_paginate_without_origin_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("index.db");
    let c = db::connect(&p).unwrap();
    db::schema(&c).unwrap();
    for i in 0..1253 {
        let id = format!("run{i:04}");
        c.execute(
            "INSERT INTO views VALUES('0','run',?,'',?,?,?)",
            rusqlite::params![
                id,
                id,
                1000,
                json!({"run_key":id,"count":1,"last_mtime":1000}).to_string()
            ],
        )
        .unwrap();
    }
    db::set_meta(&c, "import_done", "true").unwrap();
    let cfg = Arc::new(Config {
        url_base: String::new(),
        managers: vec![Manager {
            label: "fixture".into(),
            url: "http://fixture".into(),
            traj_root: None,
            capacity_mode: None,
            k8s_inventory: None,
        }],
    });
    let storage =
        Arc::new(Storage::with_operators(HashMap::new(), tmp.path().join("cache.db")).unwrap());
    let app = router(App {
        db: db::Db::new(p),
        cfg,
        storage: storage.clone(),
        http: reqwest::Client::new(),
        allow_admin: false,
    });
    let (status, first) = get(&app, "/api/traj/runs?m=0&limit=50").await;
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["runs"].as_array().unwrap().len(), 50);
    let mut ids = std::collections::HashSet::new();
    let mut current = first;
    loop {
        for r in current["runs"].as_array().unwrap() {
            assert!(ids.insert(r["run_key"].as_str().unwrap().to_string()));
        }
        let Some(cursor) = current["next_cursor"].as_str() else {
            break;
        };
        let mut url = reqwest::Url::parse("http://fixture/api/traj/runs").unwrap();
        url.query_pairs_mut()
            .append_pair("m", "0")
            .append_pair("limit", "50")
            .append_pair("cursor", cursor);
        let (status, v) = get(&app, &format!("{}?{}", url.path(), url.query().unwrap())).await;
        assert_eq!(status, 200, "{v}");
        current = v;
    }
    assert_eq!(ids.len(), 1253);
    for path in [
        "/api/overview?m=0",
        "/api/traj/info?m=0",
        "/api/traj/tasks?m=0&limit=50",
        "/api/traj/list?m=0",
        "/api/traj/rollouts?m=0&run=run0000",
        "/api/benchmarks",
        "/api/history?m=0",
        "/api/sessions?m=0&limit=50",
        "/api/health?m=0",
    ] {
        let (status, v) = get(&app, path).await;
        assert_eq!(status, 200, "{path}: {v}");
    }
    assert!(storage.metrics().as_object().unwrap().is_empty());
    let (_, v) = get(&app, "/api/traj/runs?m=0").await;
    assert_eq!(v["runs"].as_array().unwrap().len(), 1253);
}
