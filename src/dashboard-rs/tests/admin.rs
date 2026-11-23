use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::Query,
    http::Request,
    routing::post,
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
#[tokio::test]
async fn admin_forwarding_preserves_parameters_and_shadow_is_read_only() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let manager = Router::new().route(
        "/admin/scale",
        post(|Query(q): Query<HashMap<String, String>>| async move {
            Json(json!({"size":q.get("size"),"manager_key_leaked":q.contains_key("m")}))
        }),
    );
    let handle = tokio::spawn(async move { axum::serve(listener, manager).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    let state = App {
        db: db::Db::new(path),
        cfg: Arc::new(Config {
            url_base: String::new(),
            managers: vec![Manager {
                label: "test".into(),
                url,
                traj_root: None,
                capacity_mode: None,
                k8s_inventory: None,
            }],
        }),
        storage: Arc::new(
            Storage::with_operators(HashMap::new(), dir.path().join("cache.db")).unwrap(),
        ),
        http: reqwest::Client::new(),
        allow_admin: false,
    };
    let request = || {
        Request::builder()
            .method("POST")
            .uri("/api/admin/scale?m=0&size=7")
            .body(Body::empty())
            .unwrap()
    };
    assert_eq!(
        router(state.clone())
            .oneshot(request())
            .await
            .unwrap()
            .status(),
        403
    );
    let mut live = state;
    live.allow_admin = true;
    let response = router(live).oneshot(request()).await.unwrap();
    assert_eq!(response.status(), 200);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024).await.unwrap()).unwrap();
    assert_eq!(body, json!({"size":"7","manager_key_leaked":false}));
    handle.abort();
}
