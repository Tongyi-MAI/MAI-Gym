use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use mai_dashboard::{
    config::{Config, Manager},
    db,
    server::{App, router},
    shards,
    storage::Storage,
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tower::ServiceExt;

async fn get(app: &axum::Router, route: &str, pairs: &[(&str, &str)]) -> (u16, Value) {
    let mut url = reqwest::Url::parse(&format!("http://fixture{route}")).unwrap();
    url.query_pairs_mut().extend_pairs(pairs.iter().copied());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("{}?{}", url.path(), url.query().unwrap()))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn config() -> Config {
    Config {
        url_base: String::new(),
        managers: (0..2)
            .map(|_| Manager {
                label: "fixture".into(),
                url: "http://fixture".into(),
                traj_root: Some("oss://fixture/evaluations".into()),
                capacity_mode: None,
                k8s_inventory: None,
            })
            .collect(),
    }
}

fn add(
    c: &rusqlite::Connection,
    manager: &str,
    slug: &str,
    train: &str,
    task: &str,
    score: Option<f64>,
    index: usize,
) {
    db::upsert(c, manager, json!({"id":format!("{slug}/{train}/eval_r00000/{task}/attempt{index}"),"train_id":train,"task_id":task,"attempt":format!("eval_r0_s{index}"),"rollout_id":0,"evaluation":true,"score":score,"mtime":mai_dashboard::now()+index as f64,"created_at":mai_dashboard::now()-100.,"suite":"osworld_test","benchmark_model":slug,"benchmark_run_id":train})).unwrap();
}

fn app(
    path: std::path::PathBuf,
    config: Config,
    cache: std::path::PathBuf,
) -> (axum::Router, Arc<Storage>) {
    let storage = Arc::new(Storage::with_operators(HashMap::new(), cache).unwrap());
    (
        router(App {
            db: db::Db::new(path),
            cfg: Arc::new(config),
            storage: storage.clone(),
            http: reqwest::Client::new(),
            allow_admin: false,
        }),
        storage,
    )
}

#[test]
fn shard_tokens_are_conservative() {
    for (name, base, index) in [
        ("eval_s11_overlap20261003", "eval_overlap20261003", 11),
        ("eval_shard_2", "eval", 2),
        ("eval_shard3_tail", "eval_tail", 3),
        ("s0_eval", "eval", 0),
    ] {
        assert_eq!(shards::shard_name(name), Some((base.into(), index)));
    }
    for name in ["eval_s1x", "eval_s1_s2", "eval", "s1", "eval_sample1"] {
        assert_eq!(shards::shard_name(name), None, "{name}");
    }
}

#[tokio::test]
async fn merged_navigation_preserves_shards_scores_and_original_ids() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let mut c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    let cfg = config();
    add(
        &c,
        "0",
        "model",
        "eval_iter10_s0_suffix",
        "task-a",
        Some(1.),
        0,
    );
    add(
        &c,
        "0",
        "model",
        "eval_iter10_s0_suffix",
        "task-b",
        Some(0.),
        1,
    );
    add(
        &c,
        "0",
        "model",
        "eval_iter10_s1_suffix",
        "task-c",
        Some(1.),
        2,
    );
    add(
        &c,
        "0",
        "model",
        "eval_iter10_s1_suffix",
        "task-a",
        Some(0.),
        3,
    );
    add(
        &c,
        "0",
        "model",
        "eval_iter10_s1_suffix",
        "pending",
        None,
        4,
    );
    add(
        &c,
        "0",
        "model",
        "eval_iter11_s0_suffix",
        "other",
        Some(0.),
        5,
    );
    add(
        &c,
        "0",
        "different-model",
        "eval_iter10_s0_suffix",
        "other",
        Some(0.),
        6,
    );
    add(
        &c,
        "0",
        "model",
        "eval_iter10_s0_different-batch",
        "other",
        Some(0.),
        7,
    );
    add(
        &c,
        "1",
        "model",
        "eval_iter10_s2_suffix",
        "other-manager",
        Some(0.),
        8,
    );
    while db::rebuild_one(&mut c, &cfg).unwrap() {}
    let before: i64 = c
        .query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get(0))
        .unwrap();
    let (app, storage) = app(path, cfg, tmp.path().join("cache.db"));
    let (status, raw) = get(&app, "/api/traj/runs", &[("m", "0")]).await;
    assert_eq!(status, 200);
    assert_eq!(raw["runs"].as_array().unwrap().len(), 5);
    let (status, merged) = get(&app, "/api/traj/runs", &[("m", "0"), ("merge_shards", "1")]).await;
    assert_eq!(status, 200, "{merged}");
    assert_eq!(merged["runs"].as_array().unwrap().len(), 4);
    let run = merged["runs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["shard_count"] == 2)
        .unwrap();
    assert_eq!(run["count"], 5);
    assert_eq!(run["scored"], 4);
    assert_eq!(run["passed"], 2);
    assert_eq!(run["mean_score"], 0.5);
    assert_eq!(run["n_passes"], 1);
    assert_eq!(run["n_tasks_eval"], 4);
    assert_eq!(run["n_scored_eval"], 3);
    assert_eq!(run["n_passed_eval"], 2);
    assert!((run["pass_at_k"].as_f64().unwrap() - 2. / 3.).abs() < 1e-10);
    assert_eq!(run["oss_path"], Value::Null);
    assert_eq!(run["shard_indices"], json!([0, 1]));
    let key = run["run_key"].as_str().unwrap();
    let (status, rollouts) = get(&app, "/api/traj/rollouts", &[("m", "0"), ("run", key)]).await;
    assert_eq!(status, 200, "{rollouts}");
    assert_eq!(rollouts["rollouts"].as_array().unwrap().len(), 1);
    assert_eq!(rollouts["total_attempts"], 5);
    let (_, tasks) = get(
        &app,
        "/api/traj/tasks",
        &[
            ("m", "0"),
            ("run", key),
            ("rollout", "eval_r00000"),
            ("limit", "2"),
        ],
    )
    .await;
    assert_eq!(tasks["tasks"].as_array().unwrap().len(), 2);
    assert!(tasks["next_cursor"].is_string());
    assert_eq!(tasks["total_attempts"], 5);
    let (status, attempts) = get(
        &app,
        "/api/traj/list",
        &[
            ("m", "0"),
            ("run", key),
            ("rollout", "eval_r00000"),
            ("task_exact", "task-a"),
        ],
    )
    .await;
    assert_eq!(status, 200, "{attempts}");
    assert_eq!(attempts["total"], 2);
    let items = attempts["items"].as_array().unwrap();
    assert!(
        items
            .iter()
            .all(|r| r["id"].as_str().unwrap().starts_with("model/eval_iter10_s"))
    );
    assert_eq!(
        items
            .iter()
            .map(|r| r["source_shard"].as_u64().unwrap())
            .collect::<HashSet<_>>(),
        HashSet::from([0, 1])
    );
    let (_, all) = get(
        &app,
        "/api/traj/list",
        &[("m", "0"), ("run", key), ("page_size", "2")],
    )
    .await;
    assert_eq!(all["total"], 5);
    assert_eq!(all["items"].as_array().unwrap().len(), 2);
    let raw_key = run["shards"][0]["run_key"].as_str().unwrap();
    let (status, export) = get(&app, "/api/traj/results", &[("m", "0"), ("run", key)]).await;
    assert_eq!(status, 200, "{export}");
    assert_eq!(export["runs"].as_array().unwrap().len(), 1);
    let exported = export["runs"][0]["rollouts"][0]["tasks"]
        .as_array()
        .unwrap();
    assert_eq!(exported.len(), 4);
    assert_eq!(
        exported.iter().find(|t| t["task_id"] == "task-a").unwrap()["success"],
        true
    );
    assert!(
        exported
            .iter()
            .all(|t| t["traj_path"].as_str().unwrap().contains("_s"))
    );
    let (_, original) = get(&app, "/api/traj/rollouts", &[("m", "0"), ("run", raw_key)]).await;
    assert_eq!(original["total_attempts"], 2);
    let (status, other_manager) =
        get(&app, "/api/traj/rollouts", &[("m", "1"), ("run", key)]).await;
    assert_eq!(status, 200);
    assert_eq!(other_manager["total_attempts"], 1);
    assert_eq!(
        other_manager["shards"][0]["train_id"],
        "eval_iter10_s2_suffix"
    );
    assert_eq!(
        before,
        c.query_row("SELECT COUNT(*) FROM attempts", [], |r| r.get::<_, i64>(0))
            .unwrap()
    );
    assert!(storage.metrics().as_object().unwrap().is_empty());
}

#[tokio::test]
async fn grouping_precedes_pagination_and_scopes_cursors() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.db");
    let c = db::connect(&path).unwrap();
    db::schema(&c).unwrap();
    for family in 0..3 {
        for shard in 0..60 {
            let id = format!("eval_batch{family}_s{shard}_tail");
            let row = json!({"run_key":id,"train_id":id,"count":1,"scored":1,"passed":1,"mean_score":1.,"last_mtime":1000,"suite":"suite"});
            c.execute(
                "INSERT INTO views VALUES('0','run',?,'',?,?,?)",
                rusqlite::params![id, id, 1000, row.to_string()],
            )
            .unwrap();
        }
    }
    let (app, _) = app(path, config(), tmp.path().join("cache.db"));
    let (_, first) = get(
        &app,
        "/api/traj/runs",
        &[("merge_shards", "1"), ("limit", "2")],
    )
    .await;
    assert_eq!(first["total_runs"], 3);
    assert!(
        first["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["shard_count"] == 60 && r["count"] == 60)
    );
    let cursor = first["next_cursor"].as_str().unwrap();
    let (status, next) = get(
        &app,
        "/api/traj/runs",
        &[("merge_shards", "1"), ("limit", "2"), ("cursor", cursor)],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(next["runs"].as_array().unwrap().len(), 1);
    assert!(next["next_cursor"].is_null());
    assert_eq!(
        get(
            &app,
            "/api/traj/runs",
            &[("limit", "2"), ("cursor", cursor)]
        )
        .await
        .0,
        400
    );
    assert_eq!(
        get(
            &app,
            "/api/traj/runs",
            &[
                ("merge_shards", "1"),
                ("limit", "2"),
                ("cursor", cursor),
                ("task", "batch1")
            ]
        )
        .await
        .0,
        400
    );
    let (_, filtered) = get(
        &app,
        "/api/traj/runs",
        &[("merge_shards", "1"), ("task", "batch1_s59")],
    )
    .await;
    assert_eq!(filtered["runs"].as_array().unwrap().len(), 1);
    assert_eq!(filtered["runs"][0]["shard_count"], 60);
}
