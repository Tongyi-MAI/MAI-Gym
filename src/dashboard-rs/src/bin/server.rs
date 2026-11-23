use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
};
use clap::Parser;
use mai_dashboard::{
    config::Config,
    db::Db,
    server::{self, App},
    storage::{Page, Storage},
};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    db: PathBuf,
    #[arg(long)]
    cache: PathBuf,
    #[arg(long, default_value = "127.0.0.1:6092")]
    listen: String,
    #[arg(long, default_value = "127.0.0.1:6093")]
    broker_listen: std::net::SocketAddr,
    #[arg(long)]
    broker_token_file: PathBuf,
    #[arg(long)]
    allow_admin: bool,
    #[arg(long)]
    url_base: Option<String>,
}
#[derive(Clone)]
struct Broker {
    storage: Arc<Storage>,
    token: Arc<String>,
}
#[derive(Deserialize)]
struct Request {
    m: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    cursor: String,
    #[serde(default)]
    recursive: bool,
    #[serde(default = "purpose")]
    purpose: String,
    etag: Option<String>,
}
fn purpose() -> String {
    "historical".into()
}
fn authorized(b: &Broker, h: &HeaderMap) -> bool {
    h.get("authorization")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| {
            let expected = format!("Bearer {}", b.token);
            s.len() == expected.len()
                && s.bytes()
                    .zip(expected.bytes())
                    .fold(0u8, |v, (a, b)| v | (a ^ b))
                    == 0
        })
}
async fn object(State(b): State<Broker>, h: HeaderMap, Query(q): Query<Request>) -> Response {
    if !authorized(&b, &h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    match b
        .storage
        .get_checked(
            &q.m,
            &q.path,
            &q.purpose,
            q.etag.as_deref().filter(|s| !s.is_empty()),
        )
        .await
    {
        Ok(Some(v)) => {
            let mut headers = HeaderMap::new();
            if let Some(etag) = v.etag
                && let Ok(v) = etag.parse()
            {
                headers.insert("etag", v);
            }
            if let Some(t) = v.modified {
                headers.insert("x-object-mtime", t.to_string().parse().unwrap());
            }
            (headers, v.bytes).into_response()
        }
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(error) => {
            tracing::warn!("storage broker: {error}");
            (StatusCode::BAD_GATEWAY, "storage unavailable").into_response()
        }
    }
}
async fn list(State(b): State<Broker>, h: HeaderMap, Query(q): Query<Request>) -> Response {
    if !authorized(&b, &h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    match b
        .storage
        .list(&q.m, &q.path, &q.cursor, &q.purpose, q.recursive)
        .await
    {
        Ok(v) => Json::<Page>(v).into_response(),
        Err(error) => {
            tracing::warn!("storage broker: {error}");
            (StatusCode::BAD_GATEWAY, "storage unavailable").into_response()
        }
    }
}
async fn metrics(State(b): State<Broker>, h: HeaderMap) -> Response {
    if !authorized(&b, &h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Json(b.storage.metrics()).into_response()
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    anyhow::ensure!(
        args.broker_listen.ip().is_loopback(),
        "broker must bind loopback"
    );
    let mut cfg = Config::load(&args.config)?;
    if let Some(base) = args.url_base {
        cfg.url_base = base;
    }
    if !cfg.url_base.is_empty() && !cfg.url_base.ends_with('/') {
        cfg.url_base.push('/');
    }
    let cfg = Arc::new(cfg);
    let token = std::fs::read_to_string(&args.broker_token_file)?
        .trim()
        .to_string();
    anyhow::ensure!(
        token.len() >= 32,
        "broker token must be at least 32 characters"
    );
    let storage = Arc::new(Storage::new(&cfg, args.cache)?);
    let checkpoint = storage.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            if let Err(e) = checkpoint.checkpoint().await {
                tracing::error!(error=%e,"cache checkpoint failed");
            }
        }
    });
    let app = server::router(App {
        db: Db::new(args.db),
        cfg,
        storage: storage.clone(),
        http: reqwest::Client::new(),
        allow_admin: args.allow_admin,
    });
    let broker = Router::new()
        .route("/get", get(object))
        .route("/list", get(list))
        .route("/metrics", get(metrics))
        .with_state(Broker {
            storage,
            token: Arc::new(token),
        });
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    let private = tokio::net::TcpListener::bind(args.broker_listen).await?;
    tokio::try_join!(
        async {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
        },
        async {
            axum::serve(private, broker)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
        }
    )?;
    Ok(())
}
