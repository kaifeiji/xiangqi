use axum::routing::get;
use std::{net::SocketAddr, path::PathBuf};
use tower_http::{cors::CorsLayer, services::ServeDir};

mod api;
mod models;
mod players;
mod session;

use session::AppState;

fn load_local_env() {
    let path = PathBuf::from(".env.local");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches(['"', '\'']);
        if !key.is_empty() && std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    }
}

#[tokio::main]
async fn main() {
    load_local_env();
    let ort_path = std::env::var_os("ORT_DYLIB_PATH")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("LOCALAPPDATA").map(|local_app_data| {
                std::path::PathBuf::from(local_app_data)
                    .join("Xiangqi")
                    .join("onnxruntime.dll")
            })
        })
        .unwrap_or_else(|| std::path::PathBuf::from("onnxruntime.dll"));
    if !ort_path.is_file() {
        eprintln!("ONNX Runtime DLL not found: {}", ort_path.display());
        eprintln!("Set ORT_DYLIB_PATH to the full path of onnxruntime.dll.");
        std::process::exit(1);
    }
    if !ort::init_from(&ort_path)
        .map(|environment| environment.commit())
        .unwrap_or(false)
    {
        eprintln!(
            "failed to initialize ONNX Runtime DLL: {}",
            ort_path.display()
        );
        std::process::exit(1);
    }
    let host = std::env::var("XIANGQI_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned());
    let port = std::env::var("XIANGQI_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8000);
    let dist = PathBuf::from("dist");
    let state = AppState {
        games: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
    };
    let app = api::router(state)
        .route("/health", get(|| async { "ok" }))
        .fallback_service(ServeDir::new(dist))
        .layer(CorsLayer::permissive());
    let address: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid XIANGQI_HOST/XIANGQI_PORT");
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("failed to bind server");
    println!("xiangqi-server listening on http://{address}");
    axum::serve(listener, app).await.expect("server failed");
}
