use axum::{extract::State, Json};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

#[derive(Serialize)]
pub struct TrainingCheckpoint {
    pub id: String,
    pub name: String,
    pub metrics: Vec<Value>,
    pub progress: Vec<Value>,
}

pub async fn list(State(state): State<crate::session::AppState>) -> Json<Vec<TrainingCheckpoint>> {
    Json(collect(&state.checkpoint_path))
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect()
}

fn collect(root: &Path) -> Vec<TrainingCheckpoint> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut checkpoints = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_string_lossy().into_owned();
            path.is_dir().then(|| TrainingCheckpoint {
                id: name.clone(),
                name,
                metrics: read_jsonl(&path.join("metrics.jsonl")),
                progress: read_jsonl(&path.join("progress.jsonl")),
            })
        })
        .filter(|checkpoint| !checkpoint.metrics.is_empty() || !checkpoint.progress.is_empty())
        .collect::<Vec<_>>();
    checkpoints.sort_by(|left, right| right.name.cmp(&left.name));
    checkpoints
}