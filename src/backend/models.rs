use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct ModelList {
    pub models: Vec<ModelOption>,
}

#[derive(Serialize)]
pub struct ModelOption {
    pub id: String,
    pub name: String,
}

pub async fn list() -> Json<ModelList> {
    Json(ModelList {
        models: collect(std::path::Path::new("models")),
    })
}

pub fn validate(model: &str) -> Option<String> {
    if model == "pikafish" {
        return crate::players::pikafish_command().map(|_| model.to_owned());
    }
    let candidate = std::path::Path::new("models").join(model);
    (candidate.is_file() && candidate.extension().and_then(|value| value.to_str()) == Some("onnx"))
        .then(|| candidate.to_string_lossy().into_owned())
}

fn collect(root: &std::path::Path) -> Vec<ModelOption> {
    let mut models = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return models;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            models.extend(collect(&path));
        } else if path.extension().and_then(|value| value.to_str()) == Some("onnx") {
            let id = path
                .to_string_lossy()
                .replace('\\', "/")
                .trim_start_matches("models/")
                .to_owned();
            models.push(ModelOption {
                id,
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            });
        }
    }
    if crate::players::pikafish_command().is_some() {
        models.push(ModelOption {
            id: "pikafish".to_owned(),
            name: "Pikafish (NNUE + alpha-beta)".to_owned(),
        });
    }
    models
}
