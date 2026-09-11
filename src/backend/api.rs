use crate::session::{self, AppState};
use axum::{
    routing::{get, post},
    Router,
};

pub fn router(state: AppState) -> Router {
    session::start_cleanup_task(&state);
    Router::new()
        .route("/api/models", get(crate::models::list))
        .route("/api/training/checkpoints", get(crate::training::list))
        .route("/api/tournaments", get(crate::benchmark::list_tournaments).post(crate::benchmark::create_tournament))
        .route("/api/tournaments/{id}", get(crate::benchmark::get_tournament).post(crate::benchmark::resume_tournament).delete(crate::benchmark::pause_tournament))
        .route("/api/tournaments/{id}/benchmarks", get(crate::benchmark::tournament_benchmarks))
        .route("/api/benchmarks", get(crate::benchmark::list).post(crate::benchmark::create))
        .route("/api/benchmarks/{id}", get(crate::benchmark::get).delete(crate::benchmark::cancel))
        .route("/api/benchmarks/{id}/games", get(crate::benchmark::games))
        .route("/api/benchmarks/{id}/games/{number}", get(crate::benchmark::game))
        .route("/api/benchmarks/{id}/resume", post(crate::benchmark::resume))
        .route("/api/games", post(session::create_game))
        .route("/api/games/{id}", get(session::get_game))
        .route("/api/games/{id}/move", post(session::apply_move))
        .route("/api/games/{id}/step", post(session::step_model))
        .route("/api/games/{id}/undo", post(session::undo))
        .route("/api/games/{id}/close", post(session::close_game))
        .with_state(state)
}
