use crate::players::Player;
use crate::session::{ApiError, AppState};
use axum::{extract::{Path, State}, http::StatusCode, Json};
use chrono::{Local, TimeZone};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path as FilePath, PathBuf};
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;
use xiangqi::game::Game;
use xiangqi::openings::MAINSTREAM_OPENINGS;

#[derive(Deserialize)]
pub struct CreateRequest {
    pub first_model: String,
    pub second_model: String,
    pub mcts_simulations: usize,
    #[serde(default)]
    pub games: Option<usize>,
}

#[derive(Deserialize)]
pub struct TournamentCreateRequest {
    pub models: Vec<String>,
    pub mcts_simulations: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Tournament {
    pub id: Uuid,
    pub models: Vec<String>,
    pub mcts_simulations: usize,
    pub games_per_match: usize,
    pub benchmark_ids: Vec<Uuid>,
    pub status: String,
    pub created_at_ms: u128,
    pub finished_at_ms: Option<u128>,
    #[serde(default)]
    pub ratings: HashMap<String, f64>,
    #[serde(default)]
    pub rated_benchmark_ids: Vec<Uuid>,
    #[serde(default)]
    pub rated_game_keys: Vec<String>,
}

#[derive(Serialize)]
pub struct TournamentResponse {
    pub id: Uuid,
    pub models: Vec<String>,
    pub mcts_simulations: usize,
    pub games_per_match: usize,
    pub benchmark_ids: Vec<Uuid>,
    pub status: String,
    pub created_at_ms: u128,
    pub finished_at_ms: Option<u128>,
    pub ratings: HashMap<String, f64>,
}

impl Tournament {
    fn response(&self) -> TournamentResponse {
        TournamentResponse { id: self.id, models: self.models.clone(), mcts_simulations: self.mcts_simulations, games_per_match: self.games_per_match, benchmark_ids: self.benchmark_ids.clone(), status: self.status.clone(), created_at_ms: self.created_at_ms, finished_at_ms: self.finished_at_ms, ratings: self.ratings.clone() }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameResult {
    pub number: usize,
    pub opening_move: String,
    #[serde(default)]
    pub initial_fen: String,
    #[serde(default)]
    pub snapshots: Vec<GameSnapshot>,
    #[serde(default)]
    pub started_at_ms: u128,
    #[serde(default)]
    pub finished_at_ms: u128,
    pub result: Option<String>,
    pub total_plies: usize,
    pub rule60: u16,
    pub elapsed_ms: u128,
    pub error: Option<String>,
    #[serde(default)]
    pub repetition_cycle_plies: Option<(usize, usize)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameSnapshot {
    pub fen: String,
    pub side_to_move: String,
    pub turn: usize,
    pub rule60: u16,
    pub result: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcts_debug: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_debug: Option<Value>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Benchmark {
    pub id: Uuid,
    pub first_model: String,
    pub second_model: String,
    pub mcts_simulations: usize,
    pub games_requested: usize,
    #[serde(default)]
    pub started_at_ms: u128,
    #[serde(default)]
    pub finished_at_ms: Option<u128>,
    #[serde(default)]
    pub paused: bool,
    pub cancelled: bool,
    pub failed: bool,
    #[serde(default)]
    pub queued: bool,
    pub games: Vec<GameResult>,
}

impl Benchmark {
    fn completed_games(&self) -> usize {
        self.games.iter().filter(|game| game.result.is_some()).count()
    }
    fn status(&self) -> &'static str {
        if self.failed { "failed" }
        else if self.paused || self.cancelled { "paused" }
        else if self.completed_games() == self.games_requested { "completed" }
        else if self.queued { "queued" }
        else { "running" }
    }

    fn summary(&self) -> BenchmarkSummary {
        let (mut first_wins, mut second_wins, mut draws) = (0, 0, 0);
        for game in &self.games {
            match game.result.as_deref() {
                Some(result) if result.starts_with("red_win") => {
                    if game.number % 2 == 1 { first_wins += 1 } else { second_wins += 1 }
                }
                Some(result) if result.starts_with("black_win") => {
                    if game.number % 2 == 0 { first_wins += 1 } else { second_wins += 1 }
                }
                Some(_) => draws += 1,
                None => {}
            }
        }
        BenchmarkSummary { id: self.id, first_model: self.first_model.clone(), second_model: self.second_model.clone(), mcts_simulations: self.mcts_simulations, games_requested: self.games_requested, started_at_ms: self.started_at_ms, finished_at_ms: self.finished_at_ms, status: self.status(), first_wins, second_wins, draws, games_completed: self.completed_games() }
    }
}

#[derive(Serialize)]
    pub struct BenchmarkSummary {
    pub id: Uuid,
    pub first_model: String,
    pub second_model: String,
    pub mcts_simulations: usize,
    pub games_requested: usize,
    pub started_at_ms: u128,
    pub finished_at_ms: Option<u128>,
    pub status: &'static str,
    pub first_wins: usize,
    pub second_wins: usize,
    pub draws: usize,
    pub games_completed: usize,
}

#[derive(Serialize)]
pub struct GameSummary {
    pub number: usize,
    pub opening_move: String,
    pub started_at_ms: u128,
    pub finished_at_ms: u128,
    pub result: Option<String>,
    pub total_plies: usize,
    pub rule60: u16,
    pub elapsed_ms: u128,
    pub error: Option<String>,
    pub repetition_cycle_plies: Option<(usize, usize)>,
}

pub type Benchmarks = Arc<RwLock<HashMap<Uuid, Benchmark>>>;
pub type Controls = Arc<RwLock<HashMap<Uuid, Arc<AtomicBool>>>>;
pub type Tournaments = Arc<RwLock<HashMap<Uuid, Tournament>>>;

pub fn load_tournaments(path: &FilePath) -> HashMap<Uuid, Tournament> {
    let mut tournaments = HashMap::new();
    let Ok(entries) = std::fs::read_dir(path) else { return tournaments };
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else { continue };
        if let Ok(mut tournament) = serde_json::from_str::<Tournament>(&text) {
            for model in &tournament.models {
                tournament.ratings.entry(model.clone()).or_insert(1500.0);
            }
            tournaments.insert(tournament.id, tournament);
        }
    }
    tournaments
}

pub fn load(path: &FilePath) -> HashMap<Uuid, Benchmark> {
    let mut benchmarks = HashMap::new();
    let Ok(entries) = std::fs::read_dir(path) else { return benchmarks };
    for entry in entries.flatten() {
        let source = entry.path();
        let Ok(text) = std::fs::read_to_string(&source) else { continue };
        if let Ok(mut benchmark) = serde_json::from_str::<Benchmark>(&text) {
            if !benchmark.failed && benchmark.completed_games() < benchmark.games_requested {
                benchmark.paused = true;
            }
            let target = path.join(file_name(&benchmark));
            if source != target && !target.exists() {
                let _ = std::fs::rename(source, target);
            }
            benchmarks.insert(benchmark.id, benchmark);
        }
    }
    benchmarks
}

pub async fn create(State(state): State<AppState>, Json(request): Json<CreateRequest>) -> Result<Json<BenchmarkSummary>, ApiError> {
    if request.first_model == request.second_model { return Err(ApiError::bad_request("benchmark models must differ")); }
    if ![0, 1000, 5000, 10000].contains(&request.mcts_simulations) { return Err(ApiError::bad_request("invalid mcts_simulations")); }
    let first_path = crate::models::validate(&request.first_model).ok_or_else(|| ApiError::bad_request("first model not found"))?;
    let second_path = crate::models::validate(&request.second_model).ok_or_else(|| ApiError::bad_request("second model not found"))?;
    let games_requested = request.games.unwrap_or(MAINSTREAM_OPENINGS.len() * 2);
    if games_requested == 0 || games_requested > MAINSTREAM_OPENINGS.len() * 2 || games_requested % 2 != 0 {
        return Err(ApiError::bad_request("games must be a positive even number within the opening book"));
    }
    let benchmark = Benchmark { id: Uuid::new_v4(), first_model: request.first_model.clone(), second_model: request.second_model.clone(), mcts_simulations: request.mcts_simulations, games_requested, started_at_ms: now_ms(), finished_at_ms: None, paused: false, cancelled: false, failed: false, queued: false, games: Vec::new() };
    save(&state.benchmark_path, &benchmark).map_err(ApiError::bad_request)?;
    let response = benchmark.summary();
    let id = benchmark.id;
    state.benchmarks.write().await.insert(id, benchmark);
    let cancelled = Arc::new(AtomicBool::new(false));
    state.benchmark_controls.write().await.insert(id, Arc::clone(&cancelled));
    let benchmarks = Arc::clone(&state.benchmarks);
    let controls = Arc::clone(&state.benchmark_controls);
    let path = state.benchmark_path.clone();
    let queue = Arc::clone(&state.benchmark_queue);
    let active = Arc::clone(&state.active_benchmark);
    let tournaments = Arc::clone(&state.tournaments);
    tokio::spawn(async move {
        let _queue_guard = queue.lock().await;
        *active.write().await = Some(id);
        mark_started(&benchmarks, &path, id).await;
        let _ = tokio::task::spawn_blocking(move || run(request, first_path, second_path, id, path, benchmarks, controls, cancelled, tournaments)).await;
        if *active.read().await == Some(id) { *active.write().await = None; }
    });
    Ok(Json(response))
}

fn update_elo(ratings: &mut HashMap<String, f64>, first_model: &str, second_model: &str, first_score: f64) {
    let first_rating = *ratings.entry(first_model.to_owned()).or_insert(1500.0);
    let second_rating = *ratings.entry(second_model.to_owned()).or_insert(1500.0);
    let expected = 1.0 / (1.0 + 10.0_f64.powf((second_rating - first_rating) / 400.0));
    ratings.insert(first_model.to_owned(), first_rating + 32.0 * (first_score - expected));
    ratings.insert(second_model.to_owned(), second_rating + 32.0 * ((1.0 - first_score) - (1.0 - expected)));
}

pub async fn create_tournament(State(state): State<AppState>, Json(request): Json<TournamentCreateRequest>) -> Result<Json<TournamentResponse>, ApiError> {
    if request.models.len() < 2 {
        return Err(ApiError::bad_request("tournament requires at least two models"));
    }
    let mut unique = request.models.clone();
    unique.sort();
    unique.dedup();
    if unique.len() != request.models.len() {
        return Err(ApiError::bad_request("tournament models must be distinct"));
    }
    if ![0, 1000, 5000, 10000].contains(&request.mcts_simulations) {
        return Err(ApiError::bad_request("invalid mcts_simulations"));
    }
    let paths = request.models.iter().map(|model| {
        crate::models::validate(model).ok_or_else(|| ApiError::bad_request(format!("model not found: {model}")))
    }).collect::<Result<Vec<_>, _>>()?;
    let ratings = request.models.iter().map(|model| (model.clone(), 1500.0)).collect();
    let tournament = Tournament { id: Uuid::new_v4(), models: request.models.clone(), mcts_simulations: request.mcts_simulations, games_per_match: 2, benchmark_ids: Vec::new(), status: "running".to_owned(), created_at_ms: now_ms(), finished_at_ms: None, ratings, rated_benchmark_ids: Vec::new(), rated_game_keys: Vec::new() };
    let tournament_id = tournament.id;
    let mut tournament = tournament;
    for first in 0..request.models.len() {
        for second in (first + 1)..request.models.len() {
            let benchmark = Benchmark { id: Uuid::new_v4(), first_model: request.models[first].clone(), second_model: request.models[second].clone(), mcts_simulations: request.mcts_simulations, games_requested: 2, started_at_ms: now_ms(), finished_at_ms: None, paused: false, cancelled: false, failed: false, queued: true, games: Vec::new() };
            save(&state.benchmark_path, &benchmark).map_err(ApiError::bad_request)?;
            let id = benchmark.id;
            tournament.benchmark_ids.push(id);
            state.benchmarks.write().await.insert(id, benchmark);
            let paused = Arc::new(AtomicBool::new(false));
            state.benchmark_controls.write().await.insert(id, Arc::clone(&paused));
            let benchmarks = Arc::clone(&state.benchmarks);
            let controls = Arc::clone(&state.benchmark_controls);
            let path = state.benchmark_path.clone();
            let queue = Arc::clone(&state.benchmark_queue);
            let active = Arc::clone(&state.active_benchmark);
            let tournaments = Arc::clone(&state.tournaments);
            let match_request = CreateRequest { first_model: request.models[first].clone(), second_model: request.models[second].clone(), mcts_simulations: request.mcts_simulations, games: Some(2) };
            let first_path = paths[first].clone();
            let second_path = paths[second].clone();
            tokio::spawn(async move {
                let _queue_guard = queue.lock().await;
                *active.write().await = Some(id);
                mark_started(&benchmarks, &path, id).await;
                let _ = tokio::task::spawn_blocking(move || run(match_request, first_path, second_path, id, path, benchmarks, controls, paused, tournaments)).await;
                if *active.read().await == Some(id) { *active.write().await = None; }
            });
        }
    }
    save_tournament(&state.benchmark_path, &tournament).map_err(ApiError::bad_request)?;
    state.tournaments.write().await.insert(tournament_id, tournament.clone());
    Ok(Json(tournament.response()))
}

pub async fn list_tournaments(State(state): State<AppState>) -> Json<Vec<TournamentResponse>> {
    let mut tournaments: Vec<_> = state.tournaments.read().await.values().map(Tournament::response).collect();
    tournaments.sort_by_key(|tournament| std::cmp::Reverse(tournament.created_at_ms));
    Json(tournaments)
}

pub async fn get_tournament(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<Json<TournamentResponse>, ApiError> {
    state.tournaments.read().await.get(&id).map(|tournament| Json(tournament.response())).ok_or_else(ApiError::not_found)
}

pub async fn tournament_benchmarks(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<Json<Vec<BenchmarkSummary>>, ApiError> {
    let ids = state.tournaments.read().await.get(&id).map(|tournament| tournament.benchmark_ids.clone()).ok_or_else(ApiError::not_found)?;
    let active = state.active_benchmark.read().await;
    let jobs = state.benchmarks.read().await;
    Ok(Json(ids.into_iter().filter_map(|id| jobs.get(&id).map(|benchmark| summary_with_activity(benchmark.clone(), *active == Some(id)))).collect()))
}

pub async fn pause_tournament(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<StatusCode, ApiError> {
    let ids = state.tournaments.read().await.get(&id).map(|tournament| tournament.benchmark_ids.clone()).ok_or_else(ApiError::not_found)?;
    for benchmark_id in ids {
        if let Some(control) = state.benchmark_controls.read().await.get(&benchmark_id).cloned() {
            control.store(true, Ordering::Relaxed);
        }
        let mut jobs = state.benchmarks.write().await;
        if let Some(job) = jobs.get_mut(&benchmark_id) {
            if !job.failed && job.completed_games() < job.games_requested {
                job.paused = true;
                job.cancelled = false;
                save(&state.benchmark_path, job).map_err(ApiError::bad_request)?;
            }
        }
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn list(State(state): State<AppState>) -> Json<Vec<BenchmarkSummary>> {
    let active = state.active_benchmark.read().await;
    let mut benchmarks: Vec<_> = state.benchmarks.read().await.values().cloned().map(|benchmark| {
        let is_active = *active == Some(benchmark.id);
        summary_with_activity(benchmark, is_active)
    }).collect();
    benchmarks.sort_by_key(|benchmark| std::cmp::Reverse(benchmark.started_at_ms));
    Json(benchmarks)
}

pub async fn get(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<Json<BenchmarkSummary>, ApiError> {
    let active = state.active_benchmark.read().await;
    state.benchmarks.read().await.get(&id).map(|job| Json(summary_with_activity(job.clone(), *active == Some(id)))).ok_or_else(ApiError::not_found)
}

fn summary_with_activity(benchmark: Benchmark, active: bool) -> BenchmarkSummary {
    let mut summary = benchmark.summary();
    if summary.status == "running" && !active {
        summary.status = "queued";
    }
    summary
}

pub async fn games(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<Json<Vec<GameSummary>>, ApiError> {
    let jobs = state.benchmarks.read().await;
    let job = jobs.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(job.games.iter().map(|game| GameSummary {
        number: game.number,
        opening_move: game.opening_move.clone(),
        started_at_ms: game.started_at_ms,
        finished_at_ms: game.finished_at_ms,
        result: game.result.clone(),
        total_plies: game.total_plies,
        rule60: game.rule60,
        elapsed_ms: game.elapsed_ms,
        error: game.error.clone(),
        repetition_cycle_plies: game.repetition_cycle_plies,
    }).collect()))
}

pub async fn game(State(state): State<AppState>, Path((id, number)): Path<(Uuid, usize)>) -> Result<Json<GameResult>, ApiError> {
    state.benchmarks.read().await.get(&id)
        .and_then(|job| job.games.iter().find(|game| game.number == number))
        .cloned().map(Json).ok_or_else(ApiError::not_found)
}

pub async fn cancel(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<StatusCode, ApiError> {
    if let Some(control) = state.benchmark_controls.read().await.get(&id).cloned() {
        control.store(true, Ordering::Relaxed);
    }
    let mut jobs = state.benchmarks.write().await;
    let job = jobs.get_mut(&id).ok_or_else(ApiError::not_found)?;
    if !job.failed && job.games.len() < job.games_requested {
        job.paused = true;
        job.cancelled = false;
        save(&state.benchmark_path, job).map_err(ApiError::bad_request)?;
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn resume(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<StatusCode, ApiError> {
    resume_one(&state, id).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn resume_tournament(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<StatusCode, ApiError> {
    let ids = state.tournaments.read().await.get(&id).map(|tournament| tournament.benchmark_ids.clone()).ok_or_else(ApiError::not_found)?;
    for benchmark_id in ids {
        let resumable = state.benchmarks.read().await.get(&benchmark_id)
            .is_some_and(|benchmark| benchmark.completed_games() < benchmark.games_requested);
        if resumable {
            resume_one(&state, benchmark_id).await?;
        }
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn resume_one(state: &AppState, id: Uuid) -> Result<(), ApiError> {
    let is_active = *state.active_benchmark.read().await == Some(id);
    if is_active {
        let control = state.benchmark_controls.read().await.get(&id).cloned();
        if let Some(control) = control {
        control.store(false, Ordering::Relaxed);
        let mut jobs = state.benchmarks.write().await;
        let job = jobs.get_mut(&id).ok_or_else(ApiError::not_found)?;
        job.paused = false;
        job.cancelled = false;
        save(&state.benchmark_path, job).map_err(ApiError::bad_request)?;
        return Ok(());
        }
    }

    let mut jobs = state.benchmarks.write().await;
    let job = jobs.get_mut(&id).ok_or_else(ApiError::not_found)?;
    if job.completed_games() == job.games_requested {
        return Err(ApiError::bad_request("benchmark cannot be resumed"));
    }
    let request = CreateRequest { first_model: job.first_model.clone(), second_model: job.second_model.clone(), mcts_simulations: job.mcts_simulations, games: Some(job.games_requested) };
    let first_path = crate::models::validate(&request.first_model).ok_or_else(|| ApiError::bad_request("first model not found"))?;
    let second_path = crate::models::validate(&request.second_model).ok_or_else(|| ApiError::bad_request("second model not found"))?;
    job.paused = false;
    job.cancelled = false;
    job.failed = false;
    if let Some(game) = job.games.last_mut() {
        if game.result.is_none() {
            game.error = None;
        }
    }
    save(&state.benchmark_path, job).map_err(ApiError::bad_request)?;
    drop(jobs);

    let paused = Arc::new(AtomicBool::new(false));
    state.benchmark_controls.write().await.insert(id, Arc::clone(&paused));
    let benchmarks = Arc::clone(&state.benchmarks);
    let controls = Arc::clone(&state.benchmark_controls);
    let path = state.benchmark_path.clone();
    let queue = Arc::clone(&state.benchmark_queue);
    let active = Arc::clone(&state.active_benchmark);
    let tournaments = Arc::clone(&state.tournaments);
    tokio::spawn(async move {
        let _queue_guard = queue.lock().await;
        *active.write().await = Some(id);
        mark_started(&benchmarks, &path, id).await;
        let _ = tokio::task::spawn_blocking(move || run(request, first_path, second_path, id, path, benchmarks, controls, paused, tournaments)).await;
        if *active.read().await == Some(id) { *active.write().await = None; }
    });
    Ok(())
}

async fn mark_started(benchmarks: &Benchmarks, path: &FilePath, id: Uuid) {
    let mut jobs = benchmarks.write().await;
    if let Some(job) = jobs.get_mut(&id) {
        job.queued = false;
        let _ = save(path, job);
    }
}

fn run(request: CreateRequest, first_path: String, second_path: String, id: Uuid, path: PathBuf, benchmarks: Benchmarks, controls: Controls, paused: Arc<AtomicBool>, tournaments: Tournaments) {
    loop {
        if paused.load(Ordering::Relaxed) { break; }
        let index = {
            let jobs = benchmarks.blocking_read();
            let Some(job) = jobs.get(&id) else { return };
            if job.failed || job.completed_games() == job.games_requested { break; }
            if job.games.last().is_some_and(|game| game.result.is_none() && game.error.is_none()) { job.games.len() - 1 } else { job.games.len() }
        };
        let first_red = index % 2 == 0;
        let opening = &MAINSTREAM_OPENINGS[index / 2];
        let (red_path, black_path) = if first_red { (&first_path, &second_path) } else { (&second_path, &first_path) };
        if let Err(error) = play(red_path, black_path, request.mcts_simulations, index, opening.movement, opening.chinese_notation, &paused, id, &path, &benchmarks) {
            let mut jobs = benchmarks.blocking_write();
            let Some(job) = jobs.get_mut(&id) else { return };
            job.failed = true;
            job.paused = false;
            if let Some(game) = job.games.get_mut(index) {
                game.error = Some(error);
                game.finished_at_ms = now_ms();
                game.elapsed_ms = game.finished_at_ms.saturating_sub(game.started_at_ms);
            }
            let _ = save(&path, job);
            break;
        }
    }
    if let Some(job) = benchmarks.blocking_write().get_mut(&id) {
        if paused.load(Ordering::Relaxed) {
            job.paused = true;
            job.cancelled = false;
        } else if job.completed_games() == job.games_requested || job.failed {
            job.finished_at_ms = Some(now_ms());
            job.paused = false;
        }
        let _ = save(&path, job);
    }
    update_tournament_for_benchmark(id, &benchmarks, &tournaments, &path);
    controls.blocking_write().remove(&id);
}

fn update_tournament_for_benchmark(id: Uuid, benchmarks: &Benchmarks, tournaments: &Tournaments, path: &FilePath) {
    let Some(tournament_id) = tournaments.blocking_read().iter()
        .find(|(_, tournament)| tournament.benchmark_ids.contains(&id))
        .map(|(tournament_id, _)| *tournament_id) else { return };
    let mut tournament_jobs = tournaments.blocking_write();
    let Some(tournament) = tournament_jobs.get_mut(&tournament_id) else { return };
    let benchmark_jobs = benchmarks.blocking_read();
    let Some(benchmark) = benchmark_jobs.get(&id) else { return };
    for game in &benchmark.games {
        if game.result.is_none() { continue }
        let key = format!("{}:{}", id, game.number);
        if !tournament.rated_game_keys.contains(&key) {
            let first_score = if game.result.as_deref().is_some_and(|result| result.starts_with("draw")) { 0.5 } else if game.result.as_deref().is_some_and(|result| result.starts_with("red_win")) == (game.number % 2 == 1) { 1.0 } else { 0.0 };
            update_elo(&mut tournament.ratings, &benchmark.first_model, &benchmark.second_model, first_score);
            tournament.rated_game_keys.push(key);
        }
    }
    let related = tournament.benchmark_ids.iter().filter_map(|id| benchmark_jobs.get(id));
    if related.clone().any(|benchmark| benchmark.failed) {
        tournament.status = "failed".to_owned();
    } else if related.clone().all(|benchmark| benchmark.completed_games() == benchmark.games_requested) {
        tournament.status = "completed".to_owned();
        tournament.finished_at_ms = Some(now_ms());
    } else if related.clone().any(|benchmark| benchmark.status() == "running" || benchmark.status() == "queued") {
        tournament.status = "running".to_owned();
        tournament.finished_at_ms = None;
    } else {
        tournament.status = "paused".to_owned();
    }
    let _ = save_tournament(path, tournament);
}

fn play(
    red_path: &str,
    black_path: &str,
    simulations: usize,
    game_index: usize,
    opening: (u8, u8),
    opening_notation: &str,
    paused: &AtomicBool,
    id: Uuid,
    path: &FilePath,
    benchmarks: &Benchmarks,
) -> Result<(), String> {
    let stored = ensure_game_record(game_index, opening, opening_notation, id, path, benchmarks)?;
    let mut game = restore_game(&stored.initial_fen, &stored.snapshots)?;
    if stored.snapshots.is_empty() {
        game.apply(opening.0, opening.1)?;
        update_game_record(id, game_index, &game, None, None, path, benchmarks)?;
    }
    let mut players = [Player::from_model(red_path)?, Player::from_model(black_path)?];
    while !game.is_finished() {
        if paused.load(Ordering::Relaxed) {
            return Ok(());
        }
        let player_index = usize::from(game.side_to_move() == "b");
        let searched_fen = game.fen();
        let searched_side = game.side_to_move().to_owned();
        let result = players[player_index].choose_move(&game, simulations)?;
        let movement = result.movement;
        game.apply(movement.0, movement.1)?;
        let (mcts_debug, policy_debug) = search_debug(result, simulations, searched_fen, searched_side);
        update_game_record(id, game_index, &game, mcts_debug, policy_debug, path, benchmarks)?;
    }
    Ok(())
}

fn ensure_game_record(game_index: usize, opening: (u8, u8), opening_notation: &str, id: Uuid, path: &FilePath, benchmarks: &Benchmarks) -> Result<GameResult, String> {
    let mut jobs = benchmarks.blocking_write();
    let job = jobs.get_mut(&id).ok_or_else(|| "benchmark not found".to_owned())?;
    if job.games.len() == game_index {
        let opening_move = format!("{}（{}）", format_move(opening), opening_notation);
        job.games.push(GameResult {
            number: game_index + 1,
            opening_move,
            initial_fen: xiangqi::START_FEN.to_owned(),
            snapshots: Vec::new(),
            started_at_ms: now_ms(),
            finished_at_ms: 0,
            result: None,
            total_plies: 0,
            rule60: 0,
            elapsed_ms: 0,
            error: None,
            repetition_cycle_plies: None,
        });
        let _ = save(path, job);
    }
    job.games.get(game_index).cloned().ok_or_else(|| "benchmark game not found".to_owned())
}

fn update_game_record(id: Uuid, game_index: usize, game: &Game, mcts_debug: Option<Value>, policy_debug: Option<Value>, path: &FilePath, benchmarks: &Benchmarks) -> Result<(), String> {
    let mut jobs = benchmarks.blocking_write();
    let job = jobs.get_mut(&id).ok_or_else(|| "benchmark not found".to_owned())?;
    let record = job.games.get_mut(game_index).ok_or_else(|| "benchmark game not found".to_owned())?;
    record.snapshots.push(game_snapshot(game, mcts_debug, policy_debug));
    record.result = game.result_code().map(str::to_owned);
    record.total_plies = game.total_plies();
    record.rule60 = game.rule60();
    record.elapsed_ms = now_ms().saturating_sub(record.started_at_ms);
    record.repetition_cycle_plies = game.repetition_cycle_plies();
    if record.result.is_some() {
        record.finished_at_ms = now_ms();
    }
    save(path, job)
}

fn restore_game(initial_fen: &str, snapshots: &[GameSnapshot]) -> Result<Game, String> {
    let mut game = Game::new(if initial_fen.is_empty() { xiangqi::START_FEN } else { initial_fen })?;
    for snapshot in snapshots {
        if same_position(&game.fen(), &snapshot.fen) {
            continue;
        }
        let mut restored = None;
        for movement in game.legal_moves()? {
            let mut candidate = game.clone();
            candidate.apply(movement.0, movement.1)?;
            if same_position(&candidate.fen(), &snapshot.fen) {
                restored = Some(candidate);
                break;
            }
        }
        game = restored.ok_or_else(|| format!("cannot restore benchmark snapshot: {}", snapshot.fen))?;
    }
    Ok(game)
}

fn same_position(left: &str, right: &str) -> bool {
    let left_fields = left.split_whitespace().collect::<Vec<_>>();
    let right_fields = right.split_whitespace().collect::<Vec<_>>();
    left_fields.first() == right_fields.first() && left_fields.get(1) == right_fields.get(1)
}

fn game_snapshot(game: &Game, mcts_debug: Option<Value>, policy_debug: Option<Value>) -> GameSnapshot {
    GameSnapshot { fen: game.fen(), side_to_move: game.side_to_move().to_owned(), turn: game.turn(), rule60: game.rule60(), result: game.result_code().map(str::to_owned), mcts_debug, policy_debug }
}

fn search_debug(result: crate::players::PlayerMove, simulations: usize, searched_fen: String, searched_side: String) -> (Option<Value>, Option<Value>) {
    let selected_move = format_move(result.movement);
    if let Some(result) = result.mcts_debug {
        (Some(json!({
            "searched_fen": searched_fen,
            "searched_side": searched_side,
            "selected_move": selected_move,
            "simulations": simulations,
            "average_leaf_depth": result.average_leaf_depth,
            "max_leaf_depth": result.max_leaf_depth,
            "root_network_value": result.root_network_value,
            "root_children": result.root_children.into_iter().map(|(start, end, visits, q, prior)| json!({
                "move": format_move((start, end)),
                "visits": visits,
                "q": q,
                "prior": prior,
            })).collect::<Vec<_>>(),
        })), None)
    } else if let Some(result) = result.policy_debug {
        (None, Some(json!({
            "searched_fen": searched_fen,
            "searched_side": searched_side,
            "selected_move": selected_move,
            "network_value": result.network_value,
            "candidates": result.candidates.into_iter().map(|(start, end, probability)| json!({
                "move": format_move((start, end)),
                "probability": probability,
            })).collect::<Vec<_>>(),
        })))
    } else {
        (None, None)
    }
}

fn format_move((start, end): (u8, u8)) -> String {
    format!(
        "{}{}-{}{}",
        (b'A' + start % 9) as char,
        start / 9,
        (b'A' + end % 9) as char,
        end / 9,
    )
}

fn save(path: &FilePath, benchmark: &Benchmark) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| error.to_string())?;
    let data = serde_json::to_vec_pretty(benchmark).map_err(|error| error.to_string())?;
    let target = path.join(file_name(benchmark));
    std::fs::write(target, data).map_err(|error| error.to_string())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn file_name(benchmark: &Benchmark) -> String {
    let timestamp = i64::try_from(benchmark.started_at_ms)
        .ok()
        .and_then(|milliseconds| Local.timestamp_millis_opt(milliseconds).single())
        .map(|time| time.format("%Y%m%d-%H%M%S-%3f").to_string())
        .unwrap_or_else(|| "unknown-time".to_owned());
    format!("{timestamp}.json")
}

fn save_tournament(path: &FilePath, tournament: &Tournament) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| error.to_string())?;
    let data = serde_json::to_vec_pretty(tournament).map_err(|error| error.to_string())?;
    let target = path.join(tournament_file_name(tournament));
    std::fs::write(target, data).map_err(|error| error.to_string())
}

fn tournament_file_name(tournament: &Tournament) -> String {
    format!("tournament-{}.json", tournament.id)
}

pub fn migrate_tournaments(path: &FilePath, benchmarks: &HashMap<Uuid, Benchmark>, tournaments: &mut HashMap<Uuid, Tournament>) {
    for tournament in tournaments.values_mut() {
        tournament.ratings = tournament.models.iter().map(|model| (model.clone(), 1500.0)).collect();
        tournament.rated_benchmark_ids.clear();
        for benchmark_id in &tournament.benchmark_ids {
            let Some(benchmark) = benchmarks.get(benchmark_id) else { continue };
            for game in &benchmark.games {
                let Some(result) = game.result.as_deref() else { continue };
                let first_score = if result.starts_with("draw") { 0.5 } else if result.starts_with("red_win") == (game.number % 2 == 1) { 1.0 } else { 0.0 };
                update_elo(&mut tournament.ratings, &benchmark.first_model, &benchmark.second_model, first_score);
                tournament.rated_game_keys.push(format!("{}:{}", benchmark_id, game.number));
            }
            if benchmark.completed_games() == benchmark.games_requested { tournament.rated_benchmark_ids.push(*benchmark_id); }
        }
        tournament.status = if tournament.benchmark_ids.iter().any(|id| benchmarks.get(id).is_some_and(|benchmark| benchmark.failed)) {
            "failed".to_owned()
        } else if tournament.benchmark_ids.iter().all(|id| benchmarks.get(id).is_some_and(|benchmark| benchmark.completed_games() == benchmark.games_requested)) {
            tournament.finished_at_ms = tournament.finished_at_ms.or_else(|| Some(now_ms()));
            "completed".to_owned()
        } else {
            tournament.finished_at_ms = None;
            "paused".to_owned()
        };
        let _ = save_tournament(path, tournament);
    }
}