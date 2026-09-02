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
    pub games: Vec<GameResult>,
}

impl Benchmark {
    fn status(&self) -> &'static str {
        if self.failed { "failed" }
        else if self.paused || self.cancelled { "paused" }
        else if self.games.len() == self.games_requested { "completed" }
        else { "running" }
    }

    fn response(&self) -> Response {
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
        Response { id: self.id, first_model: self.first_model.clone(), second_model: self.second_model.clone(), mcts_simulations: self.mcts_simulations, games_requested: self.games_requested, started_at_ms: self.started_at_ms, finished_at_ms: self.finished_at_ms, status: self.status(), first_wins, second_wins, draws, games: self.games.clone() }
    }
}

#[derive(Serialize)]
pub struct Response {
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
    pub games: Vec<GameResult>,
}

pub type Benchmarks = Arc<RwLock<HashMap<Uuid, Benchmark>>>;
pub type Controls = Arc<RwLock<HashMap<Uuid, Arc<AtomicBool>>>>;

pub fn load(path: &FilePath) -> HashMap<Uuid, Benchmark> {
    let mut benchmarks = HashMap::new();
    let Ok(entries) = std::fs::read_dir(path) else { return benchmarks };
    for entry in entries.flatten() {
        let source = entry.path();
        let Ok(text) = std::fs::read_to_string(&source) else { continue };
        if let Ok(benchmark) = serde_json::from_str::<Benchmark>(&text) {
            let target = path.join(file_name(&benchmark));
            if source != target && !target.exists() {
                let _ = std::fs::rename(source, target);
            }
            benchmarks.insert(benchmark.id, benchmark);
        }
    }
    benchmarks
}

pub async fn create(State(state): State<AppState>, Json(request): Json<CreateRequest>) -> Result<Json<Response>, ApiError> {
    if request.first_model == request.second_model { return Err(ApiError::bad_request("benchmark models must differ")); }
    if ![0, 1000, 5000, 10000].contains(&request.mcts_simulations) { return Err(ApiError::bad_request("invalid mcts_simulations")); }
    let first_path = crate::models::validate(&request.first_model).ok_or_else(|| ApiError::bad_request("first model not found"))?;
    let second_path = crate::models::validate(&request.second_model).ok_or_else(|| ApiError::bad_request("second model not found"))?;
    let games_requested = MAINSTREAM_OPENINGS.len() * 2;
    let benchmark = Benchmark { id: Uuid::new_v4(), first_model: request.first_model.clone(), second_model: request.second_model.clone(), mcts_simulations: request.mcts_simulations, games_requested, started_at_ms: now_ms(), finished_at_ms: None, paused: false, cancelled: false, failed: false, games: Vec::new() };
    save(&state.benchmark_path, &benchmark).map_err(ApiError::bad_request)?;
    let response = benchmark.response();
    let id = benchmark.id;
    state.benchmarks.write().await.insert(id, benchmark);
    let cancelled = Arc::new(AtomicBool::new(false));
    state.benchmark_controls.write().await.insert(id, Arc::clone(&cancelled));
    let benchmarks = Arc::clone(&state.benchmarks);
    let controls = Arc::clone(&state.benchmark_controls);
    let path = state.benchmark_path.clone();
    tokio::task::spawn_blocking(move || run(request, first_path, second_path, id, path, benchmarks, controls, cancelled));
    Ok(Json(response))
}

pub async fn list(State(state): State<AppState>) -> Json<Vec<Response>> {
    let mut benchmarks: Vec<_> = state.benchmarks.read().await.values().cloned().collect();
    benchmarks.sort_by_key(|benchmark| std::cmp::Reverse(benchmark.started_at_ms));
    Json(benchmarks.iter().map(Benchmark::response).collect())
}

pub async fn get(State(state): State<AppState>, Path(id): Path<Uuid>) -> Result<Json<Response>, ApiError> {
    state.benchmarks.read().await.get(&id).map(|job| Json(job.response())).ok_or_else(ApiError::not_found)
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
    if let Some(control) = state.benchmark_controls.read().await.get(&id).cloned() {
        control.store(false, Ordering::Relaxed);
        let mut jobs = state.benchmarks.write().await;
        let job = jobs.get_mut(&id).ok_or_else(ApiError::not_found)?;
        job.paused = false;
        job.cancelled = false;
        save(&state.benchmark_path, job).map_err(ApiError::bad_request)?;
        return Ok(StatusCode::NO_CONTENT);
    }

    let mut jobs = state.benchmarks.write().await;
    let job = jobs.get_mut(&id).ok_or_else(ApiError::not_found)?;
    if job.failed || job.games.len() == job.games_requested {
        return Err(ApiError::bad_request("benchmark cannot be resumed"));
    }
    let request = CreateRequest { first_model: job.first_model.clone(), second_model: job.second_model.clone(), mcts_simulations: job.mcts_simulations };
    let first_path = crate::models::validate(&request.first_model).ok_or_else(|| ApiError::bad_request("first model not found"))?;
    let second_path = crate::models::validate(&request.second_model).ok_or_else(|| ApiError::bad_request("second model not found"))?;
    job.paused = false;
    job.cancelled = false;
    save(&state.benchmark_path, job).map_err(ApiError::bad_request)?;
    drop(jobs);

    let paused = Arc::new(AtomicBool::new(false));
    state.benchmark_controls.write().await.insert(id, Arc::clone(&paused));
    let benchmarks = Arc::clone(&state.benchmarks);
    let controls = Arc::clone(&state.benchmark_controls);
    let path = state.benchmark_path.clone();
    tokio::task::spawn_blocking(move || run(request, first_path, second_path, id, path, benchmarks, controls, paused));
    Ok(StatusCode::NO_CONTENT)
}

fn run(request: CreateRequest, first_path: String, second_path: String, id: Uuid, path: PathBuf, benchmarks: Benchmarks, controls: Controls, paused: Arc<AtomicBool>) {
    loop {
        if paused.load(Ordering::Relaxed) { break; }
        let index = {
            let jobs = benchmarks.blocking_read();
            let Some(job) = jobs.get(&id) else { return };
            if job.failed || job.games.len() == job.games_requested { break; }
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
        } else if job.games.len() == job.games_requested || job.failed {
            job.finished_at_ms = Some(now_ms());
            job.paused = false;
        }
        let _ = save(&path, job);
    }
    controls.blocking_write().remove(&id);
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
    let temporary = target.with_extension("json.tmp");
    std::fs::write(&temporary, data).map_err(|error| error.to_string())?;
    std::fs::rename(temporary, target).map_err(|error| error.to_string())
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