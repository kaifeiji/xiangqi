use crate::players::Player;
use crate::session::{ApiError, AppState};
use axum::{extract::{Path, State}, http::StatusCode, Json};
use chrono::{Local, TimeZone};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path as FilePath, PathBuf};
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
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
    pub started_at_ms: u128,
    #[serde(default)]
    pub finished_at_ms: u128,
    pub result: Option<String>,
    pub total_plies: usize,
    pub rule60: u16,
    pub elapsed_ms: u128,
    pub error: Option<String>,
    #[serde(default)]
    pub moves: Vec<String>,
    #[serde(default)]
    pub final_fen: String,
    #[serde(default)]
    pub repetition_cycle_plies: Option<(usize, usize)>,
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
    pub cancelled: bool,
    pub failed: bool,
    pub games: Vec<GameResult>,
}

impl Benchmark {
    fn status(&self) -> &'static str {
        if self.failed { "failed" }
        else if self.cancelled { "cancelled" }
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
    let benchmark = Benchmark { id: Uuid::new_v4(), first_model: request.first_model.clone(), second_model: request.second_model.clone(), mcts_simulations: request.mcts_simulations, games_requested, started_at_ms: now_ms(), finished_at_ms: None, cancelled: false, failed: false, games: Vec::new() };
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
    let control = state.benchmark_controls.read().await.get(&id).cloned().ok_or_else(ApiError::not_found)?;
    control.store(true, Ordering::Relaxed);
    Ok(StatusCode::NO_CONTENT)
}

fn run(request: CreateRequest, first_path: String, second_path: String, id: Uuid, path: PathBuf, benchmarks: Benchmarks, controls: Controls, cancelled: Arc<AtomicBool>) {
    for index in 0..MAINSTREAM_OPENINGS.len() * 2 {
        if cancelled.load(Ordering::Relaxed) { break; }
        let first_red = index % 2 == 0;
        let opening = &MAINSTREAM_OPENINGS[index / 2];
        let opening_move = format!(
            "{}（{}）",
            format_move(opening.movement),
            opening.chinese_notation,
        );
        let (red_path, black_path) = if first_red { (&first_path, &second_path) } else { (&second_path, &first_path) };
        let started_at_ms = now_ms();
        let started = Instant::now();
        let result = play(red_path, black_path, request.mcts_simulations, opening.movement, &cancelled);
        let game = match result {
            Ok(game) => GameResult { number: index + 1, opening_move: opening_move.clone(), started_at_ms, finished_at_ms: now_ms(), result: game.result_code().map(str::to_owned), total_plies: game.total_plies(), rule60: game.rule60(), elapsed_ms: started.elapsed().as_millis(), error: None, moves: game.iccs_moves(), final_fen: game.fen(), repetition_cycle_plies: game.repetition_cycle_plies() },
            Err(error) => GameResult { number: index + 1, opening_move, started_at_ms, finished_at_ms: now_ms(), result: None, total_plies: 0, rule60: 0, elapsed_ms: started.elapsed().as_millis(), error: Some(error), moves: Vec::new(), final_fen: String::new(), repetition_cycle_plies: None },
        };
        let mut jobs = benchmarks.blocking_write();
        let Some(job) = jobs.get_mut(&id) else { return };
        job.failed |= game.error.is_some();
        job.games.push(game);
        let _ = save(&path, job);
        if job.failed { break; }
    }
    if let Some(job) = benchmarks.blocking_write().get_mut(&id) {
        job.cancelled |= cancelled.load(Ordering::Relaxed);
        job.finished_at_ms = Some(now_ms());
        let _ = save(&path, job);
    }
    controls.blocking_write().remove(&id);
}

fn play(red_path: &str, black_path: &str, simulations: usize, opening: (u8, u8), cancelled: &AtomicBool) -> Result<Game, String> {
    let mut game = Game::new(xiangqi::START_FEN)?;
    game.apply(opening.0, opening.1)?;
    let mut players = [Player::from_model(red_path)?, Player::from_model(black_path)?];
    while !game.is_finished() {
        if cancelled.load(Ordering::Relaxed) { break; }
        let index = usize::from(game.side_to_move() == "b");
        let movement = players[index].choose_move(&game, simulations)?.movement;
        game.apply(movement.0, movement.1)?;
    }
    Ok(game)
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