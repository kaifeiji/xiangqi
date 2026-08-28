use crate::players::Player;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;
use xiangqi::game::Game;

#[derive(Deserialize)]
pub struct CreateGameRequest {
    pub mode: Option<String>,
    pub human_side: Option<String>,
    pub fen: Option<String>,
    pub model: Option<String>,
    pub red_model: Option<String>,
    pub black_model: Option<String>,
    pub mcts_simulations: Option<usize>,
}
#[derive(Deserialize)]
pub struct MoveRequest {
    pub start: Option<u8>,
    pub end: Option<u8>,
    pub r#move: Option<String>,
}
#[derive(Deserialize)]
pub struct UndoRequest {
    pub plies: Option<usize>,
}
use serde::Serialize;
#[derive(Serialize)]
pub struct GameResponse {
    pub game_id: Uuid,
    pub mode: String,
    pub human_side: Option<String>,
    pub side_to_move: String,
    pub turn: usize,
    pub rule60: u16,
    pub total_plies: usize,
    pub result: Option<&'static str>,
    pub legal_moves: Vec<String>,
    pub in_check: bool,
    pub is_human_turn: bool,
    pub board: Vec<Option<String>>,
    pub last_error: Option<String>,
    pub mcts_debug: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct AppState {
    pub games: Arc<RwLock<HashMap<Uuid, Session>>>,
}

pub struct Session {
    pub game: Game,
    pub mode: String,
    pub human_side: Option<String>,
    pub players: [Option<Player>; 2],
    pub mcts_simulations: usize,
    pub last_accessed_at: Instant,
    pub last_error: Option<String>,
}

impl Session {
    pub fn step_model(&mut self) -> Result<((u8, u8), Option<Value>), String> {
        if self.game.result_code().is_some() {
            return Err("game already finished".into());
        }
        if self.human_side.as_deref() == Some(self.game.side_to_move()) {
            return Err("current side is controlled by human".into());
        }
        let player_index = usize::from(self.game.side_to_move() == "b");
        let searched_fen = self.game.fen();
        let searched_side = self.game.side_to_move().to_owned();
        let mut debug = None;
        let movement = if let Some(movement) = self.game.opening_book_move() {
            movement
        } else {
            let player = self.players[player_index]
                .as_mut()
                .ok_or_else(|| "model is required".to_owned())?;
            let result = player.choose_move(&self.game, self.mcts_simulations)?;
            if let Some(result) = result.mcts_debug {
                let selected_move = format!(
                    "{}{}-{}{}",
                    (b'A' + result.movement.0 % 9) as char,
                    result.movement.0 / 9,
                    (b'A' + result.movement.1 % 9) as char,
                    result.movement.1 / 9,
                );
                debug = Some(json!({
                    "searched_fen": searched_fen,
                    "searched_side": searched_side,
                    "selected_move": selected_move,
                    "simulations": self.mcts_simulations,
                    "average_leaf_depth": result.average_leaf_depth,
                    "max_leaf_depth": result.max_leaf_depth,
                    "effective_batch_size": std::env::var("MCTS_BATCH_SIZE")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|&value| value > 0)
                        .unwrap_or(8),
                    "effective_max_depth": 256,
                    "root_network_value": result.root_network_value,
                    "root_children": result.root_children.into_iter().map(|(start, end, visits, q, prior)| json!({
                        "move": format!("{}{}-{}{}", (b'A' + start % 9) as char, start / 9, (b'A' + end % 9) as char, end / 9),
                        "visits": visits,
                        "q": q,
                        "prior": prior,
                    })).collect::<Vec<_>>(),
                }));
            }
            result.movement
        };
        self.apply(movement.0, movement.1)?;
        self.last_error = None;
        Ok((movement, debug))
    }

    pub fn apply_human_move(&mut self, movement: (u8, u8)) -> Result<(), String> {
        if self.game.result_code().is_some() {
            return Err("game already finished".into());
        }
        if self.human_side.as_deref() != Some(self.game.side_to_move()) {
            return Err("current side is controlled by model".into());
        }
        self.apply(movement.0, movement.1)?;
        self.last_error = None;
        Ok(())
    }

    pub fn undo(&mut self, plies: usize) -> Result<(), String> {
        if self.mode != "human-model" {
            return Err("undo is only supported in human-model mode".into());
        }
        self.game.undo(plies)?;
        self.last_error = None;
        Ok(())
    }

    fn apply(&mut self, start: u8, end: u8) -> Result<(), String> {
        self.game.apply(start, end)
    }
}

pub fn start_cleanup_task(state: &AppState) {
    let games = Arc::clone(&state.games);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let mut games = games.write().await;
            games.retain(|_, session| {
                session.last_accessed_at.elapsed() < std::time::Duration::from_secs(600)
            });
        }
    });
}

pub async fn step_model(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let session = games.get_mut(&id).ok_or(ApiError::not_found())?;
    session.last_accessed_at = Instant::now();
    let (_, debug) = session.step_model().map_err(|error| {
        session.last_error = Some(error.clone());
        ApiError::bad_request(error)
    })?;
    Ok(Json(snapshot(id, session, debug)?))
}

pub async fn create_game(
    State(state): State<AppState>,
    Json(request): Json<CreateGameRequest>,
) -> Result<Json<GameResponse>, ApiError> {
    let game = Game::new(&request.fen.unwrap_or_else(|| xiangqi::START_FEN.to_owned()))
        .map_err(ApiError::bad_request)?;
    let id = Uuid::new_v4();
    let mode = request.mode.unwrap_or_else(|| "human-model".to_owned());
    if !matches!(mode.as_str(), "human-model" | "model-model") {
        return Err(ApiError::bad_request("invalid mode"));
    }
    let human_side = if mode == "human-model" {
        let side = request.human_side.unwrap_or_else(|| "w".to_owned());
        if !matches!(side.as_str(), "w" | "b") {
            return Err(ApiError::bad_request("invalid human_side"));
        }
        Some(side)
    } else {
        None
    };
    let players = if mode == "human-model" {
        let model = request
            .model
            .as_deref()
            .ok_or(ApiError::bad_request("model is required"))?;
        let model_path =
            crate::models::validate(model).ok_or(ApiError::bad_request("model not found"))?;
        let player = Player::from_model(&model_path).map_err(ApiError::bad_request)?;
        if human_side.as_deref() == Some("w") {
            [None, Some(player)]
        } else {
            [Some(player), None]
        }
    } else {
        let red = request
            .red_model
            .as_deref()
            .ok_or(ApiError::bad_request("red_model is required"))?;
        let black = request
            .black_model
            .as_deref()
            .ok_or(ApiError::bad_request("black_model is required"))?;
        let red_path =
            crate::models::validate(red).ok_or(ApiError::bad_request("model not found"))?;
        let black_path =
            crate::models::validate(black).ok_or(ApiError::bad_request("model not found"))?;
        [
            Some(Player::from_model(&red_path).map_err(ApiError::bad_request)?),
            Some(Player::from_model(&black_path).map_err(ApiError::bad_request)?),
        ]
    };
    let mcts_simulations = request.mcts_simulations.unwrap_or(0);
    if ![0, 1000, 5000, 10000].contains(&mcts_simulations) {
        return Err(ApiError::bad_request("invalid mcts_simulations"));
    }
    let session = Session {
        game,
        mode,
        human_side,
        players,
        mcts_simulations,
        last_accessed_at: Instant::now(),
        last_error: None,
    };
    let response = snapshot(id, &session, None)?;
    state.games.write().await.insert(id, session);
    Ok(Json(response))
}

pub async fn get_game(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let session = games.get_mut(&id).ok_or(ApiError::not_found())?;
    session.last_accessed_at = Instant::now();
    Ok(Json(snapshot(id, session, None)?))
}

pub async fn apply_move(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(request): Json<MoveRequest>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let session = games.get_mut(&id).ok_or(ApiError::not_found())?;
    session.last_accessed_at = Instant::now();
    let movement = request
        .start
        .zip(request.end)
        .or_else(|| request.r#move.as_deref().and_then(xiangqi::parse_iccs_move))
        .ok_or(ApiError::bad_request("move requires start/end or ICCS"))?;
    session
        .apply_human_move(movement)
        .map_err(ApiError::bad_request)?;
    Ok(Json(snapshot(id, session, None)?))
}

pub async fn undo(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(request): Json<UndoRequest>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let session = games.get_mut(&id).ok_or(ApiError::not_found())?;
    session.last_accessed_at = Instant::now();
    session
        .undo(request.plies.unwrap_or(1))
        .map_err(ApiError::bad_request)?;
    Ok(Json(snapshot(id, session, None)?))
}

pub async fn close_game(State(state): State<AppState>, Path(id): Path<Uuid>) -> StatusCode {
    state.games.write().await.remove(&id);
    StatusCode::NO_CONTENT
}

fn snapshot(
    id: Uuid,
    session: &Session,
    mcts_debug: Option<serde_json::Value>,
) -> Result<GameResponse, ApiError> {
    let side = session.game.side_to_move();
    let board = session
        .game
        .board()
        .into_iter()
        .map(|piece| (piece != b' ').then(|| (piece as char).to_string()))
        .collect();
    let legal_moves = session
        .game
        .legal_moves()
        .map_err(ApiError::bad_request)?
        .into_iter()
        .map(|(start, end)| {
            format!(
                "{}{}-{}{}",
                (b'A' + start % 9) as char,
                start / 9,
                (b'A' + end % 9) as char,
                end / 9
            )
        })
        .collect();
    Ok(GameResponse {
        game_id: id,
        mode: session.mode.clone(),
        human_side: session.human_side.clone(),
        side_to_move: side.to_owned(),
        turn: session.game.turn(),
        rule60: session.game.rule60(),
        total_plies: session.game.total_plies(),
        result: session.game.result_code(),
        legal_moves,
        in_check: session.game.in_check().map_err(ApiError::bad_request)?,
        is_human_turn: session.human_side.as_deref() == Some(side)
            && session.game.result_code().is_none(),
        board,
        last_error: session.last_error.clone(),
        mcts_debug,
    })
}

pub struct ApiError {
    status: StatusCode,
    message: String,
}
impl ApiError {
    fn bad_request(message: impl ToString) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.to_string(),
        }
    }
    fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: "game not found".to_owned(),
        }
    }
}
impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}
