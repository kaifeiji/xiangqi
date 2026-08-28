use std::io::{BufRead, BufReader, Lines, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use xiangqi::game::Game;

pub enum Player {
    Onnx { model_path: String },
    Pikafish(PikafishPlayer),
}

pub struct PlayerMove {
    pub movement: (u8, u8),
    pub mcts_debug: Option<xiangqi::mcts::MctsSearchResult>,
}

pub struct PikafishPlayer {
    command: std::path::PathBuf,
    move_time_ms: String,
    nnue_path: Option<String>,
    process: Option<Child>,
    stdin: Option<ChildStdin>,
    output: Option<Lines<BufReader<ChildStdout>>>,
}

impl Player {
    pub fn onnx(model_path: String) -> Self {
        Self::Onnx { model_path }
    }

    pub fn pikafish() -> Result<Self, String> {
        Ok(Self::Pikafish(PikafishPlayer::from_environment()?))
    }

    pub fn from_model(model: &str) -> Result<Self, String> {
        if model == "pikafish" {
            Self::pikafish()
        } else {
            Ok(Self::onnx(model.to_owned()))
        }
    }

    pub fn choose_move(&mut self, game: &Game, simulations: usize) -> Result<PlayerMove, String> {
        match self {
            Self::Onnx { model_path } if simulations == 0 => Ok(PlayerMove {
                movement: game.policy_search(model_path)?,
                mcts_debug: None,
            }),
            Self::Onnx { model_path } => {
                let result = game.search(model_path, simulations, 8, 256)?;
                Ok(PlayerMove {
                    movement: result.movement,
                    mcts_debug: Some(result),
                })
            }
            Self::Pikafish(player) => Ok(PlayerMove {
                movement: player.choose_move(game)?,
                mcts_debug: None,
            }),
        }
    }
}

impl PikafishPlayer {
    fn from_environment() -> Result<Self, String> {
        let command =
            pikafish_command().ok_or_else(|| "Pikafish executable not found".to_owned())?;
        Ok(Self {
            command,
            move_time_ms: std::env::var("PIKAFISH_MOVE_TIME_MS")
                .unwrap_or_else(|_| "1000".to_owned()),
            nnue_path: std::env::var("PIKAFISH_NNUE_PATH").ok(),
            process: None,
            stdin: None,
            output: None,
        })
    }

    fn choose_move(&mut self, game: &Game) -> Result<(u8, u8), String> {
        self.start()?;
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "Pikafish stdin is unavailable".to_owned())?;
        writeln!(stdin, "position fen {}", game.fen())
            .map_err(|error| format!("failed to send Pikafish position: {error}"))?;
        writeln!(stdin, "go movetime {}", self.move_time_ms)
            .map_err(|error| format!("failed to send Pikafish search: {error}"))?;
        stdin
            .flush()
            .map_err(|error| format!("failed to flush Pikafish search: {error}"))?;
        let bestmove = self.wait_for("bestmove")?;
        let bestmove = bestmove
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| "Pikafish returned no move".to_owned())?;
        xiangqi::parse_iccs_move(bestmove)
            .ok_or_else(|| format!("invalid Pikafish move: {bestmove}"))
    }

    fn start(&mut self) -> Result<(), String> {
        if let Some(process) = self.process.as_mut() {
            if process.try_wait().ok().flatten().is_none() {
                return Ok(());
            }
        }
        let mut process = Command::new(&self.command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("failed to start Pikafish: {error}"))?;
        let mut stdin = process
            .stdin
            .take()
            .ok_or_else(|| "failed to open Pikafish stdin".to_owned())?;
        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| "failed to open Pikafish stdout".to_owned())?;
        let mut output = BufReader::new(stdout).lines();
        let eval_file = self
            .nnue_path
            .as_ref()
            .map(|path| format!("setoption name EvalFile value {path}\n"))
            .unwrap_or_default();
        write!(stdin, "uci\n{eval_file}")
            .map_err(|error| format!("failed to initialize Pikafish: {error}"))?;
        stdin
            .flush()
            .map_err(|error| format!("failed to flush Pikafish initialization: {error}"))?;
        wait_for_line(&mut output, "uciok")?;
        writeln!(stdin, "isready")
            .map_err(|error| format!("failed to send Pikafish readiness check: {error}"))?;
        stdin
            .flush()
            .map_err(|error| format!("failed to flush Pikafish readiness check: {error}"))?;
        wait_for_line(&mut output, "readyok")?;
        self.process = Some(process);
        self.stdin = Some(stdin);
        self.output = Some(output);
        Ok(())
    }

    fn wait_for(&mut self, expected: &str) -> Result<String, String> {
        let output = self
            .output
            .as_mut()
            .ok_or_else(|| "Pikafish output is unavailable".to_owned())?;
        wait_for_line(output, expected)
    }
}

impl Drop for PikafishPlayer {
    fn drop(&mut self) {
        if let Some(stdin) = self.stdin.as_mut() {
            let _ = writeln!(stdin, "quit");
            let _ = stdin.flush();
        }
        if let Some(process) = self.process.as_mut() {
            let _ = process.wait();
        }
    }
}

pub fn pikafish_command() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("PIKAFISH_PATH").map(std::path::PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let executable = if cfg!(windows) {
        "pikafish.exe"
    } else {
        "pikafish"
    };
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(executable))
            .find(|path| path.is_file())
    })
}

fn wait_for_line<R: BufRead>(lines: &mut Lines<R>, expected: &str) -> Result<String, String> {
    loop {
        let line = lines
            .next()
            .ok_or_else(|| format!("Pikafish returned no {expected}"))?
            .map_err(|error| format!("failed to read Pikafish output: {error}"))?;
        if line == expected || line.starts_with(&format!("{expected} ")) {
            return Ok(line);
        }
    }
}
