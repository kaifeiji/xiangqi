use crate::position::Position;
use crate::rules::RuleState;
use uuid::Uuid;

const MAINSTREAM_OPENING_MOVES: &[((u8, u8), u32)] = &[
    ((19, 22), 24), // B2-E2, central cannon
    ((25, 22), 24), // H2-E2, central cannon
    ((19, 23), 6),  // B2-F2
    ((25, 21), 6),  // H2-D2
    ((19, 21), 5),  // B2-D2
    ((25, 23), 5),  // H2-F2
    ((1, 20), 8),   // B0-C2, horse
    ((7, 24), 8),   // H0-G2, horse
    ((2, 22), 4),   // C0-E2, elephant
    ((6, 22), 4),   // G0-E2, elephant
    ((31, 40), 6),  // E3-E4, central pawn
    ((30, 39), 2),  // C3-C4
    ((34, 43), 2),  // G3-G4
];

pub struct Game {
    position: Position,
    rules: RuleState,
    moves: Vec<(u8, u8)>,
}

impl Game {
    pub fn new(fen: &str) -> Result<Self, String> {
        let position = Position::parse(fen)?;
        let rules = RuleState::new(fen, &position)?;
        Ok(Self {
            position,
            rules,
            moves: Vec::new(),
        })
    }

    pub fn fen(&self) -> String {
        let mut fields = self
            .position
            .fen()
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        fields[4] = self.rules.rule60.to_string();
        fields[5] = ((self.rules.initial_plies + self.moves.len()) / 2 + 1).to_string();
        fields.join(" ")
    }

    pub fn side_to_move(&self) -> &'static str {
        if self.position.red_to_move {
            "w"
        } else {
            "b"
        }
    }

    pub fn board(&self) -> [u8; 90] {
        self.position.board
    }

    pub fn total_plies(&self) -> usize {
        self.moves.len()
    }

    pub fn turn(&self) -> usize {
        self.moves.len() + 1
    }

    pub fn rule60(&self) -> u16 {
        self.rules.rule60
    }

    pub fn result_code(&self) -> Option<&'static str> {
        self.result()
    }

    pub fn moves(&self) -> &[(u8, u8)] {
        &self.moves
    }

    pub fn is_finished(&self) -> bool {
        self.result().is_some()
    }

    pub fn search(
        &self,
        model_path: &str,
        simulations: usize,
        _batch_size: usize,
        max_depth: usize,
    ) -> Result<crate::mcts::MctsSearchResult, String> {
        let batch_size = std::env::var("MCTS_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(8);
        crate::mcts::search_onnx(
            &self.fen(),
            model_path,
            simulations,
            1.25,
            batch_size,
            max_depth,
        )
    }

    pub fn opening_book_move(&self) -> Option<(u8, u8)> {
        if !self.moves.is_empty() || !self.position.red_to_move || self.fen() != crate::START_FEN {
            return None;
        }
        let legal = self.position.legal().ok()?;
        let legal_openings: Vec<_> = MAINSTREAM_OPENING_MOVES
            .iter()
            .copied()
            .filter(|&((start, end), _)| legal.contains(&(start as usize, end as usize)))
            .collect();
        if legal_openings.is_empty() {
            return None;
        }
        let total_weight: u32 = legal_openings.iter().map(|&(_, weight)| weight).sum();
        let mut choice = u32::from_le_bytes(
            Uuid::new_v4().as_bytes()[..4]
                .try_into()
                .expect("UUID has 16 bytes"),
        ) % total_weight;
        for &((start, end), weight) in &legal_openings {
            if choice < weight {
                return Some((start, end));
            }
            choice -= weight;
        }
        legal_openings.last().map(|&((start, end), _)| (start, end))
    }

    pub fn policy_search(
        &self,
        model_path: &str,
    ) -> Result<crate::mcts::PolicySearchResult, String> {
        if self.result().is_some() {
            return Err("game already finished".into());
        }
        let result = crate::mcts::policy_search_onnx(&self.fen(), model_path)?;
        if !self
            .position
            .legal()?
            .contains(&(result.movement.0 as usize, result.movement.1 as usize))
        {
            return Err("policy model returned an illegal move".into());
        }
        Ok(result)
    }

    pub fn in_check(&self) -> Result<bool, String> {
        self.position.in_check(self.position.red_to_move)
    }

    pub fn legal_moves(&self) -> Result<Vec<(u8, u8)>, String> {
        self.position.legal().map(|moves| {
            moves
                .into_iter()
                .map(|(start, end)| (start as u8, end as u8))
                .collect()
        })
    }

    pub fn apply(&mut self, start: u8, end: u8) -> Result<(), String> {
        let legal = self.position.legal()?;
        if !legal.contains(&(start as usize, end as usize)) {
            return Err("illegal move".into());
        }
        let (next, rules) = self
            .rules
            .child(&self.position, start as usize, end as usize)?;
        self.position = next;
        self.rules = rules;
        self.moves.push((start, end));
        Ok(())
    }

    pub fn undo(&mut self, plies: usize) -> Result<(), String> {
        if plies == 0 || plies > self.moves.len() {
            return Err("invalid undo plies".into());
        }
        for _ in 0..plies {
            self.rules.undo()?;
            self.moves.pop();
        }
        self.position = self.rules.current_position();
        Ok(())
    }
}

impl Game {
    fn result(&self) -> Option<&'static str> {
        self.rules.result(&self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::START_FEN;

    #[test]
    fn game_applies_and_undoes_a_legal_move() {
        let mut game = Game::new(START_FEN).unwrap();
        game.apply(19, 28).unwrap();
        assert_eq!(game.moves.len(), 1);
        game.undo(1).unwrap();
        assert_eq!(game.moves.len(), 0);
        assert_eq!(game.position.fen(), START_FEN);
    }

    #[test]
    fn game_reports_theoretical_draw_for_king_and_advisor() {
        let game = Game::new("4k4/9/9/9/9/9/9/9/3A5/4K4 w - - 0 1").unwrap();

        assert_eq!(game.result_code(), Some("draw_insufficient_material"));
    }

    #[test]
    fn game_reports_direct_draw_for_a_lone_cannon() {
        let game = Game::new("4k4/9/9/9/9/9/9/9/4C4/4K4 w - - 0 1").unwrap();

        assert_eq!(game.result_code(), Some("draw_insufficient_material"));
    }

    #[test]
    fn game_does_not_report_theoretical_draw_with_a_rook() {
        let game = Game::new("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1").unwrap();

        assert_ne!(game.result_code(), Some("draw_insufficient_material"));
    }

    #[test]
    fn game_reports_a_non_checking_threefold_as_a_draw() {
        let mut game = Game::new("4k4/3r5/9/9/4p4/9/9/9/3R5/4K4 w - - 0 1").unwrap();
        for _ in 0..2 {
            game.apply(12, 11).unwrap();
            game.apply(75, 74).unwrap();
            game.apply(11, 12).unwrap();
            game.apply(74, 75).unwrap();
        }

        assert_eq!(game.result_code(), Some("draw_repetition"));
    }

    #[test]
    fn opening_book_only_applies_to_the_red_first_move() {
        let mut game = Game::new(START_FEN).unwrap();
        assert!(game.opening_book_move().is_some_and(|movement| {
            MAINSTREAM_OPENING_MOVES
                .iter()
                .any(|&((start, end), _)| (start, end) == movement)
        }));

        game.apply(19, 22).unwrap();
        assert_eq!(game.opening_book_move(), None);
    }

    #[test]
    fn opening_book_does_not_apply_to_black_or_custom_positions() {
        let mut game = Game::new(START_FEN).unwrap();
        game.apply(19, 22).unwrap();
        assert_eq!(game.side_to_move(), "b");
        assert_eq!(game.opening_book_move(), None);

        let game =
            Game::new("rnbakabnr/9/1c5c1/p1p1p1p1p/9/8P/P1P1P1P2/1C5C1/9/RNBAKABNR w - - 0 1")
                .unwrap();
        assert_eq!(game.opening_book_move(), None);
    }
}
