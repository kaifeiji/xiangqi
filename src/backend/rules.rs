use super::position::Position;

#[derive(Clone)]
pub(crate) struct RuleState {
    pub(crate) initial_plies: usize,
    pub(crate) rule60: u16,
    rule60_history: Vec<u16>,
    position_keys: Vec<[u8; 91]>,
    position_history: Vec<Position>,
}

impl RuleState {
    pub(crate) fn new(fen: &str, position: &Position) -> Result<Self, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        let rule60 = fields
            .get(4)
            .ok_or("FEN is missing rule60 counter")?
            .parse::<u16>()
            .map_err(|_| "FEN rule60 counter must be an integer")?;
        let fullmove = fields
            .get(5)
            .unwrap_or(&"1")
            .parse::<usize>()
            .map_err(|_| "FEN fullmove counter must be an integer")?;
        let initial_plies =
            fullmove.saturating_sub(1) * 2 + usize::from(fields.get(1) == Some(&"b"));
        Ok(Self {
            initial_plies,
            rule60,
            rule60_history: vec![rule60],
            position_keys: vec![position.repetition_key()],
            position_history: vec![*position],
        })
    }

    pub(crate) fn child(
        &self,
        position: &Position,
        start: usize,
        end: usize,
    ) -> Result<(Position, Self), String> {
        let mut next = self.clone();
        let child = position.apply(start, end)?;
        let captured = position.board[end] != b' ';
        if captured {
            next.rule60 = 0;
        } else {
            next.rule60 = next.rule60.saturating_add(1);
        }
        next.rule60_history.push(next.rule60);
        next.position_history.push(child);
        next.position_keys.push(child.repetition_key());
        Ok((child, next))
    }

    pub(crate) fn undo(&mut self) -> Result<(), String> {
        if self.position_history.len() <= 1 {
            return Err("cannot undo initial position".into());
        }
        self.position_history.pop();
        self.position_keys.pop();
        self.rule60_history.pop();
        self.rule60 = *self
            .rule60_history
            .last()
            .ok_or("rule60 history is empty")?;
        Ok(())
    }

    pub(crate) fn total_plies(&self) -> usize {
        self.position_history.len() - 1
    }

    pub(crate) fn current_position(&self) -> Position {
        *self
            .position_history
            .last()
            .expect("rule state has an initial position")
    }

    pub(crate) fn result(&self, position: &Position) -> Option<&'static str> {
        let legal_moves = position.legal().unwrap_or_default();
        self.result_with_legal_moves(position, !legal_moves.is_empty())
    }

    fn result_with_legal_moves(
        &self,
        position: &Position,
        has_legal_moves: bool,
    ) -> Option<&'static str> {
        if !position.board.contains(&b'K') {
            return Some("black_win");
        }
        if !position.board.contains(&b'k') {
            return Some("red_win");
        }
        if !has_legal_moves {
            return Some(if position.red_to_move {
                "black_win"
            } else {
                "red_win"
            });
        }
        if Self::is_direct_insufficient_material(position) {
            return Some("draw_insufficient_material");
        }
        if self.has_threefold_repetition() {
            return Some("draw_repetition");
        }
        if self.rule60 >= 120 {
            return Some("draw_natural_limit");
        }
        if self.initial_plies + self.total_plies() >= 600 {
            return Some("draw_move_limit");
        }
        None
    }

    pub(crate) fn move_repeats_position(
        &self,
        position: &Position,
        start: usize,
        end: usize,
    ) -> Result<bool, String> {
        let child = position.apply(start, end)?;
        let child_key = child.repetition_key();
        Ok(self.position_keys.iter().any(|key| *key == child_key))
    }

    pub(crate) fn position_repeats(&self, position: &Position) -> bool {
        let key = position.repetition_key();
        self.position_keys.iter().any(|seen| *seen == key)
    }

    pub(crate) fn repetition_cycle_plies(&self) -> Option<(usize, usize)> {
        if !self.has_threefold_repetition() {
            return None;
        }
        let current = self.position_keys.last()?;
        let cycle_start = self
            .position_keys
            .iter()
            .enumerate()
            .filter(|(_, key)| *key == current)
            .nth(self.position_keys.iter().filter(|key| *key == current).count() - 3)?
            .0;
        Some((cycle_start + 1, self.total_plies()))
    }

    fn has_threefold_repetition(&self) -> bool {
        let Some(current) = self.position_keys.last() else {
            return false;
        };
        self.position_keys.iter().filter(|key| *key == current).count() >= 3
    }

    pub(crate) fn terminal_value_with_legal_moves(
        &self,
        position: &Position,
        has_legal_moves: bool,
    ) -> Option<f32> {
        self.result_with_legal_moves(position, has_legal_moves)
            .map(|result| Self::terminal_value_for_result(result, position.red_to_move))
    }

    fn terminal_value_for_result(result: &str, side_to_move_is_red: bool) -> f32 {
        match result {
            "red_win" => {
                if side_to_move_is_red { 1.0 } else { -1.0 }
            }
            "black_win" => {
                if side_to_move_is_red { -1.0 } else { 1.0 }
            }
            _ => 0.0,
        }
    }

    fn is_direct_insufficient_material(position: &Position) -> bool {
        if position.by_type[6] != 0 {
            return false;
        }
        let rooks = position.by_type[4].count_ones();
        let knights = position.by_type[3].count_ones();
        let cannons = position.by_type[5].count_ones();
        let advisors = position.by_type[1].count_ones();
        let bishops = position.by_type[2].count_ones();
        if rooks + knights + cannons == 0 {
            return true;
        }
        if rooks == 0 && knights == 0 && cannons == 1 {
            let cannon_side = if position.by_color[0] & position.by_type[5] != 0 {
                0
            } else {
                1
            };
            let other_side = 1 - cannon_side;
            let own_advisors = (position.by_color[cannon_side] & position.by_type[1]).count_ones();
            let other_advisors = (position.by_color[other_side] & position.by_type[1]).count_ones();
            let own_bishops = (position.by_color[cannon_side] & position.by_type[2]).count_ones();
            return own_advisors == 0
                && (other_advisors == 0 || (other_advisors == 1 && own_bishops == 0));
        }
        rooks == 0
            && knights == 0
            && cannons == 2
            && advisors == 0
            && bishops == 0
            && (position.by_color[0] & position.by_type[5]).count_ones() == 1
            && (position.by_color[1] & position.by_type[5]).count_ones() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::RuleState;

    #[test]
    fn terminal_value_treats_repetition_as_draw() {
        assert_eq!(RuleState::terminal_value_for_result("draw_repetition", true), 0.0);
        assert_eq!(RuleState::terminal_value_for_result("draw_repetition", false), 0.0);
    }
}
