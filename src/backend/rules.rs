use super::position::Position;
use std::collections::HashSet;

#[derive(Clone)]
pub(crate) struct RuleState {
    pub(crate) initial_plies: usize,
    pub(crate) rule60: u16,
    rule60_history: Vec<u16>,
    check_counts: [u8; 2],
    check_counts_history: Vec<[u8; 2]>,
    position_keys: Vec<String>,
    position_history: Vec<Position>,
    checking_sides: Vec<Option<bool>>,
    piece_ids: [u8; 90],
    piece_id_history: Vec<[u8; 90]>,
    chasing_targets: Vec<[HashSet<(bool, u8)>; 2]>,
}

impl RuleState {
    pub(crate) fn new(fen: &str, position: &Position) -> Result<Self, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        let rule60 = fields
            .get(4)
            .ok_or("FEN is missing rule60 counter")?
            .parse::<u16>()
            .map_err(|_| "FEN rule60 counter must be an integer")?;
        if rule60 >= 120 {
            return Err("FEN rule60 counter must be less than 120".into());
        }
        let fullmove = fields
            .get(5)
            .unwrap_or(&"1")
            .parse::<usize>()
            .map_err(|_| "FEN fullmove counter must be an integer")?;
        let initial_plies =
            fullmove.saturating_sub(1) * 2 + usize::from(fields.get(1) == Some(&"b"));
        let mut piece_ids = [u8::MAX; 90];
        let mut next_ids = [0u8; 2];
        for (square, &piece) in position.board.iter().enumerate() {
            if piece != b' ' {
                let side = usize::from(!piece.is_ascii_uppercase());
                piece_ids[square] = next_ids[side];
                next_ids[side] += 1;
            }
        }
        Ok(Self {
            initial_plies,
            rule60,
            rule60_history: vec![rule60],
            check_counts: [0; 2],
            check_counts_history: vec![[0; 2]],
            position_keys: vec![position.repetition_key()],
            position_history: vec![*position],
            checking_sides: Vec::new(),
            piece_ids,
            piece_id_history: vec![piece_ids],
            chasing_targets: Vec::new(),
        })
    }

    pub(crate) fn child(
        &self,
        position: &Position,
        start: usize,
        end: usize,
    ) -> Result<(Position, Self), String> {
        let mut next = self.clone();
        let moving_side = position.red_to_move;
        let child = position.apply(start, end)?;
        let captured = position.board[end] != b' ';
        let was_in_check = position.in_check(moving_side)?;
        let gives_check = child.in_check(child.red_to_move)?;
        let side = usize::from(!moving_side);
        let opponent = 1 - side;
        let mut advances_rule60 = true;
        if gives_check {
            next.check_counts[side] = next.check_counts[side].saturating_add(1);
            advances_rule60 = next.check_counts[side] <= 10;
            if !advances_rule60 && next.check_counts[opponent] > 10 && was_in_check {
                next.check_counts[opponent] = next.check_counts[opponent].saturating_add(1);
                advances_rule60 = true;
            }
        }
        if captured {
            next.rule60 = 0;
            next.check_counts = [0; 2];
        } else if advances_rule60 {
            next.rule60 = next.rule60.saturating_add(1);
        }
        let mut piece_ids = next.piece_ids;
        piece_ids[end] = piece_ids[start];
        piece_ids[start] = u8::MAX;
        next.piece_ids = piece_ids;
        next.piece_id_history.push(piece_ids);
        next.rule60_history.push(next.rule60);
        next.check_counts_history.push(next.check_counts);
        next.checking_sides.push(gives_check.then_some(moving_side));
        next.position_history.push(child);
        next.position_keys.push(child.repetition_key());
        next.chasing_targets.push([
            Self::attack_targets(&child, true, &piece_ids),
            Self::attack_targets(&child, false, &piece_ids),
        ]);
        Ok((child, next))
    }

    pub(crate) fn undo(&mut self) -> Result<(), String> {
        if self.position_history.len() <= 1 {
            return Err("cannot undo initial position".into());
        }
        self.position_history.pop();
        self.position_keys.pop();
        self.rule60_history.pop();
        self.check_counts_history.pop();
        self.piece_id_history.pop();
        self.checking_sides.pop();
        self.chasing_targets.pop();
        self.rule60 = *self
            .rule60_history
            .last()
            .ok_or("rule60 history is empty")?;
        self.check_counts = *self
            .check_counts_history
            .last()
            .ok_or("check count history is empty")?;
        self.piece_ids = *self
            .piece_id_history
            .last()
            .ok_or("piece id history is empty")?;
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
        if !position.board.contains(&b'K') {
            return Some("black_win");
        }
        if !position.board.contains(&b'k') {
            return Some("red_win");
        }
        if position.legal().map_or(true, |moves| moves.is_empty()) {
            return Some(if position.red_to_move {
                "black_win"
            } else {
                "red_win"
            });
        }
        if Self::is_direct_insufficient_material(position) {
            return Some("draw_insufficient_material");
        }
        let current = self.position_keys.last()?;
        if self
            .position_keys
            .iter()
            .filter(|key| *key == current)
            .count()
            >= 3
        {
            if let Some(result) = self.repetition_violation() {
                return Some(result);
            }
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

    pub(crate) fn terminal_value(&self, position: &Position) -> Option<f32> {
        self.result(position).map(|result| match result {
            "red_win" | "black_win_long_check" | "black_win_long_chase" => {
                if position.red_to_move {
                    1.0
                } else {
                    -1.0
                }
            }
            "black_win" | "red_win_long_check" | "red_win_long_chase" => {
                if position.red_to_move {
                    -1.0
                } else {
                    1.0
                }
            }
            _ => 0.0,
        })
    }

    fn attack_targets(position: &Position, red: bool, piece_ids: &[u8; 90]) -> HashSet<(bool, u8)> {
        position
            .chase_targets(red)
            .into_iter()
            .filter_map(|(square, _)| {
                (piece_ids[square as usize] != u8::MAX).then_some((red, piece_ids[square as usize]))
            })
            .collect()
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

    fn repetition_violation(&self) -> Option<&'static str> {
        let current = self.position_keys.last()?;
        let occurrences: Vec<usize> = self
            .position_keys
            .iter()
            .enumerate()
            .filter(|(_, key)| key == &current)
            .map(|(index, _)| index)
            .collect();
        let cycle_start = *occurrences.get(occurrences.len().checked_sub(3)?)?;
        for &side in &[true, false] {
            let moves: Vec<usize> = (cycle_start..self.checking_sides.len())
                .filter(|&index| self.position_history[index].red_to_move == side)
                .collect();
            if moves.len() >= 2
                && moves
                    .iter()
                    .all(|&index| self.checking_sides[index] == Some(side))
            {
                return Some(if side {
                    "black_win_long_check"
                } else {
                    "red_win_long_check"
                });
            }
            if moves.len() >= 2 {
                let common = moves.iter().skip(1).fold(
                    self.chasing_targets[moves[0]][usize::from(side)].clone(),
                    |targets, &index| {
                        targets
                            .intersection(&self.chasing_targets[index][usize::from(side)])
                            .filter(|target| {
                                !self.chasing_targets[index][usize::from(!side)].contains(target)
                            })
                            .copied()
                            .collect()
                    },
                );
                if !common.is_empty() {
                    return Some(if side {
                        "black_win_long_chase"
                    } else {
                        "red_win_long_chase"
                    });
                }
            }
        }
        None
    }
}
