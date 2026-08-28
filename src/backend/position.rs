const ROWS: i32 = 10;
const COLS: i32 = 9;
const BOARD_SIZE: usize = 90;
const TYPE_COUNT: usize = 8;

use std::collections::HashSet;
#[derive(Clone, Copy)]
pub struct Position {
    pub board: [u8; BOARD_SIZE],
    pub by_type: [u128; TYPE_COUNT],
    pub by_color: [u128; 2],
    pub red_to_move: bool,
}

fn index(row: i32, col: i32) -> Option<usize> {
    if (0..ROWS).contains(&row) && (0..COLS).contains(&col) {
        Some((row * COLS + col) as usize)
    } else {
        None
    }
}
fn row(square: usize) -> i32 {
    square as i32 / COLS
}
fn col(square: usize) -> i32 {
    square as i32 % COLS
}
fn is_red(piece: u8) -> bool {
    piece.is_ascii_uppercase()
}
fn own(piece: u8, red: bool) -> bool {
    is_red(piece) == red
}
fn type_index(piece: u8) -> Option<usize> {
    match piece.to_ascii_uppercase() {
        b'K' => Some(0),
        b'A' => Some(1),
        b'B' => Some(2),
        b'N' => Some(3),
        b'R' => Some(4),
        b'C' => Some(5),
        b'P' => Some(6),
        _ => None,
    }
}
fn bit(square: usize) -> u128 {
    1u128 << square
}
fn candidate_mask(position: &Position, kind: usize, red: bool) -> u128 {
    position.by_type[kind] & position.by_color[usize::from(red)]
}
fn in_palace(r: i32, c: i32, red: bool) -> bool {
    (3..=5).contains(&c)
        && if red {
            (0..=2).contains(&r)
        } else {
            (7..=9).contains(&r)
        }
}
fn add_target(position: &Position, out: &mut Vec<usize>, r: i32, c: i32, red: bool) {
    if let Some(square) = index(r, c) {
        if position.board[square] == b' ' || !own(position.board[square], red) {
            out.push(square);
        }
    }
}
fn valid_piece_region(piece: u8, square: usize) -> bool {
    let red = is_red(piece);
    match piece.to_ascii_uppercase() {
        b'K' | b'A' => in_palace(row(square), col(square), red),
        b'B' => (red && row(square) <= 4) || (!red && row(square) >= 5),
        _ => true,
    }
}

impl Position {
    pub fn parse(fen: &str) -> Result<Self, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 || !matches!(fields[1], "w" | "b") {
            return Err("FEN must contain six fields and side must be w or b".into());
        }
        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 10 {
            return Err("FEN board must contain ten ranks".into());
        }
        let mut position = Self {
            board: [b' '; BOARD_SIZE],
            by_type: [0; TYPE_COUNT],
            by_color: [0; 2],
            red_to_move: fields[1] == "w",
        };
        for (fen_row, text) in ranks.iter().enumerate() {
            let board_row = 9 - fen_row as i32;
            let mut board_col = 0;
            for token in text.bytes() {
                if token.is_ascii_digit() {
                    board_col += (token - b'0') as i32;
                } else {
                    if board_col >= COLS {
                        return Err("FEN row width must be 9".into());
                    }
                    if type_index(token).is_none() {
                        return Err(format!("invalid piece: {}", token as char));
                    }
                    let square = (board_row * COLS + board_col) as usize;
                    if !valid_piece_region(token, square) {
                        return Err("piece is outside its legal region".into());
                    }
                    position.put(square, token);
                    board_col += 1;
                }
            }
            if board_col != COLS {
                return Err("FEN row width must be 9".into());
            }
        }
        if !position.board.contains(&b'K') || !position.board.contains(&b'k') {
            return Err("FEN must contain one red king and one black king".into());
        }
        let limits = [1, 2, 2, 2, 2, 2, 5];
        for (kind, &limit) in limits.iter().enumerate() {
            for color in 0..2 {
                if (position.by_type[kind] & position.by_color[color]).count_ones() > limit {
                    return Err("FEN contains too many pieces of one type".into());
                }
            }
        }
        Ok(position)
    }

    pub fn fen(&self) -> String {
        let mut ranks = Vec::with_capacity(10);
        for board_row in (0..ROWS).rev() {
            let mut text = String::new();
            let mut empty: u8 = 0;
            for board_col in 0..COLS {
                let piece = self.board[(board_row * COLS + board_col) as usize];
                if piece == b' ' {
                    empty += 1;
                } else {
                    if empty > 0 {
                        text.push(char::from(b'0' + empty));
                        empty = 0;
                    }
                    text.push(piece as char);
                }
            }
            if empty > 0 {
                text.push(char::from(b'0' + empty));
            }
            ranks.push(text);
        }
        format!(
            "{} {} - - {} 1",
            ranks.join("/"),
            if self.red_to_move { "w" } else { "b" },
            0
        )
    }

    pub(crate) fn repetition_key(&self) -> String {
        let fen = self.fen();
        let mut fields = fen.split_whitespace();
        format!("{} {}", fields.next().unwrap(), fields.next().unwrap())
    }

    pub(crate) fn pseudo(&self, start: usize) -> Vec<usize> {
        let piece = self.board[start];
        if piece == b' ' || !own(piece, self.red_to_move) {
            return Vec::new();
        }
        let (r, c, red, kind) = (
            row(start),
            col(start),
            self.red_to_move,
            piece.to_ascii_uppercase(),
        );
        let mut moves = Vec::new();
        match kind {
            b'K' => {
                for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if in_palace(r + dr, c + dc, red) {
                        add_target(self, &mut moves, r + dr, c + dc, red);
                    }
                }
                let step = if red { 1 } else { -1 };
                let mut nr = r + step;
                while let Some(dst) = index(nr, c) {
                    if self.board[dst] != b' ' {
                        if self.board[dst].to_ascii_uppercase() == b'K'
                            && !own(self.board[dst], red)
                        {
                            moves.push(dst);
                        }
                        break;
                    }
                    nr += step;
                }
            }
            b'A' => {
                for (dr, dc) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    if in_palace(r + dr, c + dc, red) {
                        add_target(self, &mut moves, r + dr, c + dc, red);
                    }
                }
            }
            b'B' => {
                for (dr, dc) in [(2, 2), (2, -2), (-2, 2), (-2, -2)] {
                    let (nr, nc) = (r + dr, c + dc);
                    if let Some(eye) = index(r + dr / 2, c + dc / 2) {
                        if index(nr, nc).is_some()
                            && self.board[eye] == b' '
                            && ((red && nr <= 4) || (!red && nr >= 5))
                        {
                            add_target(self, &mut moves, nr, nc, red);
                        }
                    }
                }
            }
            b'N' => {
                for (dr, dc, lr, lc) in [
                    (2, 1, 1, 0),
                    (2, -1, 1, 0),
                    (-2, 1, -1, 0),
                    (-2, -1, -1, 0),
                    (1, 2, 0, 1),
                    (-1, 2, 0, 1),
                    (1, -2, 0, -1),
                    (-1, -2, 0, -1),
                ] {
                    if index(r + lr, c + lc).map_or(true, |leg| self.board[leg] == b' ') {
                        add_target(self, &mut moves, r + dr, c + dc, red);
                    }
                }
            }
            b'R' | b'C' => {
                for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (mut nr, mut nc, mut jumped) = (r + dr, c + dc, false);
                    while let Some(dst) = index(nr, nc) {
                        let target = self.board[dst];
                        if kind == b'R' {
                            if target == b' ' {
                                moves.push(dst);
                            } else {
                                if !own(target, red) {
                                    moves.push(dst);
                                }
                                break;
                            }
                        } else if !jumped {
                            if target == b' ' {
                                moves.push(dst);
                            } else {
                                jumped = true;
                            }
                        } else if target != b' ' {
                            if !own(target, red) {
                                moves.push(dst);
                            }
                            break;
                        }
                        nr += dr;
                        nc += dc;
                    }
                }
            }
            b'P' => {
                let forward = if red { 1 } else { -1 };
                add_target(self, &mut moves, r + forward, c, red);
                if (red && r >= 5) || (!red && r <= 4) {
                    add_target(self, &mut moves, r, c - 1, red);
                    add_target(self, &mut moves, r, c + 1, red);
                }
            }
            _ => {}
        };
        moves
    }

    pub(crate) fn chase_targets(&self, red: bool) -> HashSet<(u8, u8)> {
        let mut targets = HashSet::new();
        let mut attack_position = *self;
        attack_position.red_to_move = red;
        let mut attackers = attack_position.by_color[usize::from(!red)];
        while attackers != 0 {
            let start = attackers.trailing_zeros() as usize;
            attackers &= attackers - 1;
            let piece = attack_position.board[start].to_ascii_uppercase();
            if matches!(piece, b'K' | b'P') {
                continue;
            }
            for end in attack_position.pseudo(start) {
                let target = attack_position.board[end];
                if target == b' '
                    || target.is_ascii_uppercase() == red
                    || target.to_ascii_uppercase() == b'K'
                {
                    continue;
                }
                let Ok(after_attack) = attack_position.apply(start, end) else {
                    continue;
                };
                if after_attack.in_check(red).unwrap_or(true) {
                    continue;
                }
                let can_recapture = after_attack
                    .legal()
                    .map(|moves| moves.iter().any(|&(_, destination)| destination == end))
                    .unwrap_or(true);
                let attacker_kind = piece;
                let target_kind = target.to_ascii_uppercase();
                let strong_chase = matches!(
                    (attacker_kind, target_kind),
                    (b'N' | b'C', b'R') | (b'A' | b'B', b'R' | b'C' | b'N')
                );
                let mutual_attack =
                    attacker_kind == target_kind && Self::can_attack_square(self, !red, end, start);
                if strong_chase || (!can_recapture && !mutual_attack) {
                    targets.insert((end as u8, target.to_ascii_uppercase()));
                }
            }
        }
        targets
    }

    fn can_attack_square(&self, red: bool, start: usize, end: usize) -> bool {
        if start >= BOARD_SIZE || end >= BOARD_SIZE {
            return false;
        }
        let mut position = *self;
        position.red_to_move = red;
        position.pseudo(start).contains(&end)
            && position
                .apply(start, end)
                .map(|next| !next.in_check(red).unwrap_or(true))
                .unwrap_or(false)
    }

    fn put(&mut self, square: usize, piece: u8) {
        self.board[square] = piece;
        if let Some(kind) = type_index(piece) {
            self.by_type[kind] |= bit(square);
            self.by_color[usize::from(!is_red(piece))] |= bit(square);
        }
    }
    fn remove(&mut self, square: usize) -> u8 {
        let piece = self.board[square];
        self.board[square] = b' ';
        if let Some(kind) = type_index(piece) {
            self.by_type[kind] &= !bit(square);
            self.by_color[usize::from(!is_red(piece))] &= !bit(square);
        }
        piece
    }
    pub fn apply(&self, start: usize, end: usize) -> Result<Self, String> {
        if start >= BOARD_SIZE || end >= BOARD_SIZE {
            return Err("square index must be in 0..90".into());
        }
        let piece = self.board[start];
        if piece == b' ' {
            return Err("empty start square".into());
        }
        if !own(piece, self.red_to_move) {
            return Err("cannot move enemy piece".into());
        }
        let mut next = *self;
        next.remove(start);
        next.remove(end);
        next.put(end, piece);
        next.red_to_move = !self.red_to_move;
        Ok(next)
    }
    pub(crate) fn in_check(&self, red: bool) -> Result<bool, String> {
        let king = self
            .board
            .iter()
            .position(|&piece| piece == if red { b'K' } else { b'k' })
            .ok_or("missing king")?;
        let target_row = row(king);
        let target_col = col(king);
        let attackers = |kind: usize| candidate_mask(self, kind, red);

        let mut knights = attackers(3);
        while knights != 0 {
            let start = knights.trailing_zeros() as usize;
            knights &= knights - 1;
            let (source_row, source_col) = (row(start), col(start));
            let dr = target_row - source_row;
            let dc = target_col - source_col;
            let leg = if dr.abs() == 2 {
                index(source_row + dr / 2, source_col)
            } else {
                index(source_row, source_col + dc / 2)
            };
            if matches!((dr.abs(), dc.abs()), (1, 2) | (2, 1))
                && leg.map_or(false, |square| self.board[square] == b' ')
            {
                return Ok(true);
            }
        }

        let pawn_mask = attackers(6);
        let mut pawns = pawn_mask;
        while pawns != 0 {
            let start = pawns.trailing_zeros() as usize;
            pawns &= pawns - 1;
            let source_row = row(start);
            let source_col = col(start);
            let forward = if red { -1 } else { 1 };
            if source_row + forward == target_row && source_col == target_col {
                return Ok(true);
            }
            let crossed = if !red {
                source_row >= 5
            } else {
                source_row <= 4
            };
            if crossed && source_row == target_row && (source_col - target_col).abs() == 1 {
                return Ok(true);
            }
        }

        let mut kings = attackers(0);
        while kings != 0 {
            let start = kings.trailing_zeros() as usize;
            kings &= kings - 1;
            if col(start) == target_col {
                let step = if row(start) < target_row { 1 } else { -1 };
                let mut square_row = row(start) + step;
                let mut clear = true;
                while square_row != target_row {
                    let Some(square) = index(square_row, target_col) else {
                        clear = false;
                        break;
                    };
                    if self.board[square] != b' ' {
                        clear = false;
                        break;
                    }
                    square_row += step;
                }
                if clear {
                    return Ok(true);
                }
            }
        }

        for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let mut scan_row = target_row + dr;
            let mut scan_col = target_col + dc;
            while let Some(square) = index(scan_row, scan_col) {
                let piece = self.board[square];
                if piece != b' ' {
                    if own(piece, !red) && piece.to_ascii_uppercase() == b'R' {
                        return Ok(true);
                    }
                    break;
                }
                scan_row += dr;
                scan_col += dc;
            }
            let mut jumped = false;
            scan_row = target_row + dr;
            scan_col = target_col + dc;
            while let Some(square) = index(scan_row, scan_col) {
                let piece = self.board[square];
                if piece != b' ' {
                    if jumped {
                        if own(piece, !red) && piece.to_ascii_uppercase() == b'C' {
                            return Ok(true);
                        }
                        break;
                    }
                    jumped = true;
                }
                scan_row += dr;
                scan_col += dc;
            }
        }
        Ok(false)
    }
    pub fn legal(&self) -> Result<Vec<(usize, usize)>, String> {
        let mut moves = Vec::new();
        let mut pieces = self.by_color[usize::from(!self.red_to_move)];
        while pieces != 0 {
            let start = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            for end in self.pseudo(start) {
                if self.board[end].to_ascii_uppercase() == b'K' {
                    continue;
                }
                let next = self.apply(start, end)?;
                if !next.in_check(self.red_to_move)? {
                    moves.push((start, end));
                }
            }
        }
        Ok(moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::START_FEN;

    #[test]
    fn start_has_expected_legal_moves() {
        let moves = Position::parse(START_FEN).unwrap().legal().unwrap();
        assert!(moves.contains(&(27, 36)));
        assert!(moves.contains(&(1, 20)));
    }

    #[test]
    fn kings_cannot_face_after_a_move() {
        let position = Position::parse("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1").unwrap();
        let moves = position.legal().unwrap();
        assert!(!moves.contains(&(13, 12)));
        assert!(moves.contains(&(13, 22)));
        assert!(!moves.contains(&(13, 4)));
    }

    #[test]
    fn king_cannot_enter_crossed_pawn_attack() {
        let position = Position::parse("3k5/9/9/9/9/9/9/9/3p5/4K4 w - - 0 1").unwrap();
        assert!(!position.legal().unwrap().contains(&(4, 13)));
    }

    #[test]
    fn fen_round_trip_is_stable() {
        let position = Position::parse(START_FEN).unwrap();
        assert_eq!(position.fen(), START_FEN);
    }

    #[test]
    fn fen_does_not_store_rule60_counter() {
        let position = Position::parse("4k4/9/9/9/9/9/9/9/9/4K4 w - - 37 1").unwrap();
        assert_eq!(position.fen(), "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1");
    }

    #[test]
    fn fen_rejects_invalid_positions() {
        assert!(Position::parse("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1").is_ok());
        assert!(Position::parse("4k4/9/9/9/9/9/9/9/9/9 w - - 0 1").is_err());
        assert!(Position::parse("4k4/9/9/9/9/9/9/9/4X4/4K4 w - - 0 1").is_err());
        assert!(Position::parse("4k4/P8/P8/P8/P8/P8/P3P4/9/9/4K4 w - - 0 1").is_err());
        assert!(Position::parse("4k4/9/9/9/9/9/9/9/9/2K6 w - - 0 1").is_err());
        assert!(Position::parse("4k4/9/9/9/9/9/9/9/9/2b6/4K4 w - - 0 1").is_err());
    }
}
