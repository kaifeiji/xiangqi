pub mod game;
pub mod mcts;
pub(crate) mod position;
pub(crate) mod rules;

pub const START_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

pub fn parse_iccs_move(value: &str) -> Option<(u8, u8)> {
    let compact = value.replace('-', "").to_ascii_uppercase();
    if compact.len() != 4 {
        return None;
    }
    let file = |value: u8| value.checked_sub(b'A').filter(|&value| value < 9);
    let parse = |offset: usize| {
        Some((
            file(compact.as_bytes()[offset])?,
            compact.as_bytes()[offset + 1].checked_sub(b'0')?,
        ))
    };
    let (start_file, start_rank) = parse(0)?;
    let (end_file, end_rank) = parse(2)?;
    Some((start_rank * 9 + start_file, end_rank * 9 + end_file))
}
