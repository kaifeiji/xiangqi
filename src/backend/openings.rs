pub struct Opening {
    pub movement: (u8, u8),
    pub weight: u32,
    pub chinese_notation: &'static str,
}

pub const MAINSTREAM_OPENINGS: &[Opening] = &[
    Opening { movement: (19, 22), weight: 24, chinese_notation: "炮二平五" },
    Opening { movement: (25, 22), weight: 24, chinese_notation: "炮八平五" },
    Opening { movement: (19, 23), weight: 6, chinese_notation: "炮二平四" },
    Opening { movement: (25, 21), weight: 6, chinese_notation: "炮八平六" },
    Opening { movement: (19, 21), weight: 5, chinese_notation: "炮二平六" },
    Opening { movement: (25, 23), weight: 5, chinese_notation: "炮八平四" },
    Opening { movement: (1, 20), weight: 8, chinese_notation: "马二进三" },
    Opening { movement: (7, 24), weight: 8, chinese_notation: "马八进七" },
    Opening { movement: (2, 22), weight: 4, chinese_notation: "相三进五" },
    Opening { movement: (6, 22), weight: 4, chinese_notation: "相七进五" },
    Opening { movement: (31, 40), weight: 6, chinese_notation: "兵五进一" },
    Opening { movement: (29, 38), weight: 2, chinese_notation: "兵三进一" },
    Opening { movement: (33, 42), weight: 2, chinese_notation: "兵七进一" },
];

#[cfg(test)]
mod tests {
    use super::MAINSTREAM_OPENINGS;
    use crate::position::Position;
    use crate::START_FEN;

    #[test]
    fn mainstream_openings_are_legal_from_the_start_position() {
        let legal = Position::parse(START_FEN).unwrap().legal().unwrap();

        for opening in MAINSTREAM_OPENINGS {
            assert!(legal.contains(&(opening.movement.0 as usize, opening.movement.1 as usize)));
            assert!(!opening.chinese_notation.is_empty());
        }
    }
}