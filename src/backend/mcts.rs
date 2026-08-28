use super::position::Position;
use super::rules::RuleState;
use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

static ONNX_MODELS: OnceLock<Mutex<HashMap<String, Arc<Mutex<OnnxEvaluator>>>>> = OnceLock::new();
const ROOT_MIN_VISITS: u32 = 4;

const PIECE_CHANNELS: [(u8, usize); 14] = [
    (b'K', 0),
    (b'A', 1),
    (b'B', 2),
    (b'N', 3),
    (b'R', 4),
    (b'C', 5),
    (b'P', 6),
    (b'k', 7),
    (b'a', 8),
    (b'b', 9),
    (b'n', 10),
    (b'r', 11),
    (b'c', 12),
    (b'p', 13),
];

fn board_tensor(position: &Position, output: &mut [f32]) {
    output.fill(0.0);
    let flip = !position.red_to_move;
    for source in 0..90 {
        let piece = position.board[source];
        let Some((_, channel)) = PIECE_CHANNELS
            .iter()
            .find(|(candidate, _)| *candidate == piece)
        else {
            continue;
        };
        let target = if flip { 89 - source } else { source };
        let target_channel = if flip {
            if *channel < 7 {
                channel + 7
            } else {
                channel - 7
            }
        } else {
            *channel
        };
        output[target_channel * 90 + target] = 1.0;
    }
    if position.red_to_move {
        output[14 * 90..15 * 90].fill(1.0);
    }
}

fn current_view_action(position: &Position, start: usize, end: usize) -> (usize, usize) {
    if position.red_to_move {
        (start, end)
    } else {
        (89 - start, 89 - end)
    }
}

pub struct OnnxEvaluator {
    session: Session,
}

pub struct MctsSearchResult {
    pub movement: (u8, u8),
    pub root_children: Vec<(u8, u8, u32, f32, f32)>,
    pub average_leaf_depth: f32,
    pub max_leaf_depth: usize,
    pub root_network_value: f32,
}

impl OnnxEvaluator {
    fn load(path: &str) -> Result<Self, String> {
        Session::builder()
            .map_err(|error| error.to_string())?
            .commit_from_file(path)
            .map(|session| Self { session })
            .map_err(|error| error.to_string())
    }

    fn evaluate(
        &mut self,
        positions: &[Position],
        legal: &[Vec<(usize, usize)>],
        temperature: f32,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        let batch = positions.len();
        let mut data = vec![0.0; batch * 15 * 90];
        for (index, position) in positions.iter().enumerate() {
            board_tensor(position, &mut data[index * 15 * 90..(index + 1) * 15 * 90]);
        }
        let input =
            Array4::from_shape_vec((batch, 15, 10, 9), data).map_err(|error| error.to_string())?;
        let input = TensorRef::from_array_view(input.view()).map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(inputs!["board" => input])
            .map_err(|error| error.to_string())?;
        let (_, logits) = outputs["move_logits"]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        let (_, values) = outputs["value"]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        let mut result = Vec::with_capacity(batch);
        for (index, moves) in legal.iter().enumerate() {
            let offset = index * 8100;
            let scores: Vec<f32> = moves
                .iter()
                .map(|&(start, end)| {
                    let (policy_start, policy_end) =
                        current_view_action(&positions[index], start, end);
                    logits[offset + policy_start * 90 + policy_end]
                })
                .collect();
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut priors: Vec<f32> = scores
                .iter()
                .map(|score| ((*score - max_score) / temperature).exp())
                .collect();
            let sum: f32 = priors.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err("ONNX policy scores are invalid".into());
            }
            for prior in &mut priors {
                *prior /= sum;
            }
            result.push((priors, values[index].clamp(-1.0, 1.0)));
        }
        Ok(result)
    }
}

struct Node {
    position: Position,
    rules: RuleState,
    prior: f32,
    visits: u32,
    value_sum: f32,
    children: Vec<(u8, u8, usize)>,
    expanded: bool,
    virtual_visits: u32,
    virtual_loss: f32,
}

fn evaluator_for(model_path: &str) -> Result<Arc<Mutex<OnnxEvaluator>>, String> {
    let models = ONNX_MODELS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut models = models
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(evaluator) = models.get(model_path) {
        return Ok(Arc::clone(evaluator));
    }
    let evaluator = Arc::new(Mutex::new(OnnxEvaluator::load(model_path)?));
    models.insert(model_path.to_owned(), Arc::clone(&evaluator));
    Ok(evaluator)
}

impl Node {
    fn new(position: Position, rules: RuleState, prior: f32) -> Self {
        Self {
            position,
            rules,
            prior,
            visits: 0,
            value_sum: 0.0,
            children: Vec::new(),
            expanded: false,
            virtual_visits: 0,
            virtual_loss: 0.0,
        }
    }
}

fn select(nodes: &mut [Node], exploration: f32, max_depth: usize) -> (Vec<usize>, usize, usize) {
    let mut path = vec![0];
    let mut current = 0;
    let mut depth = 0;
    nodes[current].virtual_visits += 1;
    nodes[current].virtual_loss += 1.0;
    while nodes[current].expanded && depth < max_depth && !nodes[current].children.is_empty() {
        let parent_visits = (nodes[current].visits + nodes[current].virtual_visits).max(1) as f32;
        let children = &nodes[current].children;
        let underexplored = current == 0
            && children.iter().any(|&(_, _, index)| {
                nodes[index].visits + nodes[index].virtual_visits < ROOT_MIN_VISITS
            });
        let child_index = children
            .iter()
            .copied()
            .max_by(|left, right| {
                let score = |index: usize| {
                    let child = &nodes[index];
                    let visits = child.visits + child.virtual_visits;
                    if underexplored {
                        return if visits < ROOT_MIN_VISITS {
                            (ROOT_MIN_VISITS - visits) as f32 + child.prior
                        } else {
                            child.prior - ROOT_MIN_VISITS as f32
                        };
                    }
                    let q = if visits == 0 {
                        0.0
                    } else {
                        (child.value_sum - child.virtual_loss) / visits as f32
                    };
                    -q + exploration * child.prior * parent_visits.sqrt() / (1 + visits) as f32
                };
                score(left.2)
                    .partial_cmp(&score(right.2))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("expanded node has a child")
            .2;
        current = child_index;
        nodes[current].virtual_visits += 1;
        nodes[current].virtual_loss += 1.0;
        path.push(current);
        depth += 1;
    }
    (path, current, depth)
}

fn backup(nodes: &mut [Node], path: &[usize], mut value: f32) {
    for &index in path.iter().rev() {
        nodes[index].virtual_visits -= 1;
        nodes[index].virtual_loss -= 1.0;
        nodes[index].visits += 1;
        nodes[index].value_sum += value;
        value = -value;
    }
}

pub fn mcts_search_onnx(
    root_fen: &str,
    model_path: &str,
    simulations: usize,
    exploration: f32,
    batch_size: usize,
    max_depth: usize,
) -> Result<MctsSearchResult, String> {
    if simulations == 0
        || batch_size == 0
        || max_depth == 0
        || !exploration.is_finite()
        || exploration <= 0.0
    {
        return Err("invalid MCTS parameters".into());
    }
    let evaluator = evaluator_for(model_path)?;
    let mut evaluator = evaluator
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = Position::parse(root_fen)?;
    let rules = RuleState::new(root_fen, &root)?;
    let mut nodes = vec![Node::new(root, rules, 1.0)];
    let timing_enabled = std::env::var("MCTS_TIMING_LOG").as_deref() != Ok("0");
    let total_started = Instant::now();
    let mut selection_time = Duration::ZERO;
    let mut primary_inference_time = Duration::ZERO;
    let mut completed = 0;
    let mut leaf_depth_sum = 0usize;
    let mut max_leaf_depth = 0usize;
    let mut root_network_value = None;
    while completed < simulations {
        let count = if !nodes[0].expanded {
            1
        } else {
            batch_size.min(simulations - completed)
        };
        let mut pending: Vec<(usize, Vec<Vec<usize>>, Vec<(usize, usize)>)> = Vec::new();
        let mut pending_indices: HashMap<usize, usize> = HashMap::new();
        for _ in 0..count {
            let selection_started = Instant::now();
            let (path, leaf_index, depth) = select(&mut nodes, exploration, max_depth);
            selection_time += selection_started.elapsed();
            leaf_depth_sum += depth;
            max_leaf_depth = max_leaf_depth.max(depth);
            if let Some(value) = nodes[leaf_index]
                .rules
                .terminal_value(&nodes[leaf_index].position)
            {
                backup(&mut nodes, &path, value);
            } else if depth >= max_depth {
                backup(&mut nodes, &path, 0.0);
            } else {
                let legal = nodes[leaf_index].position.legal()?;
                if let Some(&pending_index) = pending_indices.get(&leaf_index) {
                    pending[pending_index].1.push(path);
                } else {
                    pending_indices.insert(leaf_index, pending.len());
                    pending.push((leaf_index, vec![path], legal));
                }
            }
        }
        if !pending.is_empty() {
            let positions: Vec<Position> = pending
                .iter()
                .map(|(index, _, _)| nodes[*index].position)
                .collect();
            let legal: Vec<Vec<(usize, usize)>> =
                pending.iter().map(|(_, _, moves)| moves.clone()).collect();
            let temperature = std::env::var("MCTS_POLICY_TEMPERATURE")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.25);
            let primary_started = Instant::now();
            let predictions = evaluator.evaluate(&positions, &legal, temperature)?;
            primary_inference_time += primary_started.elapsed();
            for ((_, (leaf_index, paths, legal)), (priors, value)) in
                pending.into_iter().enumerate().zip(predictions)
            {
                if priors.len() != legal.len()
                    || !value.is_finite()
                    || !(-1.0..=1.0).contains(&value)
                {
                    return Err("invalid evaluator result".into());
                }
                let position = nodes[leaf_index].position;
                if paths.iter().any(|path| path.len() == 1) {
                    root_network_value = Some(value);
                }
                let mut children = Vec::with_capacity(legal.len());
                for ((start, end), prior) in legal.into_iter().zip(priors) {
                    if !prior.is_finite() || prior < 0.0 {
                        return Err("invalid policy prior".into());
                    }
                    let child_index = nodes.len();
                    let (child_position, child_rules) =
                        nodes[leaf_index].rules.child(&position, start, end)?;
                    nodes.push(Node::new(child_position, child_rules, prior));
                    children.push((start as u8, end as u8, child_index));
                }
                nodes[leaf_index].children = children;
                nodes[leaf_index].expanded = true;
                for path in paths {
                    backup(&mut nodes, &path, value);
                }
            }
        }
        completed += count;
    }
    let root = &nodes[0];
    let best = root
        .children
        .iter()
        .max_by_key(|child| nodes[child.2].visits)
        .ok_or_else(|| "MCTS root has no children".to_owned())?;
    let stats = root
        .children
        .iter()
        .map(|&(start, end, index)| {
            let child = &nodes[index];
            (
                start,
                end,
                child.visits,
                if child.visits == 0 {
                    0.0
                } else {
                    child.value_sum / child.visits as f32
                },
                child.prior,
            )
        })
        .collect();
    if timing_enabled {
        let total = total_started.elapsed();
        eprintln!(
            "[MCTS timing] simulations={} batch={} total={:.3}s select={:.3}s onnx_primary={:.3}s",
            simulations,
            batch_size,
            total.as_secs_f64(),
            selection_time.as_secs_f64(),
            primary_inference_time.as_secs_f64(),
        );
    }
    Ok(MctsSearchResult {
        movement: (best.0, best.1),
        root_children: stats,
        average_leaf_depth: leaf_depth_sum as f32 / simulations as f32,
        max_leaf_depth,
        root_network_value: root_network_value.unwrap_or(0.0),
    })
}

pub fn policy_search_onnx(root_fen: &str, model_path: &str) -> Result<(u8, u8), String> {
    let evaluator = evaluator_for(model_path)?;
    let mut evaluator = evaluator
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = Position::parse(root_fen)?;
    let legal = root.legal()?;
    if legal.is_empty() {
        return Err("position has no legal moves".into());
    }
    let (priors, _) = evaluator
        .evaluate(&[root], std::slice::from_ref(&legal), 1.0)?
        .into_iter()
        .next()
        .ok_or_else(|| "missing evaluator result".to_owned())?;
    legal
        .into_iter()
        .zip(priors)
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .map(|(movement, _)| (movement.0 as u8, movement.1 as u8))
        .ok_or_else(|| "policy has no legal move".to_owned())
}

pub(crate) fn search_onnx(
    root_fen: &str,
    model_path: &str,
    simulations: usize,
    exploration: f32,
    batch_size: usize,
    max_depth: usize,
) -> Result<MctsSearchResult, String> {
    mcts_search_onnx(
        root_fen,
        model_path,
        simulations,
        exploration,
        batch_size,
        max_depth,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_action_uses_current_view_coordinates() {
        let red = Position::parse(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )
        .unwrap();
        let black = Position::parse(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1",
        )
        .unwrap();

        assert_eq!(current_view_action(&red, 0, 9), (0, 9));
        assert_eq!(current_view_action(&black, 0, 9), (89, 80));
    }

    #[test]
    fn backup_flips_value_for_each_parent_ply() {
        let position = Position::parse(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )
        .unwrap();
        let rules = RuleState::new(crate::START_FEN, &position).unwrap();
        let mut nodes = vec![
            Node::new(position, rules.clone(), 1.0),
            Node::new(position, rules, 1.0),
        ];
        nodes[0].virtual_visits = 1;
        nodes[1].virtual_visits = 1;
        backup(&mut nodes, &[0, 1], 0.75);

        assert_eq!(nodes[1].visits, 1);
        assert_eq!(nodes[1].value_sum, 0.75);
        assert_eq!(nodes[0].visits, 1);
        assert_eq!(nodes[0].value_sum, -0.75);
        assert_eq!(nodes[0].virtual_visits, 0);
        assert_eq!(nodes[1].virtual_visits, 0);
    }

    #[test]
    fn selection_reserves_root_without_expanding_it() {
        let position = Position::parse(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )
        .unwrap();
        let rules = RuleState::new(crate::START_FEN, &position).unwrap();
        let mut nodes = vec![Node::new(position, rules, 1.0)];
        let (path, leaf, depth) = select(&mut nodes, 1.25, 8);

        assert_eq!(path, vec![0]);
        assert_eq!(leaf, 0);
        assert_eq!(depth, 0);
        assert_eq!(nodes[0].virtual_visits, 1);
    }

    #[test]
    fn search_rules_end_at_the_natural_limit() {
        let position = Position::parse("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 119 1").unwrap();
        let rules = RuleState::new("4k4/9/9/9/9/9/9/9/4R4 w - - 119 1", &position).unwrap();
        let (child, child_rules) = rules.child(&position, 13, 22).unwrap();

        assert_eq!(child_rules.rule60, 120);
        assert_eq!(child_rules.terminal_value(&child), Some(0.0));
    }
}
