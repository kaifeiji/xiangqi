export type GameMode = 'human-model' | 'model-model'
export type Mode = GameMode | 'replay'
export type Side = 'w' | 'b'

export interface BoardCell {
  piece: string | null
  square: string
}

export interface Game {
  game_id: string
  mode: GameMode
  human_side: Side | null
  side_to_move: Side
  turn: number
  rule60: number
  result: string | null
  in_check: boolean
  legal_moves: string[]
  is_human_turn: boolean
  board: BoardCell[][]
  last_error: string | null
  mcts_debug?: MctsDebug | null
}

export interface MctsDebug {
  simulations: number
  average_leaf_depth: number
  max_leaf_depth: number
  effective_batch_size?: number
  effective_max_depth?: number
  root_network_value?: number | null
  legal_cache_hits?: number
  legal_cache_misses?: number
  root_visits_before?: Record<string, number>
  root_children: Array<{ move: string; visits: number; new_visits?: number; q: number; prior: number }>
}

export interface ModelOption {
  id: string
  name: string
}

export interface GameArchive {
  savedAt: string
  mode: GameMode
  humanSide: Side | null
  initial_fen: string
  snapshots: CompactArchiveSnapshot[]
}

export interface CompactArchiveSnapshot {
  fen: string
  side_to_move: Side
  turn: number
  rule60: number
  result: string | null
  mcts_debug?: MctsDebug | null
}