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
  quiet_plies: number
  result: string | null
  in_check: boolean
  legal_moves: string[]
  is_human_turn: boolean
  board: BoardCell[][]
  last_error: string | null
}

export interface ModelOption {
  id: string
  name: string
}

export interface GameArchive {
  version: 1
  savedAt: string
  mode: GameMode
  humanSide: Side | null
  snapshots: Game[]
}