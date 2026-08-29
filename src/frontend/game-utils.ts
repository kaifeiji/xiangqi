import type { Key } from 'xiangqiground/types'
import type { BoardCell, Game, Side } from './types'

export function resultText(result: string): string {
  const texts: Record<string, string> = {
    red_win: '红方胜',
    black_win: '黑方胜',
    draw_stalemate: '判负（无合法着法）',
    draw_repetition: '和棋（三次重复局面）',
    draw_natural_limit: '和棋（自然限着）',
    draw_move_limit: '和棋（达到最大回合数）',
    draw_insufficient_material: '和棋（理论无胜）',
    black_win_long_check: '黑方胜（红方长将）',
    red_win_long_check: '红方胜（黑方长将）',
    black_win_long_chase: '黑方胜（红方长捉）',
    red_win_long_chase: '红方胜（黑方长捉）',
  }
  return texts[result] ?? '对局结束'
}

export function statusFor(game: Game | undefined, modelsLoaded: boolean, thinking: boolean): string {
  if (!modelsLoaded) return '加载模型中...'
  if (!game) return '未开始'
  if (game.result) return `对局结束：${resultText(game.result)}`
  if (thinking) return '模型思考中...'
  const side = game.side_to_move === 'w' ? '红方' : '黑方'
  return `第 ${game.turn} 回合，轮到${side}${game.in_check ? '（被将军）' : ''}`
}

export function toKey(square: string): Key {
  const file = square.slice(0, 1).toLowerCase()
  const rank = Number(square.slice(1))
  return `${file}${rank + 1}` as Key
}

export function toIccs(key: Key): string {
  const file = key.slice(0, 1).toUpperCase()
  const rank = Number(key.slice(1))
  return `${file}${rank - 1}`
}

export function boardToFen(position: BoardCell[][]): string {
  return position
    .map((row) => {
      const expanded = row.map((cell) => cell.piece ?? '1').join('')
      return expanded.replace(/1+/g, (empty) => String(empty.length))
    })
    .join('/')
}

export function fenToBoard(fen: string): BoardCell[][] {
  const rows = fen.split(' ')[0].split('/')
  return rows.map((row, rowIndex) => {
    const cells: BoardCell[] = []
    for (const token of row) {
      if (/\d/.test(token)) {
        for (let index = 0; index < Number(token); index += 1) cells.push({ piece: null, square: `${String.fromCharCode(65 + cells.length)}${9 - rowIndex}` })
      } else {
        cells.push({ piece: token, square: `${String.fromCharCode(65 + cells.length)}${9 - rowIndex}` })
      }
    }
    return cells
  })
}

export function gameWithPreviewMove(game: Game, move: string): Game | undefined {
  const searchedFen = game.mcts_debug?.searched_fen ?? game.policy_debug?.searched_fen
  const [origin, destination] = move.split('-')
  if (!searchedFen || !origin || !destination) return undefined
  const board = fenToBoard(searchedFen)
  const originCell = board.flat().find((cell) => cell.square === origin)
  if (!originCell?.piece) return undefined
  return {
    ...game,
    board: board.map((row) => row.map((cell) => cell.square === origin
      ? { ...cell, piece: null }
      : cell.square === destination
        ? { ...cell, piece: originCell.piece }
        : cell)),
    side_to_move: game.mcts_debug?.searched_side ?? game.policy_debug?.searched_side ?? game.side_to_move,
    in_check: false,
    legal_moves: [],
  }
}

export function piecesToBoard(pieces: Array<string | null>): BoardCell[][] {
  if (pieces.length !== 90) throw new Error('棋盘数据无效')
  return Array.from({ length: 10 }, (_, displayRow) => {
    const row = 9 - displayRow
    return Array.from({ length: 9 }, (_, col) => ({
      piece: pieces[row * 9 + col] ?? null,
      square: `${String.fromCharCode(65 + col)}${row}`,
    }))
  })
}

export function legalDests(moves: string[]): Map<Key, Key[]> {
  const destinations = new Map<Key, Key[]>()
  for (const move of moves) {
    const [origin, destination] = move.split('-')
    if (origin && destination) {
      const key = toKey(origin)
      destinations.set(key, [...(destinations.get(key) ?? []), toKey(destination)])
    }
  }
  return destinations
}

export function changedMove(before: BoardCell[][], after: BoardCell[][], sideToMove: Side): Key[] | undefined {
  const beforeBySquare = new Map(before.flat().map((cell) => [cell.square, cell.piece]))
  const afterBySquare = new Map(after.flat().map((cell) => [cell.square, cell.piece]))
  const isRed = sideToMove === 'b'
  let origin: string | undefined
  let destination: string | undefined

  for (const [square, piece] of beforeBySquare) {
    const nextPiece = afterBySquare.get(square)
    const isMovedPiece = piece !== null && piece === (isRed ? piece.toUpperCase() : piece.toLowerCase())
    if (isMovedPiece && nextPiece === null) {
      origin = square
    }
    const isMovedPieceAtDestination = nextPiece != null && nextPiece === (isRed ? nextPiece.toUpperCase() : nextPiece.toLowerCase())
    if (isMovedPieceAtDestination && nextPiece !== piece) {
      destination = square
    }
  }
  return origin && destination ? [toKey(origin), toKey(destination)] : undefined
}