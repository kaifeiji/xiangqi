import type { Key } from 'xiangqiground/types'
import type { BoardCell, Side } from './types'

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