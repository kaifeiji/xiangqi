import { useEffect, useEffectEvent, useRef } from 'react'
import { changedMove, toIccs } from './game-utils'
import { boardToFen } from './game-utils'
import type { CompactArchiveSnapshot, Game, GameArchive, GameMode, Side } from './types'

interface MoveRecordProps {
  snapshots: Game[]
  status: string
  error: string
  archiveMode?: GameMode
  archiveHumanSide?: Side | null
  currentIndex?: number
  onNavigate?: (index: number) => void
  keyboardNavigationEnabled?: boolean
}

export function MoveRecord({ snapshots, status, error, archiveMode, archiveHumanSide, currentIndex, onNavigate, keyboardNavigationEnabled = false }: MoveRecordProps): React.JSX.Element {
  const moves = snapshots.map((snapshot, index) => {
    if (index === 0) {
      return {
        id: `${snapshot.game_id}-${snapshot.turn}`,
        number: 0,
        snapshotIndex: 0,
        text: '初始局面',
      }
    }

    const previous = snapshots[index - 1]
    const move = changedMove(previous.board, snapshot.board, snapshot.side_to_move)
    if (!move) return undefined
    const side = snapshot.side_to_move === 'b' ? '红方' : '黑方'
    return {
      id: `${snapshot.game_id}-${snapshot.turn}`,
      number: index,
      snapshotIndex: index,
      text: `${side} ${toIccs(move[0])}-${toIccs(move[1])}`,
    }
  }).filter((move): move is { id: string; number: number; snapshotIndex: number; text: string } => move !== undefined)

  const activeMoveRef = useRef<HTMLLIElement | null>(null)

  useEffect(() => {
    activeMoveRef.current?.scrollIntoView({ block: 'nearest' })
  }, [currentIndex])

  const replayEnabled = typeof currentIndex === 'number' && typeof onNavigate === 'function'
  const currentPly = currentIndex ?? Math.max(snapshots.length - 1, 0)
  const maxPly = Math.max(snapshots.length - 1, 0)
  const canSaveArchive = Boolean(archiveMode) && snapshots.length > 0

  useEffect(() => {
    const mctsDebug = snapshots[currentPly]?.mcts_debug
    if (mctsDebug) console.log('[MCTS]', mctsDebug)
  }, [currentPly, snapshots])

  const navigateWithKeyboard = useEffectEvent((event: KeyboardEvent): void => {
    if (!keyboardNavigationEnabled || !replayEnabled || !onNavigate) return
    const nextIndex = event.key === 'ArrowLeft' ? currentPly - 1 : event.key === 'ArrowRight' ? currentPly + 1 : currentPly
    if (nextIndex === currentPly || nextIndex < 0 || nextIndex > maxPly) return
    event.preventDefault()
    onNavigate(nextIndex)
  })

  useEffect(() => {
    window.addEventListener('keydown', navigateWithKeyboard)
    return () => window.removeEventListener('keydown', navigateWithKeyboard)
  }, [])

  const saveArchive = (): void => {
    if (!archiveMode || snapshots.length === 0) return
    const archive: GameArchive = {
      savedAt: new Date().toISOString(),
      mode: archiveMode,
      humanSide: archiveHumanSide ?? null,
      initial_fen: `${boardToFen(snapshots[0].board)} ${snapshots[0].side_to_move} - - 0 1`,
      snapshots: snapshots.map((snapshot): CompactArchiveSnapshot => ({
        fen: boardToFen(snapshot.board),
        side_to_move: snapshot.side_to_move,
        turn: snapshot.turn,
        rule60: snapshot.rule60,
        result: snapshot.result,
        mcts_debug: snapshot.mcts_debug ?? null,
      })),
    }
    const blob = new Blob([JSON.stringify(archive, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'xiangqi-save.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="move-record" aria-label="行棋记录">
      <header className="record-header">
        <div>
          <h2>行棋记录</h2>
          <p className="status" role="status">{status}</p>
        </div>
        {canSaveArchive && <button className="icon-button" type="button" aria-label="保存存档" title="保存存档" onClick={saveArchive}>&#8595;</button>}
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {replayEnabled && (
        <div className="record-controls">
          <button type="button" onClick={() => onNavigate(currentPly - 1)} disabled={currentPly <= 0}>上一步</button>
          <span>{currentPly} / {maxPly}</span>
          <button type="button" onClick={() => onNavigate(currentPly + 1)} disabled={currentPly >= maxPly}>下一步</button>
        </div>
      )}
      <ol className="move-list">
        {moves.map((move) => {
          const isActive = currentPly === move.snapshotIndex
          return (
            <li
              key={move.id}
              ref={isActive ? activeMoveRef : null}
              className={isActive ? 'active' : undefined}
            >
              <button
                type="button"
                className="move-entry"
                onClick={() => onNavigate?.(move.snapshotIndex)}
                disabled={!onNavigate}
                aria-current={isActive ? 'true' : undefined}
              >
                <span className="move-number">{move.number}.</span>{move.text}
              </button>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
