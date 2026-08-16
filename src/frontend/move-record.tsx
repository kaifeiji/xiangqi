import { useEffect, useRef } from 'react'
import { changedMove, toIccs } from './game-utils'
import type { Game, GameArchive, GameMode, Side } from './types'

interface MoveRecordProps {
  snapshots: Game[]
  status: string
  error: string
  archiveMode?: GameMode
  archiveHumanSide?: Side | null
  currentIndex?: number
  onNavigate?: (index: number) => void
}

export function MoveRecord({ snapshots, status, error, archiveMode, archiveHumanSide, currentIndex, onNavigate }: MoveRecordProps): React.JSX.Element {
  const moves = snapshots.slice(1).map((snapshot, index) => {
    const previous = snapshots[index]
    const move = changedMove(previous.board, snapshot.board, snapshot.side_to_move)
    if (!move) return undefined
    const side = snapshot.side_to_move === 'b' ? '红方' : '黑方'
    return {
      id: `${snapshot.game_id}-${snapshot.turn}`,
      number: index + 1,
      snapshotIndex: index + 1,
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

  const saveArchive = (): void => {
    if (!archiveMode || snapshots.length === 0) return
    const archive: GameArchive = {
      version: 1,
      savedAt: new Date().toISOString(),
      mode: archiveMode,
      humanSide: archiveHumanSide ?? null,
      snapshots,
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
