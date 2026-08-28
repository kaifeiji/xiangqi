import { useEffectEvent, useState } from 'react'
import { changedMove, gameWithPreviewMove, fenToBoard, toKey } from './game-utils'
import { MoveRecord } from './move-record'
import { XiangqiBoard } from './xiangqi-board'
import type { Game, GameArchive } from './types'

interface ReplayViewProps {
  active: boolean
}

export function ReplayView({ active }: ReplayViewProps): React.JSX.Element {
  const [snapshots, setSnapshots] = useState<Game[]>([])
  const [position, setPosition] = useState(0)
  const [previewMove, setPreviewMove] = useState<string | null>(null)
  const [error, setError] = useState('')

  const loadArchive = useEffectEvent(async (file: File) => {
    try {
      const archive = JSON.parse(await file.text()) as GameArchive
      if (
        typeof archive.initial_fen !== 'string'
        || !Array.isArray(archive.snapshots)
        || archive.snapshots.length === 0
        || archive.snapshots.some((snapshot) => typeof snapshot?.fen !== 'string')
      ) {
        throw new Error('存档格式无效')
      }
      const normalized = archive.snapshots.map((snapshot, index): Game => ({
        game_id: `archive-${index}`,
        mode: archive.mode,
        human_side: archive.humanSide,
        side_to_move: snapshot.side_to_move,
        turn: snapshot.turn,
        rule60: snapshot.rule60,
        result: snapshot.result,
        in_check: false,
        legal_moves: [],
        is_human_turn: false,
        board: fenToBoard(snapshot.fen),
        last_error: null,
        mcts_debug: snapshot.mcts_debug ?? null,
      }))
      if (
        archive.initial_fen
        && normalized[0]
        && archive.initial_fen.split(' ')[0] !== archive.snapshots[0].fen.split(' ')[0]
      ) {
        normalized.unshift({
          ...normalized[0],
          game_id: 'archive-initial',
          turn: 1,
          side_to_move: archive.initial_fen.split(' ')[1] as Game['side_to_move'],
          rule60: 0,
          result: null,
          board: fenToBoard(archive.initial_fen),
          mcts_debug: null,
        })
      }
      setError('')
      setSnapshots(normalized)
      setPosition(0)
      setPreviewMove(null)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : '无法读取存档')
    }
  })

  const navigateReplay = useEffectEvent((index: number) => {
    if (index < 0 || index >= snapshots.length) return
    setPreviewMove(null)
    setPosition(index)
  })

  const game = snapshots[position]
  const previewGame = game && previewMove ? gameWithPreviewMove(game, previewMove) : undefined
  const boardGame = previewGame ?? game
  const previousGame = position > 0 ? snapshots[position - 1] : undefined
  const lastMove = previewMove
    ? (() => {
        const [origin, destination] = previewMove.split('-')
        return origin && destination ? [toKey(origin), toKey(destination)] : undefined
      })()
    : game && previousGame ? changedMove(previousGame.board, game.board, game.side_to_move) : undefined

  return (
    <section className="mode-view" hidden={!active} aria-label="回放">
      <XiangqiBoard active={active} game={boardGame} lastMove={lastMove} readOnly onMove={() => undefined} />
      <div className="side-panels">
        <aside className="controls" aria-label="回放设置">
          <label>
            加载存档
            <input type="file" accept="application/json,.json" onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) {
                setPosition(0)
                setSnapshots([])
                void loadArchive(file)
              }
              event.target.value = ''
            }} />
          </label>
        </aside>
        <MoveRecord
          snapshots={snapshots}
          status={snapshots.length ? '回放中' : '请选择存档'}
          error={error}
          currentIndex={position}
          onNavigate={navigateReplay}
          onPreviewMove={(move) => setPreviewMove((current) => current === move ? null : move)}
          previewMove={previewMove}
          keyboardNavigationEnabled={active}
        />
      </div>
    </section>
  )
}
