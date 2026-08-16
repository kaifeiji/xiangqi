import { useEffectEvent, useState } from 'react'
import { changedMove } from './game-utils'
import { MoveRecord } from './move-record'
import { XiangqiBoard } from './xiangqi-board'
import type { Game, GameArchive } from './types'

interface ReplayViewProps {
  active: boolean
}

export function ReplayView({ active }: ReplayViewProps): React.JSX.Element {
  const [snapshots, setSnapshots] = useState<Game[]>([])
  const [position, setPosition] = useState(0)
  const [error, setError] = useState('')

  const loadArchive = useEffectEvent(async (file: File) => {
    try {
      const archive = JSON.parse(await file.text()) as GameArchive
      if (archive.version !== 1 || !Array.isArray(archive.snapshots) || archive.snapshots.length === 0) {
        throw new Error('存档格式无效')
      }
      setError('')
      setSnapshots(archive.snapshots)
      setPosition(0)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : '无法读取存档')
    }
  })

  const navigateReplay = useEffectEvent((index: number) => {
    if (index < 0 || index >= snapshots.length) return
    setPosition(index)
  })

  const game = snapshots[position]
  const previousGame = position > 0 ? snapshots[position - 1] : undefined
  const lastMove = game && previousGame ? changedMove(previousGame.board, game.board, game.side_to_move) : undefined

  return (
    <section className="mode-view" hidden={!active} aria-label="回放">
      <XiangqiBoard active={active} game={game} lastMove={lastMove} readOnly onMove={() => undefined} />
      <div className="side-panels">
        <aside className="controls" aria-label="回放设置">
          <label>
            加载存档
            <input type="file" accept="application/json,.json" onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void loadArchive(file)
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
        />
      </div>
    </section>
  )
}
