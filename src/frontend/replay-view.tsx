import { useEffectEvent, useState } from 'react'
import type { Key } from 'xiangqiground/types'
import { changedMove } from './game-utils'
import { ReplayPanel } from './replay-panel'
import { XiangqiBoard } from './xiangqi-board'
import type { Game, GameArchive } from './types'

interface ReplayViewProps {
  active: boolean
}

export function ReplayView({ active }: ReplayViewProps): React.JSX.Element {
  const [snapshots, setSnapshots] = useState<Game[]>([])
  const [position, setPosition] = useState(0)
  const [lastMove, setLastMove] = useState<Key[]>()
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
      setLastMove(undefined)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : '无法读取存档')
    }
  })

  const step = useEffectEvent((offset: number) => {
    const nextPosition = position + offset
    if (nextPosition < 0 || nextPosition >= snapshots.length) return
    const nextGame = snapshots[nextPosition]
    const previousGame = snapshots[position]
    setPosition(nextPosition)
    setLastMove(changedMove(previousGame.board, nextGame.board, nextGame.side_to_move))
  })

  const game = snapshots[position]

  return (
    <section className="mode-view" hidden={!active} aria-label="回放">
      <XiangqiBoard active={active} game={game} lastMove={lastMove} readOnly onMove={() => undefined} />
      <ReplayPanel
        error={error}
        position={position}
        length={snapshots.length}
        onLoad={loadArchive}
        onStep={step}
      />
    </section>
  )
}
