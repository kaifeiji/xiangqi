import { useEffect, useEffectEvent, useRef, useState } from 'react'
import type { Key } from 'xiangqiground/types'
import { request } from './api'
import { changedMove, resultText, statusFor } from './game-utils'
import { MoveRecord } from './move-record'
import { XiangqiBoard } from './xiangqi-board'
import type { Game, ModelOption } from './types'

interface ModelMatchViewProps {
  active: boolean
  models: ModelOption[]
  modelsLoaded: boolean
}

export function ModelMatchView({ active, models, modelsLoaded }: ModelMatchViewProps): React.JSX.Element {
  const [redModel, setRedModel] = useState('')
  const [blackModel, setBlackModel] = useState('')
  const [game, setGame] = useState<Game>()
  const [lastMove, setLastMove] = useState<Key[]>()
  const [snapshots, setSnapshots] = useState<Game[]>([])
  const [position, setPosition] = useState(0)
  const [error, setError] = useState('')
  const [startingGame, setStartingGame] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [autoPlay, setAutoPlay] = useState(true)
  const [mctsTimeSeconds, setMctsTimeSeconds] = useState(0)
  const announcedResult = useRef<string | undefined>(undefined)

  useEffect(() => {
    const firstModel = models[0]?.id ?? ''
    if (!redModel) setRedModel(firstModel)
    if (!blackModel) setBlackModel(firstModel)
  }, [blackModel, models, redModel])

  const updateGame = useEffectEvent((nextGame: Game) => {
    setGame((previousGame) => {
      if (!previousGame || previousGame.game_id !== nextGame.game_id) return previousGame
      setLastMove(changedMove(previousGame.board, nextGame.board, nextGame.side_to_move))
      setSnapshots((previousSnapshots) => {
        const nextSnapshots = [...previousSnapshots, nextGame]
        setPosition(nextSnapshots.length - 1)
        return nextSnapshots
      })
      return nextGame
    })
  })

  const startGame = useEffectEvent(async () => {
    if (startingGame) return
    try {
      setStartingGame(true)
      setError('')
      setThinking(false)
      setAutoPlay(true)
      setGame(undefined)
      setLastMove(undefined)
      setSnapshots([])
      setPosition(0)
      announcedResult.current = undefined
      const nextGame = await request<Game>('/api/games', {
        method: 'POST',
        body: JSON.stringify({ mode: 'model-model', red_model: redModel, black_model: blackModel, mcts_time_seconds: mctsTimeSeconds }),
      })
      setGame(nextGame)
      setSnapshots([nextGame])
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : String(startError))
    } finally {
      setStartingGame(false)
    }
  })

  const endGame = useEffectEvent(async () => {
    if (!game || thinking) return
    const gameId = game.game_id
    try {
      await request(`/api/games/${gameId}/close`, { method: 'POST' })
    } catch (endError) {
      setError(endError instanceof Error ? endError.message : String(endError))
      return
    }
    setAutoPlay(false)
    setGame(undefined)
    setSnapshots([])
    setPosition(0)
    setLastMove(undefined)
    announcedResult.current = undefined
    setError('')
  })

  const stepModel = useEffectEvent(async () => {
    if (!game || game.result || thinking) return
    try {
      setError('')
      setThinking(true)
      const nextGame = await request<Game>(`/api/games/${game.game_id}/step`, { method: 'POST', body: '{}' })
      updateGame(nextGame)
    } catch (stepError) {
      setError(stepError instanceof Error ? stepError.message : String(stepError))
      setAutoPlay(false)
    } finally {
      setThinking(false)
    }
  })

  useEffect(() => {
    if (!game || !autoPlay || game.result || thinking || position + 1 !== snapshots.length) return
    void stepModel()
  }, [autoPlay, game?.game_id, game?.result, game?.turn, thinking, position, snapshots.length])

  useEffect(() => {
    if (!game?.result || announcedResult.current === game.game_id) return
    const result = game.result
    announcedResult.current = game.game_id
    window.setTimeout(() => {
      window.alert(`对局结束：${resultText(result)}`)
    }, 0)
  }, [game?.game_id, game?.result])

  const navigateReplay = useEffectEvent((index: number) => {
    if (index < 0 || index >= snapshots.length) return
    setPosition(index)
  })

  const displayedGame = snapshots[position] ?? game
  const displayedPreviousGame = position > 0 ? snapshots[position - 1] : undefined
  const displayedLastMove = displayedGame && displayedPreviousGame
    ? changedMove(displayedPreviousGame.board, displayedGame.board, displayedGame.side_to_move)
    : lastMove

  return (
    <section className="mode-view" hidden={!active} aria-label="模型对弈">
      <XiangqiBoard active={active} game={displayedGame} lastMove={displayedLastMove} readOnly onMove={() => undefined} />
      <div className="side-panels">
        <aside className="controls" aria-label="模型对弈设置">
          <label>
            红方模型
            <select value={redModel} onChange={(event) => setRedModel(event.target.value)} disabled={startingGame || Boolean(game) || !models.length}>
              {models.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}
            </select>
          </label>
          <label>
            黑方模型
            <select value={blackModel} onChange={(event) => setBlackModel(event.target.value)} disabled={startingGame || Boolean(game) || !models.length}>
              {models.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}
            </select>
          </label>
          <label>
            MCTS时间
            <select value={String(mctsTimeSeconds)} onChange={(event) => setMctsTimeSeconds(Number(event.target.value))} disabled={startingGame || Boolean(game)}>
              <option value="0">不加 MCTS</option>
              <option value="1">1 秒</option>
              <option value="3">3 秒</option>
              <option value="5">5 秒</option>
              <option value="10">10 秒</option>
            </select>
          </label>
          {!game && <button type="button" onClick={() => void startGame()} disabled={startingGame || !modelsLoaded || !models.length}>
            {startingGame ? '创建中...' : '开始新对局'}
          </button>}
          {game && <button type="button" onClick={endGame} disabled={startingGame || thinking}>结束对局</button>}
          {game && !game.result && <button id="step-button" type="button" onClick={() => setAutoPlay((enabled) => !enabled)}>{autoPlay ? '暂停' : '继续'}</button>}
        </aside>
        <MoveRecord
          snapshots={snapshots}
          status={statusFor(game, modelsLoaded, thinking)}
          error={error}
          archiveMode="model-model"
          archiveHumanSide={null}
          currentIndex={position}
          onNavigate={navigateReplay}
        />
      </div>
    </section>
  )
}
