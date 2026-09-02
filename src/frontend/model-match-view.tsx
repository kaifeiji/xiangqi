import { useEffect, useEffectEvent, useRef, useState } from 'react'
import type { Key } from 'xiangqiground/types'
import { request } from './api'
import { changedMove, gameWithPreviewMove, resultText, toKey } from './game-utils'
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
  const [singleStepMode, setSingleStepMode] = useState(false)
  const [previewMove, setPreviewMove] = useState<string | null>(null)
  const [mctsSimulations, setMctsSimulations] = useState(0)
  const announcedResult = useRef<string | undefined>(undefined)

  useEffect(() => {
    const firstModel = models[0]?.id ?? ''
    if (!redModel) setRedModel(firstModel)
    if (!blackModel) setBlackModel(firstModel)
  }, [blackModel, models, redModel])

  const updateGame = useEffectEvent((nextGame: Game) => {
    setPreviewMove(null)
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

  const startGame = useEffectEvent(async (singleStep: boolean) => {
    if (startingGame) return
    try {
      setStartingGame(true)
      setError('')
      setThinking(false)
      setAutoPlay(!singleStep)
      setSingleStepMode(singleStep)
      setPreviewMove(null)
      setGame(undefined)
      setLastMove(undefined)
      setSnapshots([])
      setPosition(0)
      announcedResult.current = undefined
      const nextGame = await request<Game>('/api/games', {
        method: 'POST',
        body: JSON.stringify({ mode: 'model-model', red_model: redModel, black_model: blackModel, mcts_simulations: mctsSimulations }),
      })
      setGame(nextGame)
      setSnapshots([nextGame])
      setPosition(0)
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : String(startError))
    } finally {
      setStartingGame(false)
    }
  })

  const endGame = useEffectEvent(async () => {
    if (!game) return
    const gameId = game.game_id
    try {
      await request(`/api/games/${gameId}/close`, { method: 'POST' })
    } catch {
    }
    setAutoPlay(false)
    setSingleStepMode(false)
    setPreviewMove(null)
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

  const continueModel = useEffectEvent(() => {
    if (!game || game.result || thinking) return
    setPosition(Math.max(snapshots.length - 1, 0))
    setAutoPlay(true)
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
    setPreviewMove(null)
    setAutoPlay(false)
    setPosition(index)
  })

  const displayedGame = snapshots[position] ?? game
  const previewGame = displayedGame && previewMove ? gameWithPreviewMove(displayedGame, previewMove) : undefined
  const boardGame = previewGame ?? displayedGame
  const displayedPreviousGame = position > 0 ? snapshots[position - 1] : undefined
  const displayedLastMove = previewMove
    ? (() => {
        const [origin, destination] = previewMove.split('-')
        return origin && destination ? [toKey(origin), toKey(destination)] : undefined
      })()
    : displayedGame && displayedPreviousGame
    ? changedMove(displayedPreviousGame.board, displayedGame.board, displayedGame.side_to_move)
    : lastMove

  return (
    <section className="mode-view" hidden={!active} aria-label="模型对弈">
      <XiangqiBoard active={active} game={boardGame} lastMove={displayedLastMove} readOnly onMove={() => undefined} />
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
          {!(redModel === 'pikafish' && blackModel === 'pikafish') && <label>
              MCTS模拟
              <select value={String(mctsSimulations)} onChange={(event) => setMctsSimulations(Number(event.target.value))} disabled={startingGame || Boolean(game)}>
                <option value="0">0次</option>
                <option value="1000">1000次</option>
                <option value="5000">5000次</option>
                <option value="10000">10000次</option>
              </select>
            </label>}
          <div className="game-actions">
            {!game && <>
              <button type="button" onClick={() => void startGame(false)} disabled={startingGame || !modelsLoaded || !models.length}>
                {startingGame ? '创建中...' : '开始'}
              </button>
              <button type="button" onClick={() => void startGame(true)} disabled={startingGame || !modelsLoaded || !models.length}>
                {startingGame ? '创建中...' : '单步'}
              </button>
            </>}
            {game && <button type="button" onClick={endGame} disabled={startingGame}>结束</button>}
            {game && !game.result && (singleStepMode ? (
              <button id="step-button" type="button" onClick={() => {
                if (position + 1 !== snapshots.length) setPosition(snapshots.length - 1)
                void stepModel()
              }} disabled={thinking || previewMove !== null}>
                下一步
              </button>
            ) : (
              <button id="step-button" type="button" onClick={() => autoPlay ? setAutoPlay(false) : continueModel()} disabled={previewMove !== null || (!autoPlay && thinking)}>
                {autoPlay ? '暂停' : '继续'}
              </button>
            ))}
          </div>
        </aside>
        <MoveRecord
          snapshots={snapshots}
          error={error}
          archiveMode="model-model"
          archiveHumanSide={null}
          currentIndex={position}
          onNavigate={navigateReplay}
          onPreviewMove={(move) => {
            setAutoPlay(false)
            setPreviewMove((current) => current === move ? null : move)
          }}
          previewMove={previewMove}
          keyboardNavigationEnabled={active}
        />
      </div>
    </section>
  )
}
