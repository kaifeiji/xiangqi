import { useEffect, useEffectEvent, useRef, useState } from 'react'
import type { Key } from 'xiangqiground/types'
import { request } from './api'
import { changedMove, gameWithPreviewMove, resultText, statusFor, toIccs, toKey } from './game-utils'
import { MoveRecord } from './move-record'
import { XiangqiBoard } from './xiangqi-board'
import type { Game, ModelOption, Side } from './types'

interface HumanModelViewProps {
  active: boolean
  models: ModelOption[]
  modelsLoaded: boolean
}

export function HumanModelView({ active, models, modelsLoaded }: HumanModelViewProps): React.JSX.Element {
  const [humanSide, setHumanSide] = useState<Side>('w')
  const [model, setModel] = useState('')
  const [game, setGame] = useState<Game>()
  const [lastMove, setLastMove] = useState<Key[]>()
  const [error, setError] = useState('')
  const [modelThinking, setModelThinking] = useState(false)
  const [startingGame, setStartingGame] = useState(false)
  const [pendingHumanMove, setPendingHumanMove] = useState(false)
  const [mctsSimulations, setMctsSimulations] = useState(0)
  const [snapshots, setSnapshots] = useState<Game[]>([])
  const [position, setPosition] = useState(0)
  const [previewMove, setPreviewMove] = useState<string | null>(null)
  const announcedResult = useRef<string | undefined>(undefined)

  const buildOptimisticGame = useEffectEvent((previousGame: Game, origin: Key, destination: Key): Game => {
    const originSquare = toIccs(origin)
    const destinationSquare = toIccs(destination)
    const board = previousGame.board.map((row) => row.map((cell) => ({ ...cell })))
    const originCell = board.flat().find((cell) => cell.square === originSquare)
    const destinationCell = board.flat().find((cell) => cell.square === destinationSquare)
    if (!originCell || !destinationCell || originCell.piece == null) {
      return previousGame
    }

    destinationCell.piece = originCell.piece
    originCell.piece = null
    return {
      ...previousGame,
      board,
      side_to_move: previousGame.side_to_move === 'w' ? 'b' : 'w',
      turn: previousGame.turn + 1,
      legal_moves: [],
      in_check: false,
      is_human_turn: false,
      last_error: null,
    }
  })

  useEffect(() => {
    const firstModel = models[0]?.id ?? ''
    if (!model) setModel(firstModel)
  }, [model, models])

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

  const startGame = useEffectEvent(async () => {
    if (startingGame) return
    try {
      setStartingGame(true)
      setError('')
      setGame(undefined)
      setLastMove(undefined)
      setSnapshots([])
      setPosition(0)
      setPreviewMove(null)
      announcedResult.current = undefined
      const payload = { mode: 'human-model', human_side: humanSide, model, mcts_simulations: mctsSimulations }
      const nextGame = await request<Game>('/api/games', { method: 'POST', body: JSON.stringify(payload) })
      setGame(nextGame)
      setSnapshots([nextGame])
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : String(startError))
    } finally {
      setStartingGame(false)
    }
  })

  const playHumanMove = useEffectEvent(async (origin: Key, destination: Key) => {
    if (!game || pendingHumanMove) return
    setPreviewMove(null)
    const previousGame = game
    const optimisticGame = buildOptimisticGame(previousGame, origin, destination)
    const hasOptimisticDelta = optimisticGame.turn !== previousGame.turn
    try {
      setError('')
      setPendingHumanMove(true)
      setLastMove([origin, destination])
      setGame(optimisticGame)
      if (hasOptimisticDelta) {
        setSnapshots((previousSnapshots) => {
          const nextSnapshots = [...previousSnapshots, optimisticGame]
          setPosition(nextSnapshots.length - 1)
          return nextSnapshots
        })
      }
      const confirmedGame = await request<Game>(`/api/games/${previousGame.game_id}/move`, {
        method: 'POST',
        body: JSON.stringify({ move: `${toIccs(origin)}-${toIccs(destination)}` }),
      })
      setGame((currentGame) => {
        if (!currentGame || currentGame.game_id !== confirmedGame.game_id) return currentGame
        return confirmedGame
      })
      setLastMove(changedMove(previousGame.board, confirmedGame.board, confirmedGame.side_to_move) ?? [origin, destination])
      setSnapshots((previousSnapshots) => {
        if (!hasOptimisticDelta || previousSnapshots.length === 0) {
          const nextSnapshots = [...previousSnapshots, confirmedGame]
          setPosition(nextSnapshots.length - 1)
          return nextSnapshots
        }
        const lastSnapshot = previousSnapshots[previousSnapshots.length - 1]
        if (lastSnapshot.game_id === confirmedGame.game_id && lastSnapshot.turn === optimisticGame.turn) {
          const nextSnapshots = [...previousSnapshots.slice(0, -1), confirmedGame]
          setPosition(nextSnapshots.length - 1)
          return nextSnapshots
        }
        const nextSnapshots = [...previousSnapshots, confirmedGame]
        setPosition(nextSnapshots.length - 1)
        return nextSnapshots
      })
    } catch (moveError) {
      setGame(previousGame)
      if (hasOptimisticDelta) {
        setSnapshots((previousSnapshots) => {
          if (previousSnapshots.length === 0) return previousSnapshots
          const lastSnapshot = previousSnapshots[previousSnapshots.length - 1]
          if (lastSnapshot.game_id === previousGame.game_id && lastSnapshot.turn === optimisticGame.turn) {
            const nextSnapshots = previousSnapshots.slice(0, -1)
            setPosition(Math.max(nextSnapshots.length - 1, 0))
            return nextSnapshots
          }
          return previousSnapshots
        })
      }
      setError(moveError instanceof Error ? moveError.message : String(moveError))
    } finally {
      setPendingHumanMove(false)
    }
  })

  const stepModel = useEffectEvent(async () => {
    if (!game) return
    try {
      setError('')
      setModelThinking(true)
      const nextGame = await request<Game>(`/api/games/${game.game_id}/step`, { method: 'POST', body: '{}' })
      updateGame(nextGame)
    } catch (stepError) {
      setError(stepError instanceof Error ? stepError.message : String(stepError))
    } finally {
      setModelThinking(false)
    }
  })

  useEffect(() => {
    if (!game || pendingHumanMove || game.is_human_turn || game.result) return
    void stepModel()
  }, [game?.game_id, game?.is_human_turn, game?.result, game?.turn, pendingHumanMove])

  useEffect(() => {
    if (!game?.result || announcedResult.current === game.game_id) return
    const result = game.result
    announcedResult.current = game.game_id
    window.setTimeout(() => {
      window.alert(`对局结束：${resultText(result)}`)
    }, 0)
  }, [game?.game_id, game?.result])

  const undoMove = useEffectEvent(async () => {
    if (!game || pendingHumanMove || modelThinking || snapshots.length < 2) return
    setPreviewMove(null)
    const undoPlies = game.is_human_turn ? 2 : 1
    try {
      setError('')
      setPendingHumanMove(true)
      const nextGame = await request<Game>(`/api/games/${game.game_id}/undo`, {
        method: 'POST',
        body: JSON.stringify({ plies: undoPlies }),
      })
      setGame(nextGame)
      setSnapshots((previousSnapshots) => {
        const nextLength = Math.max(previousSnapshots.length - undoPlies, 1)
        const nextSnapshots = previousSnapshots.slice(0, nextLength)
        setPosition(nextSnapshots.length - 1)
        return nextSnapshots
      })
    } catch (undoError) {
      setError(undoError instanceof Error ? undoError.message : String(undoError))
    } finally {
      setPendingHumanMove(false)
    }
  })

  const endGame = useEffectEvent(async () => {
    if (!game || pendingHumanMove || modelThinking) return
    const gameId = game.game_id
    try {
      await request(`/api/games/${gameId}/close`, { method: 'POST' })
    } catch {
    }
    setGame(undefined)
    setSnapshots([])
    setPosition(0)
    setPreviewMove(null)
    setLastMove(undefined)
    announcedResult.current = undefined
    setError('')
  })

  const navigateReplay = useEffectEvent((index: number) => {
    if (index < 0 || index >= snapshots.length) return
    setPreviewMove(null)
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
    <section className="mode-view" hidden={!active} aria-label="人机对弈">
      <XiangqiBoard active={active} game={boardGame} orientation={humanSide === 'b' ? 'black' : 'white'} lastMove={displayedLastMove} readOnly={previewMove !== null || position + 1 !== snapshots.length} onMove={playHumanMove} />
      <div className="side-panels">
        <aside className="controls" aria-label="人机设置">
          <label>
            人类执子
            <select value={humanSide} onChange={(event) => setHumanSide(event.target.value as Side)} disabled={startingGame || Boolean(game)}>
              <option value="w">红方</option>
              <option value="b">黑方</option>
            </select>
          </label>
          <label>
            {humanSide === 'w' ? '黑方模型' : '红方模型'}
            <select value={model} onChange={(event) => setModel(event.target.value)} disabled={startingGame || Boolean(game) || !models.length}>
              {models.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}
            </select>
          </label>
          {model !== 'pikafish' && <label>
              MCTS模拟
              <select value={String(mctsSimulations)} onChange={(event) => setMctsSimulations(Number(event.target.value))} disabled={startingGame || Boolean(game)}>
                <option value="0">0次</option>
                <option value="1000">1000次</option>
                <option value="5000">5000次</option>
                <option value="10000">10000次</option>
              </select>
            </label>}
          {!game && <button type="button" onClick={() => void startGame()} disabled={startingGame || !modelsLoaded || !models.length}>
            {startingGame ? '创建中...' : '开始新对局'}
          </button>}
          {game && <button type="button" onClick={endGame} disabled={startingGame || pendingHumanMove || modelThinking}>结束对局</button>}
          <button type="button" onClick={() => void undoMove()} disabled={!game || Boolean(game.result) || snapshots.length < 2 || pendingHumanMove || modelThinking}>
            悔棋
          </button>
        </aside>
        <MoveRecord
          snapshots={snapshots}
          status={statusFor(game, modelsLoaded, modelThinking)}
          error={error}
          archiveMode="human-model"
          archiveHumanSide={game?.human_side ?? humanSide}
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
