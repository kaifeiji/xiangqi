import { useEffect, useEffectEvent, useRef, useState } from 'react'
import type { Key } from 'xiangqiground/types'
import { request } from './api'
import { GamePanel } from './game-panel'
import { changedMove, toIccs } from './game-utils'
import { XiangqiBoard } from './xiangqi-board'
import type { Game, GameArchive, GameMode, ModelOption, Side } from './types'

interface GameViewProps {
  active: boolean
  mode: GameMode
  models: ModelOption[]
  modelsLoaded: boolean
}

function resultText(result: string): string {
  const texts: Record<string, string> = {
    red_win: '红方胜',
    black_win: '黑方胜',
    draw_stalemate: '和棋（无合法着法）',
  }
  return texts[result] ?? '对局结束'
}

function statusFor(game: Game | undefined, modelsLoaded: boolean, modelThinking: boolean): string {
  if (!modelsLoaded) return '加载模型中...'
  if (!game) return '未开始'
  if (game.result) return `对局结束：${resultText(game.result)}`
  if (modelThinking) return '模型思考中...'
  const side = game.side_to_move === 'w' ? '红方' : '黑方'
  return `第 ${game.turn} 回合，轮到${side}${game.in_check ? '（被将军）' : ''}`
}

export function GameView({ active, mode, models, modelsLoaded }: GameViewProps): React.JSX.Element {
  const [humanSide, setHumanSide] = useState<Side>('w')
  const [model, setModel] = useState('')
  const [redModel, setRedModel] = useState('')
  const [blackModel, setBlackModel] = useState('')
  const [game, setGame] = useState<Game>()
  const [lastMove, setLastMove] = useState<Key[]>()
  const [error, setError] = useState('')
  const [modelThinking, setModelThinking] = useState(false)
  const [snapshots, setSnapshots] = useState<Game[]>([])
  const announcedResult = useRef<string | undefined>(undefined)

  useEffect(() => {
    const firstModel = models[0]?.id ?? ''
    if (!model) setModel(firstModel)
    if (!redModel) setRedModel(firstModel)
    if (!blackModel) setBlackModel(firstModel)
  }, [blackModel, model, models, redModel])

  const updateGame = useEffectEvent((nextGame: Game) => {
    setGame((previousGame) => {
      if (!previousGame || previousGame.game_id !== nextGame.game_id) return previousGame
      setLastMove(changedMove(previousGame.board, nextGame.board, nextGame.side_to_move))
      setSnapshots((previousSnapshots) => [...previousSnapshots, nextGame])
      return nextGame
    })
  })

  const startGame = useEffectEvent(async () => {
    try {
      setError('')
      setGame(undefined)
      setLastMove(undefined)
      setSnapshots([])
      announcedResult.current = undefined
      const payload = mode === 'human-model'
        ? { mode, human_side: humanSide, model }
        : { mode, red_model: redModel, black_model: blackModel }
      const nextGame = await request<Game>('/api/games', { method: 'POST', body: JSON.stringify(payload) })
      setGame(nextGame)
      setSnapshots([nextGame])
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : String(startError))
    }
  })

  const playHumanMove = useEffectEvent(async (origin: Key, destination: Key) => {
    if (!game) return
    try {
      setError('')
      setLastMove([origin, destination])
      updateGame(await request<Game>(`/api/games/${game.game_id}/move`, {
        method: 'POST',
        body: JSON.stringify({ move: `${toIccs(origin)}-${toIccs(destination)}` }),
      }))
    } catch (moveError) {
      setError(moveError instanceof Error ? moveError.message : String(moveError))
    }
  })

  const stepModel = useEffectEvent(async (delayRender = false) => {
    if (!game) return
    try {
      setError('')
      setModelThinking(delayRender)
      const nextGame = await request<Game>(`/api/games/${game.game_id}/step`, { method: 'POST', body: '{}' })
      if (delayRender) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 1000))
      }
      updateGame(nextGame)
    } catch (stepError) {
      setError(stepError instanceof Error ? stepError.message : String(stepError))
    } finally {
      setModelThinking(false)
    }
  })

  useEffect(() => {
    if (!game || mode !== 'human-model' || game.is_human_turn || game.result) return
    void stepModel(true)
  }, [game?.game_id, game?.is_human_turn, game?.result, game?.turn, mode])

  useEffect(() => {
    if (!game?.result || announcedResult.current === game.game_id) return
    announcedResult.current = game.game_id
    window.alert(`对局结束：${resultText(game.result)}`)
  }, [game?.game_id, game?.result])

  const saveArchive = useEffectEvent(() => {
    if (!game || !snapshots.length) return
    const archive: GameArchive = {
      version: 1,
      savedAt: new Date().toISOString(),
      mode,
      humanSide: game.human_side,
      snapshots,
    }
    const blob = new Blob([JSON.stringify(archive, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'xiangqi-save.json'
    link.click()
    URL.revokeObjectURL(url)
  })

  return (
    <section className="mode-view" hidden={!active} aria-label={mode === 'human-model' ? '人机对弈' : '模型对弈'}>
      <XiangqiBoard active={active} game={game} lastMove={lastMove} readOnly={false} onMove={playHumanMove} />
      <GamePanel
        mode={mode}
        humanSide={humanSide}
        models={models}
        model={model}
        redModel={redModel}
        blackModel={blackModel}
        status={statusFor(game, modelsLoaded, modelThinking)}
        error={error}
        canStart={modelsLoaded && models.length > 0}
        canStep={mode === 'model-model' && Boolean(game && !game.result)}
        canSave={Boolean(game && snapshots.length)}
        onHumanSideChange={setHumanSide}
        onModelChange={setModel}
        onRedModelChange={setRedModel}
        onBlackModelChange={setBlackModel}
        onStart={startGame}
        onStep={() => void stepModel()}
        onSave={saveArchive}
      />
    </section>
  )
}
