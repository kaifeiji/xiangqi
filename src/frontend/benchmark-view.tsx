import { useEffect, useEffectEvent, useState } from 'react'
import { request } from './api'
import { changedMove, fenToBoard, gameWithPreviewMove, resultText, toKey } from './game-utils'
import { MoveRecord } from './move-record'
import { XiangqiBoard } from './xiangqi-board'
import type { Benchmark, BenchmarkGame, CompactArchiveSnapshot, Game, ModelOption } from './types'

interface BenchmarkViewProps {
  active: boolean
  models: ModelOption[]
  modelsLoaded: boolean
}

const simulationOptions = [0, 1000, 5000, 10000]

function modelName(models: ModelOption[], id: string): string {
  return models.find((model) => model.id === id)?.name ?? id
}

function statusText(status: Benchmark['status']): string {
  return { running: '运行中', paused: '已暂停', completed: '已完成', cancelled: '已暂停', failed: '失败' }[status]
}

function timeText(milliseconds: number | null): string {
  return milliseconds ? new Date(milliseconds).toLocaleString('zh-CN', { hour12: false }) : '进行中'
}

export function BenchmarkView({ active, models, modelsLoaded }: BenchmarkViewProps): React.JSX.Element {
  const [firstModel, setFirstModel] = useState('')
  const [secondModel, setSecondModel] = useState('')
  const [simulations, setSimulations] = useState(1000)
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [viewer, setViewer] = useState<{ benchmarkId: string; gameNumber: number; position: number; previewMove: string | null } | null>(null)

  const load = useEffectEvent(async () => {
    try {
      setBenchmarks(await request<Benchmark[]>('/api/benchmarks'))
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError))
    }
  })

  useEffect(() => {
    if (!firstModel) setFirstModel(models[0]?.id ?? '')
    if (!secondModel) setSecondModel(models[1]?.id ?? models[0]?.id ?? '')
  }, [firstModel, models, secondModel])

  useEffect(() => {
    if (active) void load()
  }, [active])

  useEffect(() => {
    if (!active || !benchmarks.some((benchmark) => benchmark.status === 'running')) return
    const timer = window.setInterval(() => void load(), 5_000)
    return () => window.clearInterval(timer)
  }, [active, benchmarks])

  const create = useEffectEvent(async () => {
    try {
      setCreating(true)
      setError('')
      const benchmark = await request<Benchmark>('/api/benchmarks', {
        method: 'POST',
        body: JSON.stringify({ first_model: firstModel, second_model: secondModel, mcts_simulations: simulations }),
      })
      setBenchmarks((current) => [benchmark, ...current])
      setExpanded(benchmark.id)
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : String(createError))
    } finally {
      setCreating(false)
    }
  })

  const pause = useEffectEvent(async (id: string) => {
    try {
      await request<void>(`/api/benchmarks/${id}`, { method: 'DELETE' })
      await load()
    } catch (pauseError) {
      setError(pauseError instanceof Error ? pauseError.message : String(pauseError))
    }
  })

  const resume = useEffectEvent(async (id: string) => {
    try {
      await request<void>(`/api/benchmarks/${id}/resume`, { method: 'POST', body: '{}' })
      await load()
    } catch (resumeError) {
      setError(resumeError instanceof Error ? resumeError.message : String(resumeError))
    }
  })

  const openViewer = useEffectEvent((benchmarkId: string, gameNumber: number) => {
    const game = benchmarks.find((benchmark) => benchmark.id === benchmarkId)?.games.find((entry) => entry.number === gameNumber)
    setViewer({ benchmarkId, gameNumber, position: Math.max((game?.snapshots?.length ?? 1) - 1, 0), previewMove: null })
  })

  const currentBenchmark = viewer ? benchmarks.find((benchmark) => benchmark.id === viewer.benchmarkId) : undefined
  const currentGame = currentBenchmark?.games.find((game) => game.number === viewer?.gameNumber)
  const viewerSnapshots = currentBenchmark && currentGame ? benchmarkSnapshots(currentBenchmark, currentGame) : []
  const viewerPosition = Math.min(viewer?.position ?? 0, Math.max(viewerSnapshots.length - 1, 0))
  const viewerGame = viewerSnapshots[viewerPosition]
  const previewGame = viewer && viewerGame && viewer.previewMove ? gameWithPreviewMove(viewerGame, viewer.previewMove) : undefined
  const previousViewerGame = viewerPosition > 0 ? viewerSnapshots[viewerPosition - 1] : undefined
  const viewerLastMove = viewer?.previewMove
    ? (() => {
        const [origin, destination] = viewer.previewMove.split('-')
        return origin && destination ? [toKey(origin), toKey(destination)] : undefined
      })()
    : viewerGame && previousViewerGame
    ? changedMove(previousViewerGame.board, viewerGame.board, viewerGame.side_to_move)
    : undefined
  const viewerComplete = Boolean(currentGame?.result || currentGame?.error)

  useEffect(() => {
    if (!active || !viewer || viewerComplete) return
    const timer = window.setInterval(() => void load(), 1_000)
    return () => window.clearInterval(timer)
  }, [active, viewer, viewerComplete])

  useEffect(() => {
    if (!active || !viewer || viewerComplete) return
    const latest = Math.max(viewerSnapshots.length - 1, 0)
    if (viewer.position !== latest) {
      setViewer({ ...viewer, position: latest, previewMove: null })
    }
  }, [active, viewer, viewerComplete, viewerSnapshots.length])

  return <section className="benchmark-view" hidden={!active} aria-label="模型基准">
    <form className="benchmark-form" onSubmit={(event) => { event.preventDefault(); void create() }}>
      <label>模型 A<select value={firstModel} onChange={(event) => setFirstModel(event.target.value)} disabled={!modelsLoaded || creating}>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label>
      <label>模型 B<select value={secondModel} onChange={(event) => setSecondModel(event.target.value)} disabled={!modelsLoaded || creating}>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label>
      <label>MCTS<select value={simulations} onChange={(event) => setSimulations(Number(event.target.value))} disabled={creating}>{simulationOptions.map((value) => <option key={value} value={value}>{value} 次</option>)}</select></label>
      <button type="submit" disabled={creating || !firstModel || firstModel === secondModel}>{creating ? '创建中...' : '开始基准'}</button>
    </form>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="benchmark-list">
      {benchmarks.length === 0 && <p className="benchmark-empty">尚无 benchmark</p>}
      {benchmarks.map((benchmark) => {
        const open = expanded === benchmark.id
        return <article className="benchmark-item" key={benchmark.id}>
          <button className="benchmark-summary" type="button" aria-expanded={open} onClick={() => setExpanded(open ? null : benchmark.id)}>
            <span className="benchmark-overview">
              <span className="benchmark-title"><span className={`benchmark-status ${benchmark.status}`}>{statusText(benchmark.status)}</span><strong>模型对局</strong></span>
              <span className="benchmark-models"><span title={modelName(models, benchmark.first_model)}>A · {modelName(models, benchmark.first_model)}</span><span title={modelName(models, benchmark.second_model)}>B · {modelName(models, benchmark.second_model)}</span></span>
              <span className="benchmark-config">{benchmark.mcts_simulations} sims · 开局库成对交换红黑</span>
            </span>
            <span className="benchmark-metrics"><span className="benchmark-score"><b>A</b> {benchmark.first_wins} 胜 <i>·</i> {benchmark.draws} 和 <i>·</i> <b>B</b> {benchmark.second_wins} 胜</span><span className="benchmark-progress">完成 {benchmark.games.length} / {benchmark.games_requested} 盘</span></span>
            <span className="benchmark-time"><span>开始 <time>{timeText(benchmark.started_at_ms)}</time></span><span>结束 <time>{timeText(benchmark.finished_at_ms)}</time></span></span>
          </button>
          <span className="benchmark-actions">{benchmark.status === 'running' && <button className="benchmark-cancel" type="button" onClick={() => void pause(benchmark.id)}>暂停</button>}{benchmark.status === 'paused' && <button className="benchmark-cancel" type="button" onClick={() => void resume(benchmark.id)}>继续</button>}</span>
          {open && <ol className="benchmark-games">
            {benchmark.games.map((game) => {
              const firstIsRed = game.number % 2 === 1
              const finished = Boolean(game.result || game.error)
              return <li key={game.number}><span className="benchmark-game-opening"><strong>第 {game.number} 盘</strong><span>开局 {game.opening_move}</span></span><span className="benchmark-game-result"><strong>{benchmarkGameResultText(game, firstIsRed)}</strong><span>{game.total_plies} ply</span>{game.repetition_cycle_plies && <span>循环 ply {game.repetition_cycle_plies[0]}-{game.repetition_cycle_plies[1]}</span>}</span><span className="benchmark-game-sides"><span title={modelName(models, firstIsRed ? benchmark.first_model : benchmark.second_model)}>红：{modelName(models, firstIsRed ? benchmark.first_model : benchmark.second_model)}</span><span title={modelName(models, firstIsRed ? benchmark.second_model : benchmark.first_model)}>黑：{modelName(models, firstIsRed ? benchmark.second_model : benchmark.first_model)}</span></span><span className="benchmark-game-time"><span>开始 {timeText(game.started_at_ms)}</span>{finished && <span>结束 {timeText(game.finished_at_ms)}</span>}<strong>耗时 {(game.elapsed_ms / 1000).toFixed(1)} 秒</strong></span><button type="button" className="benchmark-review" onClick={() => openViewer(benchmark.id, game.number)}>{finished ? '复盘' : '观看'}</button>{game.error && <em>{game.error}</em>}</li>
            })}
          </ol>}
        </article>
      })}
    </div>
    {viewer && currentGame && <div className="benchmark-dialog-backdrop" role="presentation" onClick={() => setViewer(null)}>
      <section className="benchmark-dialog" role="dialog" aria-modal="true" aria-label={viewerComplete ? 'benchmark 复盘' : 'benchmark 观看'} onClick={(event) => event.stopPropagation()}>
        <header className="benchmark-dialog-header">
          <div><strong>{viewerComplete ? '复盘' : '观看'} · 第 {currentGame.number} 盘 · {benchmarkGameSummary(currentGame)}</strong><span>{currentGame.opening_move}</span></div>
          <button className="icon-button" type="button" aria-label="关闭" title="关闭" onClick={() => setViewer(null)}>×</button>
        </header>
        <div className={viewerComplete ? 'benchmark-dialog-body complete' : 'benchmark-dialog-body watching'}>
          <XiangqiBoard active={active} game={previewGame ?? viewerGame} lastMove={viewerLastMove} readOnly onMove={() => undefined} />
          {viewerComplete && <MoveRecord
            snapshots={viewerSnapshots}
            error={currentGame.error ?? ''}
            currentIndex={viewerPosition}
            onNavigate={(position) => setViewer((current) => current ? { ...current, position, previewMove: null } : current)}
            onPreviewMove={(move) => setViewer((current) => current ? { ...current, previewMove: current.previewMove === move ? null : move } : current)}
            previewMove={viewer.previewMove}
            keyboardNavigationEnabled={active}
          />}
        </div>
      </section>
    </div>}
  </section>
}

const START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'

function benchmarkGameSummary(game: BenchmarkGame): string {
  const result = game.result ? resultText(game.result) : game.error ? '失败' : '进行中'
  return `${result} · ${game.total_plies} ply`
}

function benchmarkGameResultText(game: BenchmarkGame, firstIsRed: boolean): string {
  if (!game.result) return game.error ? '失败' : '进行中'
  if (game.result.startsWith('red_win')) return `红方（${firstIsRed ? 'A' : 'B'}）胜`
  if (game.result.startsWith('black_win')) return `黑方（${firstIsRed ? 'B' : 'A'}）胜`
  return resultText(game.result)
}

function benchmarkSnapshots(benchmark: Benchmark, game: BenchmarkGame): Game[] {
  const snapshots = game.snapshots ?? []
  if (snapshots.length === 0) {
    const fen = game.initial_fen || START_FEN
    return [snapshotToGame(benchmark, game, { fen, side_to_move: fen.split(' ')[1] === 'b' ? 'b' : 'w', turn: 1, rule60: 0, result: game.result, mcts_debug: null, policy_debug: null }, 0)]
  }
  const games = snapshots.map((snapshot, index) => snapshotToGame(benchmark, game, snapshot, index))
  if (game.initial_fen && games[0] && game.initial_fen.split(' ')[0] !== snapshots[0].fen.split(' ')[0]) {
    games.unshift(snapshotToGame(benchmark, game, { fen: game.initial_fen, side_to_move: game.initial_fen.split(' ')[1] === 'b' ? 'b' : 'w', turn: 1, rule60: 0, result: null, mcts_debug: null, policy_debug: null }, -1))
  }
  return games
}

function snapshotToGame(benchmark: Benchmark, game: BenchmarkGame, snapshot: CompactArchiveSnapshot, index: number): Game {
  return {
    game_id: `${benchmark.id}-${game.number}-${index}`,
    mode: 'model-model',
    human_side: null,
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
    policy_debug: snapshot.policy_debug ?? null,
  }
}