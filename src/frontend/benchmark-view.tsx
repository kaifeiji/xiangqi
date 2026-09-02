import { useEffect, useEffectEvent, useState } from 'react'
import { request } from './api'
import { resultText } from './game-utils'
import type { Benchmark, ModelOption } from './types'

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
  return { running: '运行中', completed: '已完成', cancelled: '已取消', failed: '失败' }[status]
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

  const cancel = useEffectEvent(async (id: string) => {
    try {
      await request<void>(`/api/benchmarks/${id}`, { method: 'DELETE' })
      await load()
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : String(cancelError))
    }
  })

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
          <span className="benchmark-actions">{benchmark.status === 'running' && <button className="benchmark-cancel" type="button" onClick={() => void cancel(benchmark.id)}>取消</button>}</span>
          {open && <ol className="benchmark-games">
            {benchmark.games.map((game) => {
              const firstIsRed = game.number % 2 === 1
              return <li key={game.number}><span className="benchmark-game-opening"><strong>第 {game.number} 盘</strong><span>开局 {game.opening_move}</span></span><span className="benchmark-game-result"><strong>{game.result ? resultText(game.result) : game.error ? '失败' : '已取消'}</strong><span>{game.total_plies} ply</span>{game.repetition_cycle_plies && <span>循环 ply {game.repetition_cycle_plies[0]}-{game.repetition_cycle_plies[1]}</span>}</span><span className="benchmark-game-sides"><span title={modelName(models, firstIsRed ? benchmark.first_model : benchmark.second_model)}>红：{modelName(models, firstIsRed ? benchmark.first_model : benchmark.second_model)}</span><span title={modelName(models, firstIsRed ? benchmark.second_model : benchmark.first_model)}>黑：{modelName(models, firstIsRed ? benchmark.second_model : benchmark.first_model)}</span></span><span className="benchmark-game-time"><span>开始 {timeText(game.started_at_ms)}</span><span>结束 {timeText(game.finished_at_ms)}</span><strong>耗时 {(game.elapsed_ms / 1000).toFixed(1)} 秒</strong></span>{game.error && <em>{game.error}</em>}</li>
            })}
          </ol>}
        </article>
      })}
    </div>
  </section>
}