import { useEffect, useEffectEvent, useState } from 'react'
import { request } from './api'
import { benchmarkSnapshots } from './benchmark-view'
import { changedMove, gameWithPreviewMove, toKey } from './game-utils'
import { MoveRecord } from './move-record'
import { XiangqiBoard } from './xiangqi-board'
import type { BenchmarkGame, BenchmarkGameSummary, BenchmarkSummary, ModelOption } from './types'

interface TournamentMatch { benchmarkId: string; firstModel: string; secondModel: string }
interface TournamentResponse { id: string; models: string[]; mcts_simulations: number; games_per_match: number; benchmark_ids: string[]; status: string; created_at_ms: number; finished_at_ms: number | null; ratings: Record<string, number> }
interface TournamentViewProps { active: boolean; models: ModelOption[]; modelsLoaded: boolean }

function modelName(models: ModelOption[], id: string): string { return models.find((model) => model.id === id)?.name ?? id }
export function TournamentView({ active, models, modelsLoaded }: TournamentViewProps): React.JSX.Element {
  const [selected, setSelected] = useState<string[]>([])
  const [simulations, setSimulations] = useState(1000)
  const [tournaments, setTournaments] = useState<TournamentResponse[]>([])
  const [tournamentId, setTournamentId] = useState('')
  const [matches, setMatches] = useState<TournamentMatch[]>([])
  const [summaries, setSummaries] = useState<Record<string, BenchmarkSummary>>({})
  const [games, setGames] = useState<Record<string, BenchmarkGameSummary[]>>({})
  const [expandedMatch, setExpandedMatch] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const [error, setError] = useState('')
  const [viewer, setViewer] = useState<{ match: TournamentMatch; game: BenchmarkGame; position: number; previewMove: string | null } | null>(null)

  useEffect(() => {
    if (modelsLoaded && tournaments.length === 0 && selected.length === 0) {
      setSelected(models.filter((model) => model.id !== 'pikafish').map((model) => model.id))
    }
  }, [modelsLoaded, models, selected.length, tournaments.length])

  const loadTournaments = useEffectEvent(async () => {
    try {
      const loaded = await request<TournamentResponse[]>('/api/tournaments')
      setTournaments(loaded)
      if (!tournamentId && loaded[0]) setTournamentId(loaded[0].id)
    } catch (loadError) { setError(loadError instanceof Error ? loadError.message : String(loadError)) }
  })

  const selectTournament = useEffectEvent(async (id: string) => {
    const tournament = tournaments.find((entry) => entry.id === id)
    if (!tournament) return
    try {
      const loaded = await request<BenchmarkSummary[]>('/api/benchmarks')
      const selectedSummaries = loaded.filter((summary) => tournament.benchmark_ids.includes(summary.id))
      setTournamentId(id); setSelected(tournament.models)
      setMatches(selectedSummaries.map((summary) => ({ benchmarkId: summary.id, firstModel: summary.first_model, secondModel: summary.second_model })))
      setSummaries(Object.fromEntries(selectedSummaries.map((summary) => [summary.id, summary])))
    } catch (loadError) { setError(loadError instanceof Error ? loadError.message : String(loadError)) }
  })

  useEffect(() => {
    if (active && tournamentId && tournaments.length > 0) void selectTournament(tournamentId)
  }, [active, tournamentId, tournaments])

  const refresh = useEffectEvent(async (force = false) => {
    const running = matches.filter((match) => summaries[match.benchmarkId]?.status === 'running')
    if (!force && running.length === 0) return
    try {
      const loaded = await request<BenchmarkSummary[]>(`/api/tournaments/${tournamentId}/benchmarks`)
      setSummaries((current) => ({ ...current, ...Object.fromEntries(loaded.map((summary) => [summary.id, summary])) }))
    } catch (loadError) { setError(loadError instanceof Error ? loadError.message : String(loadError)) }
  })

  const loadGames = useEffectEvent(async (match: TournamentMatch) => {
    try {
      const loaded = await request<BenchmarkGameSummary[]>(`/api/benchmarks/${match.benchmarkId}/games`)
      setGames((current) => ({ ...current, [match.benchmarkId]: loaded }))
    }
    catch (loadError) { setError(loadError instanceof Error ? loadError.message : String(loadError)) }
  })

  const openGame = useEffectEvent(async (match: TournamentMatch, number: number) => {
    try {
      const game = await request<BenchmarkGame>(`/api/benchmarks/${match.benchmarkId}/games/${number}`)
      setViewer({ match, game, position: Math.max(game.snapshots.length - 1, 0), previewMove: null })
    } catch (loadError) { setError(loadError instanceof Error ? loadError.message : String(loadError)) }
  })

  useEffect(() => {
    if (!active) return
    void loadTournaments()
  }, [active])

  useEffect(() => {
    if (!active || !tournamentId || matches.length === 0) return
    void refresh()
    const timer = window.setInterval(() => {
      void refresh()
      if (expandedMatch) {
        const match = matches.find((entry) => entry.benchmarkId === expandedMatch)
        if (match && summaries[match.benchmarkId]?.status === 'running') void loadGames(match)
      }
    }, 5_000)
    return () => window.clearInterval(timer)
  }, [active, expandedMatch, matches, summaries, tournamentId])

  const start = useEffectEvent(async () => {
    if (selected.length < 2) return
    setCreating(true); setError('')
    try {
      const tournament = await request<TournamentResponse>('/api/tournaments', { method: 'POST', body: JSON.stringify({ models: selected, mcts_simulations: simulations }) })
      const created = await Promise.all(tournament.benchmark_ids.map((id) => request<BenchmarkSummary>(`/api/benchmarks/${id}`)))
      setMatches(created.map((summary) => ({ benchmarkId: summary.id, firstModel: summary.first_model, secondModel: summary.second_model })))
      setSummaries(Object.fromEntries(created.map((summary) => [summary.id, summary])))
      setTournaments((current) => [tournament, ...current.filter((entry) => entry.id !== tournament.id)])
      setTournamentId(tournament.id)
      setConfigOpen(false)
    } catch (startError) { setError(startError instanceof Error ? startError.message : String(startError)) }
    finally { setCreating(false) }
  })

  const pauseAll = useEffectEvent(async () => {
    if (!tournamentId) return
    try {
      await request<void>(`/api/tournaments/${tournamentId}`, { method: 'DELETE' })
      await refresh(true)
    } catch (pauseError) { setError(pauseError instanceof Error ? pauseError.message : String(pauseError)) }
  })
  const resumeAll = useEffectEvent(async () => {
    if (!tournamentId) return
    try {
      await request<void>(`/api/tournaments/${tournamentId}`, { method: 'POST', body: '{}' })
      await refresh(true)
    } catch (resumeError) { setError(resumeError instanceof Error ? resumeError.message : String(resumeError)) }
  })
  const toggleModel = (modelId: string) => {
    setSelected((current) => current.includes(modelId) ? current.filter((id) => id !== modelId) : [...current, modelId])
  }
  const hasRunning = matches.some((match) => {
    const status = summaries[match.benchmarkId]?.status
    return status === 'running' || status === 'queued'
  })
  const hasPaused = matches.some((match) => {
    const summary = summaries[match.benchmarkId]
    return Boolean(summary && summary.games_completed < summary.games_requested && summary.status !== 'running' && summary.status !== 'queued')
  })
  const currentTournament = tournaments.find((tournament) => tournament.id === tournamentId)
  const standings = selected.map((model) => ({ model, rating: currentTournament?.ratings[model] ?? 1500, wins: 0, draws: 0, losses: 0, games: 0 }))
  for (const match of matches) {
    const summary = summaries[match.benchmarkId]
    if (!summary || summary.status !== 'completed') continue
    const first = standings.find((entry) => entry.model === match.firstModel)
    const second = standings.find((entry) => entry.model === match.secondModel)
    if (!first || !second) continue
    const total = summary.games_completed || summary.games_requested
    first.wins += summary.first_wins; first.draws += summary.draws; first.losses += summary.second_wins; first.games += total
    second.wins += summary.second_wins; second.draws += summary.draws; second.losses += summary.first_wins; second.games += total
  }
  standings.sort((a, b) => b.rating - a.rating)

  const currentSummary = viewer ? summaries[viewer.match.benchmarkId] : undefined
  const snapshots = currentSummary && viewer ? benchmarkSnapshots(currentSummary, viewer.game) : []
  const position = Math.min(viewer?.position ?? 0, Math.max(snapshots.length - 1, 0))
  const currentGame = snapshots[position]
  const previousGame = position > 0 ? snapshots[position - 1] : undefined
  const lastMove = viewer?.previewMove ? (() => { const [from, to] = viewer.previewMove.split('-'); return from && to ? [toKey(from), toKey(to)] : undefined })() : currentGame && previousGame ? changedMove(previousGame.board, currentGame.board, currentGame.side_to_move) : undefined
  const preview = viewer?.previewMove && currentGame ? gameWithPreviewMove(currentGame, viewer.previewMove) : undefined
  const complete = Boolean(viewer?.game.result || viewer?.game.error)
  const completedMatches = matches.filter((match) => {
    const summary = summaries[match.benchmarkId]
    return Boolean(summary && summary.games_completed >= summary.games_requested)
  }).length

  return <section className="tournament-view" hidden={!active} aria-label="ELO 积分赛">
    <header className="view-header tournament-header"><div><span className="eyebrow">MODEL LEAGUE</span><h2>ELO 积分循环赛</h2><p>每对模型进行 2 盘红黑交换对局。</p></div><div className="tournament-actions">{tournaments.length > 0 && <label><span className="visually-hidden">选择比赛</span><select aria-label="选择比赛" value={tournamentId} onChange={(event) => void selectTournament(event.target.value)} disabled={creating}>{tournaments.map((tournament, index) => <option key={tournament.id} value={tournament.id}>第 {tournaments.length - index} 届</option>)}</select></label>}<button type="button" onClick={() => setConfigOpen(true)} disabled={!modelsLoaded || creating || hasRunning}>{creating ? '创建中...' : '开始新赛事'}</button>{hasRunning ? <button className="secondary" type="button" onClick={() => void pauseAll()} disabled={creating}>暂停赛事</button> : hasPaused && <button className="secondary" type="button" onClick={() => void resumeAll()} disabled={creating}>继续赛事</button>}</div></header>
    <div className="tournament-select"><div className="selection-hint"><div><small>比赛</small><strong>{currentTournament ? `第 ${tournaments.length - tournaments.indexOf(currentTournament)} 届` : '—'}</strong></div><div><small>MCTS</small><strong>{currentTournament?.mcts_simulations ?? simulations}</strong></div><div><small>已选择模型</small><strong>{selected.length}</strong></div><div><small>比赛进度</small><strong>{completedMatches} / {matches.length}</strong></div></div></div>
    {configOpen && <div className="tournament-config-backdrop" role="presentation" onClick={() => setConfigOpen(false)}><section className="tournament-config-dialog" role="dialog" aria-modal="true" aria-label="赛事配置" onClick={(event) => event.stopPropagation()}><header><div><span className="eyebrow">NEW EVENT</span><h3>赛事配置</h3></div><button type="button" className="icon-button" aria-label="关闭" onClick={() => setConfigOpen(false)}>×</button></header><fieldset disabled={!modelsLoaded || creating}><legend>选择模型</legend><div className="model-tags">{models.map((model) => <button className={`model-tag ${selected.includes(model.id) ? 'selected' : ''}`} key={model.id} type="button" aria-pressed={selected.includes(model.id)} onClick={() => toggleModel(model.id)}>{model.name}</button>)}</div></fieldset><label className="tournament-config-mcts">MCTS<select value={simulations} onChange={(event) => setSimulations(Number(event.target.value))} disabled={creating}>{[0, 1000, 5000, 10000].map((value) => <option key={value} value={value}>{value} sims</option>)}</select></label><div className="tournament-config-actions"><button type="button" className="secondary" onClick={() => setConfigOpen(false)}>取消</button><button type="button" onClick={() => void start()} disabled={creating || selected.length < 2}>{creating ? '创建中...' : '开始赛事'}</button></div></section></div>}
    {error && <p className="error" role="alert">{error}</p>}
    <div className="tournament-layout"><section><div className="section-heading"><h3>对局</h3><span>{matches.filter((match) => { const summary = summaries[match.benchmarkId]; return Boolean(summary && summary.games_completed >= summary.games_requested) }).length} / {matches.length} 场完成</span></div><div className="tournament-matches">{matches.map((match) => { const summary = summaries[match.benchmarkId]; const matchGames = games[match.benchmarkId] ?? []; const finished = Boolean(summary && summary.games_completed >= summary.games_requested); const paused = summary?.status === 'paused'; const queued = summary?.status === 'queued'; const status = !summary ? 'loading' : finished ? 'completed' : paused ? 'paused' : queued ? 'queued' : 'running'; return <article className="tournament-match" key={match.benchmarkId} onClick={() => { setExpandedMatch(expandedMatch === match.benchmarkId ? null : match.benchmarkId); if (matchGames.length === 0) void loadGames(match) }}><div className={`match-status ${status}`}>{!summary ? '加载中' : finished ? '已完成' : paused ? '已暂停' : queued ? '排队中' : '进行中'}</div><h4>{modelName(models, match.firstModel)} <b>VS</b> {modelName(models, match.secondModel)}</h4><div className="match-score">{summary ? `${summary.first_wins} - ${summary.draws} - ${summary.second_wins}` : '等待数据'}</div><div className="match-games">{[1, 2].map((number) => { const game = matchGames.find((entry) => entry.number === number); const available = Boolean(game) || number <= (summary?.games_completed ?? 0); const action = game?.result || game?.error || (finished && available) ? '复盘' : '观看'; return <button type="button" key={number} disabled={!available} title={available ? `${action}第 ${number} 盘` : `第 ${number} 盘尚未开始`} aria-label={available ? `${action}第 ${number} 盘` : `第 ${number} 盘尚未开始`} onClick={(event) => { event.stopPropagation(); void openGame(match, number) }}>{number}</button> })}</div></article> })}</div></section><aside className="tournament-standings"><div className="section-heading"><h3>ELO 排名</h3><span>已完成赛事</span></div>{standings.map((entry, index) => <div className="standing-row" key={entry.model}><strong>{index + 1}</strong><span title={modelName(models, entry.model)}>{modelName(models, entry.model)}<small>{entry.wins}-{entry.draws}-{entry.losses} · {entry.games} 盘</small></span><b>{entry.rating.toFixed(0)}</b></div>)}</aside></div>
    {viewer && currentGame && <div className="benchmark-dialog-backdrop" role="presentation" onClick={() => setViewer(null)}><section className="benchmark-dialog" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}><header className="benchmark-dialog-header"><div><strong>{complete ? '复盘' : '观看'} · 第 {viewer.game.number} 盘</strong><span>{modelName(models, viewer.match.firstModel)} VS {modelName(models, viewer.match.secondModel)} · {viewer.game.opening_move}</span></div><button className="icon-button" type="button" onClick={() => setViewer(null)} aria-label="关闭">×</button></header><div className={complete ? 'benchmark-dialog-body complete' : 'benchmark-dialog-body watching'}><XiangqiBoard active={active} game={preview ?? currentGame} lastMove={lastMove} readOnly onMove={() => undefined} />{complete && <MoveRecord snapshots={snapshots} error={viewer.game.error ?? ''} currentIndex={position} onNavigate={(next) => setViewer((value) => value ? { ...value, position: next, previewMove: null } : value)} onPreviewMove={(move) => setViewer((value) => value ? { ...value, previewMove: value.previewMove === move ? null : move } : value)} previewMove={viewer.previewMove} keyboardNavigationEnabled={active} />}</div></section></div>}
  </section>
}
