import { useEffect, useEffectEvent, useRef, useState } from 'react'
import { boardToFen, changedMove, toIccs } from './game-utils'
import type { CompactArchiveSnapshot, Game, GameArchive, GameMode, Side } from './types'

interface MoveRecordProps {
  snapshots: Game[]
  status: string
  error: string
  archiveMode?: GameMode
  archiveHumanSide?: Side | null
  currentIndex?: number
  onNavigate?: (index: number) => void
  onPreviewMove?: (move: string) => void
  previewMove?: string | null
  keyboardNavigationEnabled?: boolean
}

export function MoveRecord({ snapshots, status, error, archiveMode, archiveHumanSide, currentIndex, onNavigate, onPreviewMove, previewMove, keyboardNavigationEnabled = false }: MoveRecordProps): React.JSX.Element {
  const [activeTab, setActiveTab] = useState<'record' | 'analysis'>('record')
  const moves = snapshots.map((snapshot, index) => {
    if (index === 0) return { id: `${snapshot.game_id}-${snapshot.turn}`, number: 0, snapshotIndex: 0, text: '初始局面' }
    const previous = snapshots[index - 1]
    const move = changedMove(previous.board, snapshot.board, snapshot.side_to_move)
    if (!move) return undefined
    const side = snapshot.side_to_move === 'b' ? '红方' : '黑方'
    return { id: `${snapshot.game_id}-${snapshot.turn}`, number: index, snapshotIndex: index, text: `${side} ${toIccs(move[0])}-${toIccs(move[1])}` }
  }).filter((move): move is { id: string; number: number; snapshotIndex: number; text: string } => move !== undefined)

  const activeMoveRef = useRef<HTMLLIElement | null>(null)
  useEffect(() => {
    activeMoveRef.current?.scrollIntoView({ block: 'nearest' })
  }, [currentIndex])

  const replayEnabled = typeof currentIndex === 'number' && typeof onNavigate === 'function'
  const currentPly = currentIndex ?? Math.max(snapshots.length - 1, 0)
  const maxPly = Math.max(snapshots.length - 1, 0)
  const canSaveArchive = Boolean(archiveMode) && snapshots.length > 0
  const currentSnapshot = snapshots[currentPly]
  const mctsDebug = currentSnapshot?.mcts_debug ?? null
  const policyDebug = currentSnapshot?.policy_debug ?? null
  const fallbackAnalysisSide = snapshots[currentPly - 1]?.side_to_move ?? currentSnapshot?.side_to_move
  const analysisSideCode = mctsDebug?.searched_side ?? policyDebug?.searched_side ?? fallbackAnalysisSide
  const analysisSide = analysisSideCode === 'w' ? '红方' : '黑方'
  const value = mctsDebug?.root_network_value ?? policyDebug?.network_value ?? null
  const advantage = value === null ? '暂无评估' : Math.abs(value) < 0.05 ? '局面均势' : value > 0 ? `${analysisSide}优势` : `${analysisSide === '红方' ? '黑方' : '红方'}优势`

  const navigateWithKeyboard = useEffectEvent((event: KeyboardEvent): void => {
    if (!keyboardNavigationEnabled || !replayEnabled || !onNavigate) return
    const nextIndex = event.key === 'ArrowLeft' ? currentPly - 1 : event.key === 'ArrowRight' ? currentPly + 1 : currentPly
    if (nextIndex === currentPly || nextIndex < 0 || nextIndex > maxPly) return
    event.preventDefault()
    onNavigate(nextIndex)
  })

  useEffect(() => {
    window.addEventListener('keydown', navigateWithKeyboard)
    return () => window.removeEventListener('keydown', navigateWithKeyboard)
  }, [])

  const saveArchive = (): void => {
    if (!archiveMode || snapshots.length === 0) return
    const archive: GameArchive = {
      savedAt: new Date().toISOString(), mode: archiveMode, humanSide: archiveHumanSide ?? null,
      initial_fen: `${boardToFen(snapshots[0].board)} ${snapshots[0].side_to_move} - - 0 1`,
      snapshots: snapshots.map((snapshot): CompactArchiveSnapshot => ({ fen: boardToFen(snapshot.board), side_to_move: snapshot.side_to_move, turn: snapshot.turn, rule60: snapshot.rule60, result: snapshot.result, mcts_debug: snapshot.mcts_debug ?? null, policy_debug: snapshot.policy_debug ?? null })),
    }
    const blob = new Blob([JSON.stringify(archive, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'xiangqi-save.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="move-record" aria-label="行棋与着法">
      <div className="record-toolbar">
        <div className="record-tabs" role="tablist" aria-label="行棋和着法">
          <button type="button" role="tab" aria-selected={activeTab === 'record'} onClick={() => setActiveTab('record')}>行棋记录</button>
          <button type="button" role="tab" aria-selected={activeTab === 'analysis'} onClick={() => setActiveTab('analysis')}>着法分析</button>
        </div>
        {canSaveArchive && <button className="icon-button" type="button" aria-label="保存存档" title="保存存档" onClick={saveArchive}>&#8595;</button>}
      </div>
      <div className="record-pane" role="tabpanel" aria-label="行棋" hidden={activeTab !== 'record'}>
        <header className="record-header">
          <div><p className="status" role="status">{status}</p></div>
        </header>
        {error && <p className="error" role="alert">{error}</p>}
        {replayEnabled && <div className="record-controls">
          <button type="button" onClick={() => onNavigate(currentPly - 1)} disabled={currentPly <= 0}>上一步</button>
          <span>{currentPly} / {maxPly}</span>
          <button type="button" onClick={() => onNavigate(currentPly + 1)} disabled={currentPly >= maxPly}>下一步</button>
        </div>}
        <ol className="move-list">
          {moves.map((move) => {
            const isActive = currentPly === move.snapshotIndex
            return <li key={move.id} ref={isActive ? activeMoveRef : null} className={isActive ? 'active' : undefined}>
              <button type="button" className="move-entry" onClick={() => onNavigate?.(move.snapshotIndex)} disabled={!onNavigate} aria-current={isActive ? 'true' : undefined}>
                <span className="move-number">{move.number}.</span>{move.text}
              </button>
            </li>
          })}
        </ol>
      </div>
      <div className="analysis-panel" role="tabpanel" aria-label="着法" hidden={activeTab !== 'analysis'}>
        {!mctsDebug && !policyDebug ? <p className="analysis-empty">该局面没有模型分析数据</p> : <>
          <div className="analysis-summary">
            <div><span className="analysis-label">分析方</span><strong>{analysisSide}</strong></div>
            <div><span className="analysis-label">局面判断</span><strong>{advantage}</strong></div>
            <div><span className="analysis-label">Root value</span><strong className={value !== null && value >= 0 ? 'value-positive' : 'value-negative'}>{value === null ? '--' : value.toFixed(3)}</strong></div>
          </div>
          {mctsDebug && <dl className="analysis-stats">
            <div><dt>模拟次数</dt><dd>{mctsDebug.simulations}</dd></div>
            <div><dt>平均叶深</dt><dd>{mctsDebug.average_leaf_depth.toFixed(1)}</dd></div>
            <div><dt>最大叶深</dt><dd>{mctsDebug.max_leaf_depth}</dd></div>
          </dl>}
          <div className="candidate-header"><h3>候选着法</h3><span>{mctsDebug ? '按访问次数排序' : '按 Policy 概率排序'}</span></div>
          <div className="candidate-list">
            {mctsDebug && mctsDebug.root_children.slice().sort((left, right) => right.visits - left.visits).map((candidate) => {
              const content = <>
                <strong>{candidate.move}</strong><span>访问 {candidate.visits}</span><span>Q {candidate.q.toFixed(3)}</span><span>Policy {(candidate.prior * 100).toFixed(2)}%</span>
              </>
              return onPreviewMove ? (
                <button type="button" className={`candidate-row${previewMove === candidate.move ? ' selected' : ''}`} key={candidate.move} aria-pressed={previewMove === candidate.move} onClick={() => onPreviewMove(candidate.move)}>
                  {content}
                </button>
              ) : <div className="candidate-row" key={candidate.move}>{content}</div>
            })}
            {policyDebug && policyDebug.candidates.slice().sort((left, right) => right.probability - left.probability).map((candidate) => {
              const content = <><strong>{candidate.move}</strong><span>Policy {(candidate.probability * 100).toFixed(2)}%</span></>
              return onPreviewMove ? (
                <button type="button" className={`candidate-row${previewMove === candidate.move ? ' selected' : ''}`} key={candidate.move} aria-pressed={previewMove === candidate.move} onClick={() => onPreviewMove(candidate.move)}>
                  {content}
                </button>
              ) : <div className="candidate-row" key={candidate.move}>{content}</div>
            })}
          </div>
        </>}
      </div>
    </section>
  )
}
