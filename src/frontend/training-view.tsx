import { useEffect, useMemo, useState } from 'react'
import { request } from './api'
import type { TrainingCheckpoint } from './types'

type NumericPoint = { x: number; y: number }
type BestMarker = { kind: 'best' | 'best-policy' | 'best-value'; point: NumericPoint }

const preferredFields = [
  'validation_j_select',
  'validation.loss',
  'validation.joint_loss',
  'training_loss',
  'training_policy_loss',
  'training_value_loss',
  'validation.policy_loss',
  'validation.policy_cp_loss',
  'validation.cp_policy_kl',
  'validation.value_loss',
  'validation.value_cp_loss',
  'validation.value_cp_mae_le_300',
  'validation.value_cp_mae_all',
  'validation.value_sign_accuracy',
  'validation.value_sign_accuracy_le_100',
  'validation.value_sign_accuracy_gt_300',
  'validation.start_top1',
  'validation.start_top5',
  'validation.end_top1',
  'validation.end_top5',
  'validation.complete_top1',
  'validation.complete_top5',
  'validation.complete_top10',
  'start_loss',
  'end_loss',
  'start_top1',
  'end_top1',
  'learning_rate',
  'value_learning_rate',
]

const hiddenFields = new Set([
  'epoch',
  'global_step',
  'epoch_progress',
  'epoch_elapsed_seconds',
  'epoch_eta_seconds',
  'epoch_seconds',
  'no_improve_epochs',
  'gradient_norm_pre_clip',
  'samples',
  'policy_valid',
  'value_valid',
  'value_mate',
  'mate_policy',
  'policy_valid_count',
  'value_valid_count',
  'mate_policy_count',
  'value_mate_count',
  'policy_mate_samples',
  'value_cp_samples',
  'value_mate_samples',
  'cp_policy_samples',
  'parameters',
  'validation.samples',
  'validation.policy_valid',
  'validation.value_valid',
  'validation.mate_policy',
  'validation.policy_mate',
  'validation.value_mate',
  'validation.policy_valid_count',
  'validation.value_valid_count',
  'validation.mate_policy_count',
  'validation.value_mate_count',
  'validation.policy_mate_samples',
  'validation.value_cp_samples',
  'validation.value_mate_samples',
  'validation.cp_policy_samples',
])

const hiddenParameters = new Set([
  'resume',
  'num_workers',
  'workers',
  'patience',
  'epochs',
  'max_grad_norm',
  'amp',
  'prefetch_factor',
  'warmup_ratio',
  'block_size',
  'policy_weight',
  'temperature',
  'weight_decay',
])

const parameterOrder = [
  'channels',
  'blocks',
  'value_head',
  'use_se',
  'se_reduction',
  'batch_size',
  'micro_batch_size',
  'global_batch_size',
  'accumulation_steps',
  'learning_rate',
  'value_learning_rate',
  'min_learning_rate',
  'weight_decay',
  'temperature',
  'value_scale',
  'policy_weight',
  'value_weight',
  'warmup_ratio',
  'warmup_steps',
  'block_size',
  'mirror',
  'current_view',
  'seed',
]

function flatten(value: unknown, prefix = ''): Record<string, number> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  return Object.entries(value).reduce<Record<string, number>>((result, [key, child]) => {
    const name = prefix ? `${prefix}.${key}` : key
    if (typeof child === 'number' && Number.isFinite(child)) result[name] = child
    else Object.assign(result, flatten(child, name))
    return result
  }, {})
}

function recordsFor(checkpoint: TrainingCheckpoint): Array<{ x: number; values: Record<string, number> }> {
  const records = checkpoint.metrics.length > 0 ? checkpoint.metrics : checkpoint.progress
  return records.map((record, index) => {
    const values = flatten(record)
    const x = values.epoch ?? values.global_step ?? index + 1
    return { x, values }
  })
}

function pointsFor(records: Array<{ x: number; values: Record<string, number> }>, field: string): NumericPoint[] {
  return records.flatMap(({ x, values }) => (field in values ? [{ x, y: values[field] }] : []))
}

function markerKindFor(field: string): BestMarker['kind'] | null {
  if (['validation_j_select', 'validation.loss', 'validation.joint_loss'].includes(field)) return 'best'
  if (field.includes('policy')) return 'best-policy'
  if (field.includes('value')) return 'best-value'
  return null
}

function layoutMarkers(
  markers: BestMarker[],
  coordinate: (point: NumericPoint) => string,
): Array<BestMarker & { x: number; y: number }> {
  const placed: Array<BestMarker & { x: number; y: number }> = []
  for (const marker of markers) {
    const [x, originalY] = coordinate(marker.point).split(',').map(Number)
    let y = originalY
    while (placed.some((other) => Math.abs(other.x - x) < 6 && Math.abs(other.y - y) < 5)) y += 5
    placed.push({ ...marker, x, y })
  }
  return placed
}

function bestEpochFor(
  records: Array<{ x: number; values: Record<string, number> }>,
  fields: string[],
): number | undefined {
  const candidates = records.flatMap((record) => fields.flatMap((field) => (
    field in record.values ? [{ x: record.x, value: record.values[field] }] : []
  )))
  return candidates.reduce<{ x: number; value: number } | undefined>(
    (best, candidate) => !best || candidate.value < best.value ? candidate : best,
    undefined,
  )?.x
}

function labelFor(field: string): string {
  const labels: Record<string, string> = {
    validation_j_select: 'Validation J-Select',
    'validation.loss': 'Validation Loss',
    'validation.joint_loss': 'Validation Joint Loss',
    'validation.policy_loss': 'Validation Policy Loss',
    'validation.value_loss': 'Validation Value Loss',
    'validation.cp_policy_kl': 'Validation CP Policy KL',
    'validation.value_cp_mae_le_300': 'Validation Value CP MAE <= 300',
    'validation.value_sign_accuracy': 'Validation Value Sign Accuracy',
    'validation.start_top1': 'Validation Start Top-1',
    'validation.start_top5': 'Validation Start Top-5',
    'validation.end_top1': 'Validation End Top-1',
    'validation.end_top5': 'Validation End Top-5',
    'validation.complete_top1': 'Validation Complete Top-1',
    'validation.complete_top5': 'Validation Complete Top-5',
    'validation.complete_top10': 'Validation Complete Top-10',
    training_loss: 'Training Loss',
    start_loss: 'Start Loss',
    end_loss: 'End Loss',
    start_top1: 'Start Top-1',
    end_top1: 'End Top-1',
    learning_rate: 'Learning Rate',
    value_learning_rate: 'Value Learning Rate',
  }
  if (labels[field]) return labels[field]
  const words = field.replace(/^validation\./, 'Validation ').split('_')
  return words.map((word) => {
    const upper = word.toUpperCase()
    if (['CP', 'KL', 'MAE', 'MSE', 'LR'].includes(upper)) return upper
    const topMatch = /^top(\d+)$/.exec(word.toLowerCase())
    if (topMatch) return `Top-${topMatch[1]}`
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
  }).join(' ')
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '进行中'
  const roundedSeconds = Math.round(seconds)
  const hours = Math.floor(roundedSeconds / 3600)
  const minutes = Math.floor((roundedSeconds % 3600) / 60)
  const remainder = roundedSeconds % 60
  if (hours > 0) return `${hours}小时${minutes}分`
  if (minutes > 0) return `${minutes}分${remainder}秒`
  return `${remainder}秒`
}

function formatParameter(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'enabled' : 'disabled'
  if (typeof value === 'number') {
    if (value !== 0 && Math.abs(value) < 0.01) return formatScientific(value)
    return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  }
  return String(value)
}

function formatScientific(value: number): string {
  const [mantissa, exponent] = value.toExponential(3).split('e')
  return `${mantissa.replace(/(?:\.0+|(?<=\d)0+)$/, '').replace(/\.$/, '')}e${exponent}`
}

function formatAxisValue(value: number): string {
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) return formatScientific(value)
  return value.toPrecision(4)
}

function formatStepAxisValue(value: number): string {
  return String(Math.round(value))
}

function MetricChart({ field, points, bestEpochs }: { field: string; points: NumericPoint[]; bestEpochs: Partial<Record<BestMarker['kind'], number>> }): React.JSX.Element {
  const width = 360
  const height = 150
  const padding = { top: 16, right: 12, bottom: 24, left: 42 }
  const ys = points.map((point) => point.y)
  const min = Math.min(...ys)
  const max = Math.max(...ys)
  const span = max - min || Math.max(Math.abs(max) * 0.08, 1)
  const xMin = points[0].x
  const xMax = points.at(-1)?.x ?? xMin
  const xSpan = xMax - xMin || 1
  const yTicks = Array.from({ length: 5 }, (_, index) => min + (span * index) / 4).reverse()
  const xTickCount = Math.min(5, points.length)
  const xTicks = Array.from({ length: xTickCount }, (_, index) => {
    const pointIndex = Math.round((index * (points.length - 1)) / Math.max(xTickCount - 1, 1))
    return points[pointIndex].x
  }).filter((tick, index, ticks) => ticks.indexOf(tick) === index)
  const markerKinds: BestMarker['kind'][] = ['best', 'best-policy', 'best-value']
  const markers = markerKinds.flatMap((kind) => {
    const x = bestEpochs[kind]
    const point = x === undefined ? undefined : points.find((candidate) => candidate.x === x)
    return point ? [{ kind, point }] : []
  })
  const coordinate = (point: NumericPoint) => {
    const x = padding.left + ((point.x - xMin) / xSpan) * (width - padding.left - padding.right)
    const y = padding.top + (1 - (point.y - min) / span) * (height - padding.top - padding.bottom)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }
  const line = points.map(coordinate).join(' ')
  const markerCoordinates = layoutMarkers(markers, coordinate)

  return (
    <article className="training-chart">
      <div className="training-chart-heading">
        <strong>{labelFor(field)}</strong>
        <span>{points.at(-1)?.y.toPrecision(5)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${labelFor(field)} curve`}>
        <line x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} className="chart-axis" />
        {yTicks.map((tick) => {
          const y = padding.top + (1 - (tick - min) / span) * (height - padding.top - padding.bottom)
          return <g key={`y-${tick}`}><line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="chart-grid" /><text x={padding.left - 6} y={y + 3} textAnchor="end" className="chart-label">{formatAxisValue(tick)}</text></g>
        })}
        <polyline points={line} className="chart-line" />
        {markerCoordinates.map((marker) => <circle key={marker.kind} cx={marker.x} cy={marker.y} r="4" className={`chart-best-dot ${marker.kind}`} />)}
        {points.length === 1 && <circle cx={coordinate(points[0]).split(',')[0]} cy={coordinate(points[0]).split(',')[1]} r="3" className="chart-dot" />}
        {xTicks.map((tick, index) => {
          const x = padding.left + ((tick - xMin) / xSpan) * (width - padding.left - padding.right)
          return <text key={`x-${tick}`} x={x} y={height - 7} textAnchor={index === 0 ? 'start' : index === xTicks.length - 1 ? 'end' : 'middle'} className="chart-label">{formatStepAxisValue(tick)}</text>
        })}
      </svg>
    </article>
  )
}

function CheckpointCard({ checkpoint, initiallyExpanded = false }: { checkpoint: TrainingCheckpoint; initiallyExpanded?: boolean }): React.JSX.Element {
  const [expanded, setExpanded] = useState(initiallyExpanded)
  const records = useMemo(() => recordsFor(checkpoint), [checkpoint])
  const fields = useMemo(() => {
    const allFields = [...new Set(records.flatMap((record) => Object.keys(record.values)))]
      .filter((field) => !hiddenFields.has(field) && !field.endsWith('_samples'))
    const rank = (field: string) => {
      const preferredRank = preferredFields.indexOf(field)
      if (preferredRank >= 0) return preferredRank
      if (field.startsWith('validation.policy')) return 100
      if (field.startsWith('validation.value')) return 110
      if (field.startsWith('validation.start')) return 120
      if (field.startsWith('validation.end')) return 130
      if (field.startsWith('validation.complete')) return 140
      if (field.startsWith('training_')) return 150
      if (field.endsWith('learning_rate')) return 160
      return 200
    }
    return allFields.sort((left, right) => rank(left) - rank(right) || left.localeCompare(right))
  }, [records])
  const latest = records.at(-1)?.values ?? {}
  const epoch = latest.epoch ?? records.length
  const start = checkpoint.progress[0] ?? {}
  const startValues = flatten(start)
  const logConfig = start && typeof start.config === 'object' && start.config !== null ? start.config as Record<string, unknown> : {}
  const config = logConfig
  const targetSteps = typeof config.target_steps === 'number' ? config.target_steps : undefined
  const totalSteps = latest.global_step ?? startValues.global_step ?? 0
  const completedDuration = checkpoint.metrics.reduce((total, record) => {
    const value = record.epoch_seconds
    return total + (typeof value === 'number' ? value : 0)
  }, 0)
  const parameters = Object.entries(config)
    .filter(([name]) => !['data_dir', 'checkpoint_dir', 'max_steps'].includes(name) && !hiddenParameters.has(name))
    .sort(([left], [right]) => {
      const leftRank = parameterOrder.indexOf(left)
      const rightRank = parameterOrder.indexOf(right)
      return (leftRank < 0 ? 1000 : leftRank) - (rightRank < 0 ? 1000 : rightRank) || left.localeCompare(right)
    })
  const bestEpochs: Partial<Record<BestMarker['kind'], number>> = {
    best: bestEpochFor(records, ['validation_j_select']),
    'best-policy': bestEpochFor(records, ['validation.cp_policy_kl']),
    'best-value': bestEpochFor(records, ['validation.value_cp_mae_le_300']),
  }

  return (
    <article className={`training-card${expanded ? ' is-expanded' : ''}`}>
      <button type="button" className="training-card-toggle" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
        <span className="training-card-title"><strong>{checkpoint.name}</strong></span>
        <span className="training-card-summary">
          {latest['validation_j_select'] !== undefined && <b>J {latest['validation_j_select'].toFixed(4)}</b>}
          {(latest['validation.loss'] ?? latest['validation.joint_loss']) !== undefined && <b>Loss {(latest['validation.loss'] ?? latest['validation.joint_loss']).toFixed(3)}</b>}
          {latest['validation.policy_loss'] !== undefined && <b>Policy {latest['validation.policy_loss'].toFixed(3)}</b>}
          {latest['validation.value_loss'] !== undefined && <b>Value {latest['validation.value_loss'].toFixed(3)}</b>}
        </span>
      </button>
      {expanded && (
        <div className="training-card-body">
            <div className="training-card-meta">
              <div><small>Epoch</small><strong>{epoch}{config.epochs ? ` / ${config.epochs}` : ''}</strong></div>
              <div><small>Total steps</small><strong>{totalSteps}{targetSteps ? ` / ${targetSteps}` : ''}</strong></div>
              <div><small>Training time</small><strong>{formatDuration(completedDuration)}</strong></div>
            </div>
            {parameters.length > 0 && <div className="training-parameters">{parameters.map(([name, value]) => <span key={name}><b>{name}</b> {formatParameter(value)}</span>)}</div>}
          <div className="training-chart-legend" aria-label="Best checkpoint legend">
            <span><i className="chart-legend-dot best" />best</span>
            {fields.some((field) => markerKindFor(field) === 'best-policy') && <span><i className="chart-legend-dot best-policy" />best-policy</span>}
            {fields.some((field) => markerKindFor(field) === 'best-value') && <span><i className="chart-legend-dot best-value" />best-value</span>}
          </div>
          {fields.length === 0 ? <p className="training-empty">No numeric metrics available.</p> : fields.map((field) => {
            const points = pointsFor(records, field)
            return points.length > 0 ? <MetricChart key={field} field={field} points={points} bestEpochs={bestEpochs} /> : null
          })}
        </div>
      )}
    </article>
  )
}

export function TrainingView({ active }: { active: boolean }): React.JSX.Element {
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpoint[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    if (!active) return
    setLoading(true)
    request<TrainingCheckpoint[]>('/api/training/checkpoints')
      .then(setCheckpoints)
      .catch((loadError: unknown) => setError(loadError instanceof Error ? loadError.message : String(loadError)))
      .finally(() => setLoading(false))
  }, [active, reloadToken])

  return (
    <section className="training-view" hidden={!active}>
      <header className="view-header training-header">
        <div><p className="eyebrow">MODEL LAB</p><h2>训练监控</h2><p>按 checkpoint 浏览训练轨迹，展开卡片查看所有可用参数。</p></div>
        <button type="button" className="training-refresh" onClick={() => setReloadToken((value) => value + 1)}>刷新</button>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {loading ? <p className="training-empty">正在读取训练记录...</p> : checkpoints.length === 0 ? <p className="training-empty">暂无训练记录。</p> : <div className="training-cards">{[...checkpoints].sort((left, right) => right.name.localeCompare(left.name)).map((checkpoint) => <CheckpointCard key={checkpoint.id} checkpoint={checkpoint} initiallyExpanded={false} />)}</div>}
    </section>
  )
}