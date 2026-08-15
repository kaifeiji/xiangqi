interface ReplayPanelProps {
  error: string
  position: number
  length: number
  onLoad: (file: File) => void
  onStep: (offset: number) => void
}

export function ReplayPanel({ error, position, length, onLoad, onStep }: ReplayPanelProps): React.JSX.Element {
  return (
    <aside className="controls" aria-label="回放设置">
      <label>
        加载存档
        <input type="file" accept="application/json,.json" onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onLoad(file)
          event.target.value = ''
        }} />
      </label>
      {length > 0 && (
        <div className="replay-controls">
          <button type="button" onClick={() => onStep(-1)} disabled={position === 0}>上一步</button>
          <span>{position + 1} / {length}</span>
          <button type="button" onClick={() => onStep(1)} disabled={position + 1 === length}>下一步</button>
        </div>
      )}
      <p className="status" role="status">{length ? '回放中' : '请选择存档'}</p>
      <p className="error" role="alert">{error}</p>
    </aside>
  )
}
