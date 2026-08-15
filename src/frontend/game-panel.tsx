import type { GameMode, ModelOption, Side } from './types'

interface GamePanelProps {
  mode: GameMode
  humanSide: Side
  models: ModelOption[]
  model: string
  redModel: string
  blackModel: string
  status: string
  error: string
  canStart: boolean
  canStep: boolean
  canSave: boolean
  onHumanSideChange: (side: Side) => void
  onModelChange: (model: string) => void
  onRedModelChange: (model: string) => void
  onBlackModelChange: (model: string) => void
  onStart: () => void
  onStep: () => void
  onSave: () => void
}

export function GamePanel(props: GamePanelProps): React.JSX.Element {
  const isHumanGame = props.mode === 'human-model'
  const modelLabel = props.humanSide === 'w' ? '黑方模型' : '红方模型'
  const options = props.models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)

  return (
    <aside className="controls" aria-label="对局设置">
      {isHumanGame ? (
        <label>
          人类执子
          <select value={props.humanSide} onChange={(event) => props.onHumanSideChange(event.target.value as Side)}>
            <option value="w">红方</option>
            <option value="b">黑方</option>
          </select>
        </label>
      ) : null}
      {isHumanGame ? (
        <label>
          {modelLabel}
          <select value={props.model} onChange={(event) => props.onModelChange(event.target.value)} disabled={!props.models.length}>
            {options}
          </select>
        </label>
      ) : (
        <>
          <label>
            红方模型
            <select value={props.redModel} onChange={(event) => props.onRedModelChange(event.target.value)} disabled={!props.models.length}>
              {options}
            </select>
          </label>
          <label>
            黑方模型
            <select value={props.blackModel} onChange={(event) => props.onBlackModelChange(event.target.value)} disabled={!props.models.length}>
              {options}
            </select>
          </label>
        </>
      )}
      <button type="button" onClick={props.onStart} disabled={!props.canStart}>开始新对局</button>
      {props.canStep && <button id="step-button" type="button" onClick={props.onStep}>模型走一步</button>}
      {props.canSave && <button type="button" onClick={props.onSave}>保存存档</button>}
      <p className="status" role="status">{props.status}</p>
      <p className="error" role="alert">{props.error}</p>
    </aside>
  )
}