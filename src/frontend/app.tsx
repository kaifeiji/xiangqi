import { useEffect, useState } from 'react'
import { request } from './api'
import { HumanModelView } from './human-model-view'
import { ModelMatchView } from './model-match-view'
import { ReplayView } from './replay-view'
import type { Mode, ModelOption } from './types'

export function App(): React.JSX.Element {
  const [mode, setMode] = useState<Mode>('human-model')
  const [models, setModels] = useState<ModelOption[]>([])
  const [error, setError] = useState('')
  const [modelsLoaded, setModelsLoaded] = useState(false)

  useEffect(() => {
    request<{ models: ModelOption[] }>('/api/models')
      .then(({ models: loadedModels }) => {
        setModels(loadedModels)
      })
      .catch((loadError: unknown) => setError(loadError instanceof Error ? loadError.message : String(loadError)))
      .finally(() => setModelsLoaded(true))
  }, [])

  return (
    <main className="app-shell">
      <header className="page-header">
        <h1>象棋对弈</h1>
        <nav className="mode-menu" aria-label="模式切换">
          {([
            ['human-model', '人机'],
            ['model-model', '模型对弈'],
            ['replay', '回放'],
          ] as const).map(([value, label]) => (
            <button key={value} type="button" aria-selected={mode === value} onClick={() => setMode(value)}>{label}</button>
          ))}
        </nav>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      <HumanModelView active={mode === 'human-model'} models={models} modelsLoaded={modelsLoaded} />
      <ModelMatchView active={mode === 'model-model'} models={models} modelsLoaded={modelsLoaded} />
      <ReplayView active={mode === 'replay'} />
    </main>
  )
}