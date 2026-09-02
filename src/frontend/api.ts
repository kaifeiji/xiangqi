import { piecesToBoard } from './game-utils'

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (response.status === 204) return undefined as T
  const payload = (await response.json()) as T & { error?: string }
  if (!response.ok) {
    throw new Error(payload.error ?? '请求失败')
  }
  if ('board' in payload && Array.isArray(payload.board) && payload.board.length === 90) {
    return { ...payload, board: piecesToBoard(payload.board as Array<string | null>) } as T
  }
  return payload as T
}
