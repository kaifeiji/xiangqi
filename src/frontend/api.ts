export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  const payload = (await response.json()) as T & { error?: string }
  if (!response.ok) {
    throw new Error(payload.error ?? '请求失败')
  }
  return payload
}
