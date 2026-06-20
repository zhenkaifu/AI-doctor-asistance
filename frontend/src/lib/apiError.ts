/** 判断是否为登录态失效（JWT 过期、未认证等） */
export function isAuthError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false
  const e = error as Record<string, unknown>
  const msg = String(e.message || e.error || '').toLowerCase()
  const code = String(e.code || '')
  const status = e.status as number | undefined
  return (
    status === 401 ||
    code === 'PGRST301' ||
    msg.includes('jwt expired') ||
    msg.includes('invalid jwt') ||
    msg.includes('not authenticated') ||
    msg.includes('unauthorized') ||
    msg.includes('未登录') ||
    (msg.includes('session') && msg.includes('expired'))
  )
}

/** 构造带 HTTP 状态码的错误，供 isAuthError 识别 401 */
export function createHttpError(status: number, message: string): Error & { status: number } {
  const err = new Error(message) as Error & { status: number }
  err.status = status
  return err
}

export function formatApiError(error: unknown, fallback = '请求失败，请稍后重试'): string {
  if (!error) return fallback
  if (isAuthError(error)) {
    return '登录已过期，请重新登录后再试'
  }
  if (typeof error === 'string') return error
  if (typeof error === 'object') {
    const e = error as Record<string, unknown>
    if (typeof e.message === 'string' && e.message.trim()) return e.message
    if (typeof e.error === 'string' && e.error.trim()) return e.error
  }
  return fallback
}
