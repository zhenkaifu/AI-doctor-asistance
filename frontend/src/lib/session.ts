import type { PostgrestError } from '@supabase/supabase-js'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './supabase'
import { createHttpError } from './apiError'

export const REQUEST_TIMEOUT_MS = 15000

/** 为 Promise / Thenable 添加超时，避免请求永久挂起导致 loading 不结束 */
export function withTimeout<T>(
  promise: PromiseLike<T>,
  ms = REQUEST_TIMEOUT_MS,
  label = '请求',
): Promise<T> {
  return Promise.race([
    Promise.resolve(promise),
    new Promise<never>((_, reject) => {
      setTimeout(
        () => reject(new Error(`${label}超时（${ms / 1000}秒），请检查网络或重新登录`)),
        ms,
      )
    }),
  ])
}

/** 执行 Supabase 查询（带超时） */
export async function runQuery<T>(
  query: PromiseLike<{ data: T; error: PostgrestError | null }>,
  label: string,
  ms = REQUEST_TIMEOUT_MS,
): Promise<{ data: T; error: PostgrestError | null }> {
  return withTimeout(query, ms, label)
}

/**
 * 发起 API 前确保会话有效：token 即将过期时主动刷新。
 * 解决「登录后过一会儿请求挂起/失败」的问题。
 */
export async function ensureValidSession(): Promise<Session> {
  const { data: { session }, error } = await withTimeout(
    supabase.auth.getSession(),
    8000,
    '获取登录状态',
  )
  if (error) throw error

  if (!session?.access_token) {
    throw createHttpError(401, '未登录，请重新登录')
  }

  const expiresAt = session.expires_at ?? 0
  const nowSec = Math.floor(Date.now() / 1000)
  const needsRefresh = expiresAt - nowSec < 120

  if (needsRefresh) {
    const { data: refreshed, error: refreshErr } = await withTimeout(
      supabase.auth.refreshSession(),
      10000,
      '刷新登录状态',
    )
    if (refreshErr || !refreshed.session?.access_token) {
      throw createHttpError(401, '登录已过期，请重新登录')
    }
    return refreshed.session
  }

  return session
}

/** 获取有效的 access token（自动刷新） */
export async function getAccessToken(): Promise<string> {
  const session = await ensureValidSession()
  return session.access_token
}

const functionsBase = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1`

/** 调用 Supabase Edge Function（带会话刷新与超时） */
export async function callEdgeFunction<T = unknown>(
  name: string,
  init?: { method?: string; body?: unknown },
): Promise<T> {
  const token = await getAccessToken()
  const method = init?.method ?? (init?.body !== undefined ? 'POST' : 'GET')

  const res = await withTimeout(
    fetch(`${functionsBase}/${name}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(init?.body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
    }),
    REQUEST_TIMEOUT_MS,
    name,
  )

  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw createHttpError(res.status, (body as { error?: string }).error || '请求失败')
  }
  return body as T
}
