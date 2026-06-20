import { ref, computed } from 'vue'
import { supabase } from '../lib/supabase'
import { ensureValidSession, withTimeout, runQuery } from '../lib/session'
import { isAuthError } from '../lib/apiError'
import type { User } from '@supabase/supabase-js'

export type UserRole = 'admin' | 'doctor' | 'nurse' | null

const user = ref<User | null>(null)
const role = ref<UserRole>(null)
const loading = ref(true)
let roleRefreshTimer: ReturnType<typeof setInterval> | null = null
let authListenerRegistered = false
let visibilityListenerRegistered = false
let lastRoleCheckTime = 0
const ROLE_REFRESH_INTERVAL = 5 * 60 * 1000

export function useAuth() {
  const isAuthenticated = computed(() => !!user.value)

  /** 查询角色（带超时，不在 onAuthStateChange 内直接调用） */
  async function fetchRole(): Promise<UserRole> {
    await ensureValidSession()
    const { data, error } = await runQuery(
      supabase.rpc('get_user_role'),
      '查询用户权限',
      10000,
    )
    if (error) throw error
    lastRoleCheckTime = Date.now()
    return (data as UserRole) || null
  }

  function scheduleRoleFetch() {
    setTimeout(() => {
      fetchRole()
        .then((r) => { role.value = r })
        .catch((e) => console.warn('[useAuth] 权限查询失败:', (e as Error).message))
    }, 0)
  }

  function setupVisibilityListener() {
    if (visibilityListenerRegistered) return
    visibilityListenerRegistered = true
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState !== 'visible' || !user.value) return
      ensureValidSession().catch(async (e) => {
        if (isAuthError(e)) await logout()
      })
    })
  }

  function registerAuthListener() {
    if (authListenerRegistered) return
    authListenerRegistered = true

    // 禁止在回调内 await Supabase API，否则会死锁导致后续请求永久挂起
    supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_OUT') {
        user.value = null
        role.value = null
        stopRoleRefreshTimer()
        return
      }

      if (session?.user) {
        user.value = session.user
        if (event === 'SIGNED_IN' || event === 'INITIAL_SESSION') {
          scheduleRoleFetch()
          startRoleRefreshTimer()
        }
      }
    })
  }

  /** 启动时恢复会话 */
  async function init() {
    loading.value = true
    registerAuthListener()
    setupVisibilityListener()

    try {
      const { data: { session } } = await withTimeout(
        supabase.auth.getSession(),
        8000,
        '恢复登录状态',
      )
      if (session?.user) {
        user.value = session.user
        try {
          role.value = await fetchRole()
        } catch (e) {
          console.warn('[useAuth.init] 权限查询失败:', (e as Error).message)
        }
        startRoleRefreshTimer()
      }
    } catch (e) {
      console.warn('[useAuth.init] 会话恢复失败:', (e as Error).message)
    } finally {
      loading.value = false
    }
  }

  function startRoleRefreshTimer() {
    if (roleRefreshTimer) return
    roleRefreshTimer = setInterval(() => {
      if (!user.value) return
      if (Date.now() - lastRoleCheckTime < ROLE_REFRESH_INTERVAL) return
      scheduleRoleFetch()
    }, ROLE_REFRESH_INTERVAL)
  }

  function stopRoleRefreshTimer() {
    if (roleRefreshTimer) {
      clearInterval(roleRefreshTimer)
      roleRefreshTimer = null
    }
  }

  async function refreshRole(): Promise<UserRole | null> {
    try {
      const result = await withTimeout(fetchRole(), 10000, '刷新权限')
      if (result) role.value = result
      return result
    } catch (e) {
      console.warn('[useAuth.refreshRole] 失败:', (e as Error).message)
      return role.value
    }
  }

  async function login(email: string, password: string) {
    const { data, error } = await withTimeout(
      supabase.auth.signInWithPassword({ email, password }),
      15000,
      '登录',
    )
    if (error) throw error

    if (data.user) {
      user.value = data.user
      role.value = await fetchRole()
      startRoleRefreshTimer()
    }

    return data
  }

  async function logout() {
    stopRoleRefreshTimer()
    try {
      await withTimeout(supabase.auth.signOut(), 5000, '登出')
    } catch {
      // 本地状态仍须清除
    }
    user.value = null
    role.value = null
  }

  return {
    user,
    role,
    loading,
    isAuthenticated,
    init,
    login,
    logout,
    refreshRole,
  }
}
