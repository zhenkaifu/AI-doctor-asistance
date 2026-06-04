import { ref, computed } from 'vue'
import { supabase } from '../lib/supabase'
import type { User } from '@supabase/supabase-js'

export type UserRole = 'admin' | 'doctor' | 'nurse' | null

const user = ref<User | null>(null)
const role = ref<UserRole>(null)
const loading = ref(true)
let roleRefreshTimer: ReturnType<typeof setInterval> | null = null
let lastRoleCheckTime = 0
const ROLE_REFRESH_INTERVAL = 60000  // 60秒刷新一次权限

export function useAuth() {
  const isAuthenticated = computed(() => !!user.value)

  /** 启动时恢复会话 */
  async function init() {
    loading.value = true
    try {
      const { data } = await supabase.auth.getSession()
      if (data.session?.user) {
        user.value = data.session.user
        role.value = await fetchRole()
        console.log('[useAuth.init] 恢复会话 role=', role.value)
        startRoleRefreshTimer()
      }
    } finally {
      loading.value = false
    }

    // 监听 auth 状态变更
    supabase.auth.onAuthStateChange(async (event, session) => {
      console.log('[useAuth.onAuthStateChange] event=', event)
      if (event === 'SIGNED_IN' && session?.user) {
        user.value = session.user
        role.value = await fetchRole()
        console.log('[useAuth.onAuthStateChange] SIGNED_IN role=', role.value)
        startRoleRefreshTimer()
      } else if (event === 'SIGNED_OUT') {
        user.value = null
        role.value = null
        stopRoleRefreshTimer()
        console.log('[useAuth.onAuthStateChange] SIGNED_OUT')
      }
    })
  }

  /** 查询角色（带重试） */
  async function fetchRole(retryCount = 0): Promise<UserRole> {
    try {
      console.log('[useAuth.fetchRole] 开始查询 get_user_role RPC (重试' + retryCount + ')...')
      const { data, error } = await supabase.rpc('get_user_role')
      console.log('[useAuth.fetchRole] RPC 返回 data=', data, 'error=', error)
      if (error) {
        throw new Error(error.message || '查询失败')
      }
      lastRoleCheckTime = Date.now()
      return (data as UserRole) || null
    } catch (e) {
      // 第一次失败时重试一次
      if (retryCount < 1) {
        console.warn('[useAuth.fetchRole] 查询失败，2秒后重试:', (e as Error).message)
        await new Promise(r => setTimeout(r, 2000))
        return fetchRole(retryCount + 1)
      }
      console.error('fetchRole error:', e)
      return null
    }
  }

  /** 启动权限定期刷新 */
  function startRoleRefreshTimer() {
    if (roleRefreshTimer) return
    console.log('[useAuth.startRoleRefreshTimer] 启动权限定期刷新 (间隔60s)')
    roleRefreshTimer = setInterval(async () => {
      if (isAuthenticated.value) {
        const now = Date.now()
        // 如果上次检查时间距离当前未超过间隔，则跳过本次请求
        if (now - lastRoleCheckTime < ROLE_REFRESH_INTERVAL) return
        try {
          const newRole = await fetchRole()
          if (newRole && newRole !== role.value) {
            console.log('[useAuth] 权限已改变:', role.value, '→', newRole)
            role.value = newRole
          }
        } catch (e) {
          console.warn('[useAuth] 定期权限刷新失败:', e)
        }
      }
    }, ROLE_REFRESH_INTERVAL)
  }

  /** 停止权限定期刷新 */
  function stopRoleRefreshTimer() {
    if (roleRefreshTimer) {
      clearInterval(roleRefreshTimer)
      roleRefreshTimer = null
    }
  }

  /** 立即刷新权限（主动查询，带超时和降级） */
  async function refreshRole() {
    try {
      console.log('[useAuth.refreshRole] 开始刷新权限，超时 8秒...')
      
      // 使用 Promise.race 实现 8 秒超时
      // 将结果断言为 `UserRole | null`，避免 Promise.race 推断出宽泛的 `{}` 类型
      const result = (await Promise.race([
        fetchRole(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('权限查询超时(8s)')), 8000)
        )
      ])) as UserRole | null
      
      if (result !== undefined && result !== null) {
        console.log('[useAuth.refreshRole] 权限刷新成功:', result)
        role.value = result
        return result
      } else if (result === null && role.value !== null) {
        // 查询返回 null，但缓存有值，保持缓存值
        console.log('[useAuth.refreshRole] 查询返回 null，保持缓存权限:', role.value)
        return role.value
      }
      
      return result
    } catch (e) {
      const errorMsg = (e as Error).message
      console.warn('[useAuth.refreshRole] 权限查询失败:', errorMsg)
      
      // 降级：使用缓存权限，允许用户继续操作
      if (role.value) {
        console.warn('[useAuth.refreshRole] 使用缓存权限进行降级:', role.value)
        return role.value
      }
      
      return null
    }
  }

  /** 邮箱+密码登录 */
  async function login(email: string, password: string) {
    console.log('[useAuth.login] 开始登录 email=', email)
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (error) throw error

    console.log('[useAuth.login] 登录成功 user.id=', data.user?.id)

    // 同步设置 user 并等待 role 查询完成，避免 LoginPage 跳转时 role 尚未就绪
    if (data.user) {
      user.value = data.user
      role.value = await fetchRole()
      startRoleRefreshTimer()
      console.log('[useAuth.login] 同步设置后 role.value=', role.value, 'isAuthenticated=', !!user.value)
    }

    return data
  }

  /** 登出 */
  async function logout() {
    await supabase.auth.signOut()
    user.value = null
    role.value = null
    stopRoleRefreshTimer()
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
