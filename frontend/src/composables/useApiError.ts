import { useRouter } from 'vue-router'
import { formatApiError, isAuthError } from '../lib/apiError'
import { useAuth } from './useAuth'

/**
 * 统一处理 API 错误：格式化文案，登录过期时自动登出并跳转登录页。
 * 返回展示给用户的错误信息。
 */
export function useApiError() {
  const router = useRouter()
  const { logout } = useAuth()

  async function resolveError(
    error: unknown,
    fallback = '请求失败，请稍后重试',
  ): Promise<string> {
    const message = formatApiError(error, fallback)
    if (isAuthError(error)) {
      await logout()
      router.replace({ name: 'login' })
    }
    return message
  }

  return { formatApiError, isAuthError, resolveError }
}
