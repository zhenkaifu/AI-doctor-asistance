<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuth } from '../composables/useAuth'

const router = useRouter()
const { login, role } = useAuth()

const email = ref('')
const password = ref('')
const errorMsg = ref('')
const logging = ref(false)

async function handleLogin() {
  errorMsg.value = ''

  if (!email.value.trim() || !password.value) {
    errorMsg.value = '请输入邮箱和密码'
    return
  }

  logging.value = true
  try {
    await login(email.value.trim(), password.value)
    console.log('[LoginPage] login 返回后 role.value=', role.value, 'isAdmin=', role.value === 'admin')
    const target =
      role.value === 'admin' ? '/admin'
      : role.value === 'nurse' ? '/nurse'
      : role.value === 'doctor' ? '/doctor'
      : '/'
    console.log('[LoginPage] 即将跳转到', target)
    router.replace(target)
  } catch (e: any) {
    errorMsg.value = e.message || '登录失败'
  } finally {
    logging.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-card">
      <!-- 标题 -->
      <div class="login-header">
        <h1>AI 医生助手</h1>
        <p class="subtitle">临床决策辅助系统</p>
      </div>

      <!-- 错误提示 -->
      <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>

      <!-- 表单 -->
      <form class="login-form" @submit.prevent="handleLogin">
        <label class="field">
          <span class="field-label">邮箱</span>
          <input
            v-model="email"
            type="email"
            class="field-input"
            placeholder="请输入邮箱"
            autocomplete="email"
          />
        </label>

        <label class="field">
          <span class="field-label">密码</span>
          <input
            v-model="password"
            type="password"
            class="field-input"
            placeholder="请输入密码"
            autocomplete="current-password"
          />
        </label>

        <button
          type="submit"
          class="login-btn"
          :disabled="logging"
        >
          {{ logging ? '登录中…' : '登 录' }}
        </button>
      </form>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 2rem;
  background: var(--bg);
}

.login-card {
  width: 100%;
  max-width: 400px;
  padding: 2.5rem 2rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow);
}

.login-header {
  text-align: center;
  margin-bottom: 2rem;
}

.login-header h1 {
  font-size: 1.8rem;
  margin: 0 0 0.4rem;
  color: var(--text-h);
  letter-spacing: 0.04em;
}

.subtitle {
  font-size: 0.9rem;
  color: var(--text);
  margin: 0;
}

.error-msg {
  margin-bottom: 1.2rem;
  padding: 0.6rem 0.9rem;
  font-size: 0.85rem;
  color: #ef4444;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 6px;
  text-align: center;
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.field-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-h);
}

.field-input {
  padding: 0.65rem 0.8rem;
  font-size: 0.95rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text-h);
  outline: none;
  transition: border-color 0.2s;
}

.field-input:focus {
  border-color: var(--accent);
}

.login-btn {
  margin-top: 0.5rem;
  padding: 0.7rem;
  font-size: 1rem;
  font-weight: 600;
  color: #fff;
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.login-btn:hover {
  opacity: 0.85;
}

.login-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
