<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAuth } from '../composables/useAuth'
import PatientSearch from './PatientSearch.vue'

const router = useRouter()
const { logout } = useAuth()

async function handleLogout() {
  await logout()
  router.push('/login')
}
</script>

<template>
  <div class="nurse-dashboard">
    <header class="top-bar">
      <div class="brand">
        <div class="brand-icon">护</div>
        <div>
          <h1 class="page-title">护士工作台</h1>
          <p class="subtitle">查询病人 · 挂号处理 · 月度统计</p>
        </div>
      </div>
      <button class="logout-btn" @click="handleLogout">退出登录</button>
    </header>

    <PatientSearch embedded />
  </div>
</template>

<style scoped>
.nurse-dashboard {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 2rem;
  text-align: left;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.brand-icon {
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 10px;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: 700;
}

.page-title {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--text-h);
}

.subtitle {
  margin: 0.2rem 0 0;
  font-size: 0.82rem;
  color: var(--text);
}

.logout-btn {
  padding: 0.4rem 0.9rem;
  font-size: 0.82rem;
  color: var(--text);
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
}
.logout-btn:hover {
  background: var(--code-bg);
  color: #2563eb;
}
</style>
