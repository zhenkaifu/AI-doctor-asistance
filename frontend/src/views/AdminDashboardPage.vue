<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAuth } from '../composables/useAuth'

const router = useRouter()
const { logout } = useAuth()

async function handleLogout() {
  await logout()
  router.replace('/login')
}
</script>

<template>
  <div class="admin-page">
    <!-- 顶部 -->
    <header class="top-bar">
      <h1 class="page-title">🔧 管理员控制台</h1>
      <div class="top-right">
        <span class="role-badge admin-badge">管理员</span>
        <button class="logout-btn" @click="handleLogout">退出登录</button>
      </div>
    </header>

    <!-- 功能卡片 -->
    <div class="cards">
      <div class="card" @click="router.push('/admin/staff')">
        <div class="card-icon">👥</div>
        <h2>医护人员管理</h2>
        <p>查看、新增医生/护士/管理员账号</p>
      </div>

      <div class="card" @click="router.push('/')">
        <div class="card-icon">🔍</div>
        <h2>病人查询</h2>
        <p>搜索病人信息，进入诊疗助手</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.admin-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 1.5rem 1rem;
  text-align: left;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.page-title {
  font-size: 1.4rem;
  margin: 0;
  color: var(--text-h);
}

.top-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.role-badge {
  display: inline-block;
  padding: 0.25rem 0.7rem;
  border-radius: 12px;
  font-size: 0.78rem;
  font-weight: 600;
}

.admin-badge {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.logout-btn {
  padding: 0.35rem 0.9rem;
  font-size: 0.82rem;
  color: var(--text);
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}
.logout-btn:hover {
  background: var(--code-bg);
}

.cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.2rem;
}

@media (max-width: 550px) {
  .cards {
    grid-template-columns: 1fr;
  }
}

.card {
  padding: 1.8rem 1.5rem;
  border: 1px solid var(--border);
  border-radius: 12px;
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
  background: var(--bg);
}
.card:hover {
  border-color: var(--accent);
  box-shadow: var(--shadow);
}

.card-icon {
  font-size: 2rem;
  margin-bottom: 0.6rem;
}

.card h2 {
  font-size: 1.1rem;
  margin: 0 0 0.4rem;
  color: var(--text-h);
}

.card p {
  font-size: 0.85rem;
  color: var(--text);
  margin: 0;
}
</style>
