<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApiError } from '../composables/useApiError'
import { callEdgeFunction } from '../lib/session'
import LineChart from '../components/LineChart.vue'

interface DailyStat {
  date: string
  count: number
}

interface StaffStatsResponse {
  staffName: string
  role: 'nurse' | 'doctor'
  daily: DailyStat[]
  total: number
  average: number
}

const route = useRoute()
const router = useRouter()
const { resolveError } = useApiError()

const userId = computed(() => String(route.query.userId || ''))
const role = computed(() => String(route.query.role || '') as 'nurse' | 'doctor')
const displayName = computed(() => String(route.query.name || stats.value?.staffName || ''))

const loading = ref(false)
const error = ref('')
const stats = ref<StaffStatsResponse | null>(null)

const chartData = computed(() =>
  (stats.value?.daily ?? []).map((d) => ({ label: d.date, value: d.count })),
)

const chartColor = computed(() => (role.value === 'nurse' ? '#10b981' : '#3b82f6'))

const pageTitle = computed(() => {
  if (role.value === 'nurse') return '挂号数据统计'
  if (role.value === 'doctor') return '问诊数据统计'
  return '数据统计'
})

const metricLabel = computed(() =>
  role.value === 'nurse' ? '挂号' : '问诊',
)

async function fetchStats() {
  if (!userId.value || (role.value !== 'nurse' && role.value !== 'doctor')) {
    error.value = '参数无效，请从员工管理页进入'
    return
  }

  loading.value = true
  error.value = ''
  try {
    stats.value = await callEdgeFunction<StaffStatsResponse>('get-staff-stats', {
      body: { userId: userId.value, role: role.value },
    })
  } catch (e: unknown) {
    stats.value = null
    error.value = await resolveError(e, '加载统计数据失败')
  } finally {
    loading.value = false
  }
}

onMounted(() => { fetchStats() })
</script>

<template>
  <div class="stats-page">
    <header class="top-bar">
      <button class="back-btn" @click="router.push('/admin/staff')">&larr; 返回</button>
      <div class="title-block">
        <h1 class="page-title">{{ pageTitle }}</h1>
        <p class="subtitle">{{ displayName }} · 近 30 天（北京时间）</p>
      </div>
    </header>

    <div v-if="error" class="error-msg">{{ error }}</div>

    <div v-if="loading" class="loading-msg">加载中…</div>

    <template v-else-if="stats">
      <div class="summary-cards">
        <div class="summary-card">
          <span class="summary-label">30 日合计</span>
          <span class="summary-value">{{ stats.total }}</span>
          <span class="summary-unit">{{ metricLabel }}次</span>
        </div>
        <div class="summary-card">
          <span class="summary-label">日均</span>
          <span class="summary-value">{{ stats.average }}</span>
          <span class="summary-unit">{{ metricLabel }}次/天</span>
        </div>
      </div>

      <section class="chart-section">
        <h2 class="section-title">每日{{ metricLabel }}趋势</h2>
        <div class="chart-wrap">
          <LineChart
            :data="chartData"
            :color="chartColor"
            :unit="metricLabel + '次'"
          />
        </div>
      </section>

      <section class="table-section">
        <h2 class="section-title">每日明细</h2>
        <div class="table-wrap">
          <table class="stats-table">
            <thead>
              <tr>
                <th>日期</th>
                <th>{{ metricLabel }}数量</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in [...stats.daily].reverse()" :key="row.date">
                <td>{{ row.date }}</td>
                <td>{{ row.count }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.stats-page {
  padding: 1.5rem;
  text-align: left;
  max-width: 900px;
}
.top-bar {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.back-btn {
  padding: 0.35rem 0.8rem;
  font-size: 0.85rem;
  color: var(--text);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  white-space: nowrap;
  margin-top: 0.2rem;
}
.back-btn:hover {
  background: var(--code-bg);
  color: var(--accent);
}
.title-block { flex: 1; }
.page-title {
  font-size: 1.2rem;
  margin: 0 0 0.25rem;
  color: var(--text-h);
}
.subtitle {
  margin: 0;
  font-size: 0.88rem;
  color: var(--text);
}
.error-msg {
  margin-bottom: 1rem;
  padding: 0.6rem 0.9rem;
  color: #ef4444;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 6px;
  font-size: 0.85rem;
}
.loading-msg {
  text-align: center;
  padding: 3rem;
  color: var(--text);
}
.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.summary-card {
  padding: 1rem 1.2rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--code-bg);
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}
.summary-label {
  font-size: 0.82rem;
  color: var(--text);
}
.summary-value {
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text-h);
  line-height: 1.1;
}
.summary-unit {
  font-size: 0.8rem;
  color: var(--text);
}
.chart-section,
.table-section {
  margin-bottom: 1.5rem;
}
.section-title {
  font-size: 0.95rem;
  margin: 0 0 0.75rem;
  color: var(--text-h);
}
.chart-wrap {
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg);
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  max-height: 320px;
  overflow-y: auto;
}
.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
.stats-table th,
.stats-table td {
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
.stats-table th {
  position: sticky;
  top: 0;
  background: var(--code-bg);
  font-weight: 600;
  font-size: 0.8rem;
  color: var(--text);
}
.stats-table td {
  color: var(--text-h);
}
</style>
