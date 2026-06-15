<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { supabase } from '../lib/supabase'
import { useAuth } from '../composables/useAuth'
import { useApiError } from '../composables/useApiError'
import { ensureValidSession, runQuery } from '../lib/session'
import { computeRegStats, normalizeRegistrationStatus, type RegStats } from '../lib/registration'
import PatientForm from '../components/PatientForm.vue'

interface Patient {
  id: string
  name: string
  gender: string | null
  date_of_birth: string | null
  phone: string | null
  id_card_last5: string | null
  past_history: string | null
  allergy_history: string | null
  surgery_history: string | null
  emergency_contact_name: string | null
  emergency_contact_phone: string | null
  address: string | null
  created_at: string
}

interface SavedPatientHint {
  name: string
  phone: string
  id_card_last5?: string | null
}

const props = withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

const router = useRouter()
const { role } = useAuth()
const { resolveError } = useApiError()

const isNurse = computed(() => role.value === 'nurse')
const backPath = computed(() => role.value === 'admin' ? '/admin' : '/login')

const patients = ref<Patient[]>([])
const regStatsMap = ref<Record<string, RegStats>>({})
const loading = ref(false)
const searched = ref(false)
const searchError = ref('')
const regFetchError = ref('')

const searchForm = reactive({
  name: '',
  phone: '',
  id_card_last5: '',
})

const showForm = ref(false)
const editingPatient = ref<Patient | null>(null)

const monthLabel = computed(() => {
  const d = new Date()
  return `${d.getFullYear()}年${d.getMonth() + 1}月`
})

const patientsWithStats = computed(() =>
  patients.value.map(p => ({
    patient: p,
    stats: regStatsMap.value[p.id] || { todayCount: 0, monthCount: 0 },
  })),
)

async function doSearch() {
  loading.value = true
  searched.value = true
  searchError.value = ''
  regFetchError.value = ''

  try {
    await ensureValidSession()

    let query = supabase.from('patients').select('*')
    const filters: string[] = []
    if (searchForm.name.trim()) filters.push(`name.ilike.%${searchForm.name.trim()}%`)
    if (searchForm.phone.trim()) filters.push(`phone.eq.${searchForm.phone.trim()}`)
    if (searchForm.id_card_last5.trim()) filters.push(`id_card_last5.eq.${searchForm.id_card_last5.trim()}`)
    if (filters.length > 0) query = query.or(filters.join(','))

    const { data, error } = await runQuery(
      query.order('created_at', { ascending: false }),
      '查询病人',
    )

    if (error) {
      patients.value = []
      regStatsMap.value = {}
      searchError.value = await resolveError(error, '查询病人失败，请稍后重试')
    } else {
      patients.value = (data as Patient[]) || []
      if (isNurse.value && patients.value.length > 0) {
        const regErr = await fetchMonthlyStats(patients.value.map(p => p.id))
        if (regErr) regFetchError.value = regErr
      } else {
        regStatsMap.value = {}
      }
    }
  } catch (e) {
    patients.value = []
    regStatsMap.value = {}
    searchError.value = await resolveError(e, '查询病人失败，请稍后重试')
  } finally {
    loading.value = false
  }
}

async function fetchMonthlyStats(patientIds: string[]): Promise<string | null> {
  try {
    await ensureValidSession()
    const { data, error } = await runQuery(
      supabase
        .from('registrations')
        .select('patient_id, appointment_time, status')
        .in('patient_id', patientIds),
      '加载挂号统计',
    )
    if (error) throw error

    const map: Record<string, { appointment_time: string; status: string }[]> = {}
    for (const id of patientIds) map[id] = []
    for (const reg of (data as { patient_id: string; appointment_time: string; status: string }[]) || []) {
      if (!map[reg.patient_id]) map[reg.patient_id] = []
      map[reg.patient_id].push({
        appointment_time: reg.appointment_time,
        status: normalizeRegistrationStatus(reg.status),
      })
    }
    const statsMap: Record<string, RegStats> = {}
    for (const id of patientIds) {
      statsMap[id] = computeRegStats(map[id] || [])
    }
    regStatsMap.value = statsMap
    return null
  } catch (e) {
    regStatsMap.value = {}
    return resolveError(e, '加载挂号统计失败')
  }
}

function openAddPatient() {
  editingPatient.value = null
  showForm.value = true
}

function onFormSaved(hint?: SavedPatientHint) {
  showForm.value = false
  if (hint) {
    searchForm.name = hint.name
    searchForm.phone = hint.phone
    searchForm.id_card_last5 = hint.id_card_last5 || ''
  }
  doSearch()
}

function goToAssistant(patient: Patient) {
  router.push({
    path: '/assistant',
    query: {
      patientId: patient.id,
      patientName: patient.name,
      patientGender: patient.gender || '',
      patientDateOfBirth: patient.date_of_birth || '',
    },
  })
}

function goToRegistration(patient: Patient) {
  router.push({
    path: '/nurse/registration',
    query: { patientId: patient.id, patientName: patient.name },
  })
}
</script>

<template>
  <div class="patient-page">
    <header v-if="!props.embedded" class="top-bar">
      <button class="back-btn" @click="router.push(backPath)">&larr; 返回</button>
      <h1 class="page-title">病人查询</h1>
      <button class="add-btn" @click="openAddPatient">+ 新增病人</button>
    </header>
    <div v-else class="embedded-toolbar">
      <button class="add-btn" @click="openAddPatient">+ 新增病人</button>
    </div>

    <div class="search-area">
      <div class="search-row">
        <input v-model="searchForm.name" type="text" placeholder="姓名（支持模糊搜索）" class="search-input" @keyup.enter="doSearch" />
        <input v-model="searchForm.phone" type="text" placeholder="手机号（精确匹配）" class="search-input" @keyup.enter="doSearch" />
        <input v-model="searchForm.id_card_last5" type="text" placeholder="身份证后5位" maxlength="5" class="search-input" @keyup.enter="doSearch" />
        <button class="search-btn" :disabled="loading" @click="doSearch">{{ loading ? '查询中…' : '查询' }}</button>
      </div>
    </div>

    <div v-if="searched" class="result-area">
      <div v-if="loading" class="loading-text">查询中…</div>
      <div v-else-if="searchError" class="error-banner">
        <span class="error-banner-icon">⚠</span>
        <span>{{ searchError }}</span>
      </div>
      <template v-else>
        <div v-if="regFetchError" class="warn-banner">{{ regFetchError }}</div>
        <div v-if="patients.length === 0" class="empty-text">未找到匹配的病人记录</div>
        <table v-else class="patient-table">
          <thead>
            <tr>
              <th>姓名</th>
              <th>性别</th>
              <th>出生日期</th>
              <th>手机号</th>
              <th>身份证后5位</th>
              <th v-if="isNurse">每月挂号信息</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="{ patient: p, stats } in patientsWithStats" :key="p.id">
              <td>{{ p.name }}</td>
              <td>{{ p.gender || '-' }}</td>
              <td>{{ p.date_of_birth || '-' }}</td>
              <td>{{ p.phone || '-' }}</td>
              <td>{{ p.id_card_last5 || '-' }}</td>
              <td v-if="isNurse" class="stats-cell">
                <div class="stats-card">
                  <div class="stat-row">
                    <span class="stat-label">今日</span>
                    <span v-if="stats.todayCount > 0" class="stat-value today">已挂号 {{ stats.todayCount }} 次</span>
                    <span v-else class="stat-value none">未挂号</span>
                  </div>
                  <div class="stat-row">
                    <span class="stat-label">{{ monthLabel }}</span>
                    <span class="stat-value month">共 {{ stats.monthCount }} 次</span>
                  </div>
                </div>
              </td>
              <td>
                <button v-if="isNurse" class="action-btn" @click="goToRegistration(p)">挂号处理</button>
                <button v-else class="action-btn primary" @click="goToAssistant(p)">开始问诊</button>
              </td>
            </tr>
          </tbody>
        </table>
      </template>
    </div>

    <PatientForm
      v-if="showForm"
      :patient="editingPatient"
      @close="showForm = false"
      @saved="onFormSaved"
    />
  </div>
</template>

<style scoped>
.patient-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0;
  text-align: left;
}

.embedded-toolbar {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 1rem;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  gap: 1rem;
}

.page-title {
  font-size: 1.5rem;
  margin: 0;
  flex: 1;
  text-align: center;
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
  width: 100px;
  text-align: left;
}
.back-btn:hover { background: var(--code-bg); color: #2563eb; }

.add-btn {
  padding: 0.5rem 1.2rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  white-space: nowrap;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
}
.add-btn:hover { opacity: 0.9; }

.search-area { margin-bottom: 1.5rem; }
.search-row { display: flex; gap: 0.75rem; flex-wrap: wrap; }

.search-input {
  flex: 1;
  min-width: 150px;
  padding: 0.6rem 0.9rem;
  font-size: 0.9rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text-h);
  outline: none;
}
.search-input:focus { border-color: #3b82f6; }

.search-btn {
  padding: 0.6rem 1.5rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: #2563eb;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}
.search-btn:disabled { opacity: 0.5; }

.loading-text, .empty-text {
  text-align: center;
  padding: 2rem;
  color: var(--text);
}

.error-banner {
  display: flex;
  gap: 0.6rem;
  padding: 0.85rem 1rem;
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.25);
  border-radius: 8px;
}

.warn-banner {
  padding: 0.65rem 1rem;
  margin-bottom: 0.75rem;
  font-size: 0.85rem;
  color: #b45309;
  background: rgba(245, 158, 11, 0.1);
  border-radius: 8px;
}

.patient-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.patient-table th,
.patient-table td {
  padding: 0.7rem 0.8rem;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: middle;
}

.patient-table th {
  font-weight: 600;
  color: var(--text-h);
  background: var(--code-bg);
  font-size: 0.85rem;
}

.stats-cell { min-width: 200px; }

.stats-card {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  padding: 0.5rem 0.65rem;
  background: var(--code-bg);
  border-radius: 8px;
  border: 1px solid var(--border);
}

.stat-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.82rem;
}

.stat-label {
  flex-shrink: 0;
  width: 4.5rem;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text);
}

.stat-value.today {
  font-weight: 600;
  color: #059669;
}

.stat-value.none {
  color: var(--text);
}

.stat-value.month {
  font-weight: 600;
  color: #2563eb;
}

.action-btn {
  padding: 0.4rem 0.9rem;
  font-size: 0.82rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid rgba(37, 99, 235, 0.35);
  background: rgba(37, 99, 235, 0.08);
  color: #2563eb;
  transition: background 0.15s;
  white-space: nowrap;
}
.action-btn:hover { background: rgba(37, 99, 235, 0.14); }

.action-btn.primary {
  background: #2563eb;
  color: #fff;
  border-color: #2563eb;
}
.action-btn.primary:hover { opacity: 0.9; }
</style>
