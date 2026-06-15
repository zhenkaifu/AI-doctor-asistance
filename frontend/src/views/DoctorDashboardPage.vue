<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { supabase } from '../lib/supabase'
import { useAuth } from '../composables/useAuth'
import { useApiError } from '../composables/useApiError'
import { ensureValidSession, runQuery } from '../lib/session'

const router = useRouter()
const { user, logout } = useAuth()
const { resolveError } = useApiError()

const purposeLabels: Record<string, string> = {
  initial: '初诊',
  follow_up: '复查',
  medication: '开药',
  consultation: '咨询',
  chronic_follow_up: '慢性病复诊',
}

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
}

interface QueueItem {
  id: string
  patient_id: string
  appointment_time: string
  chief_complaint: string
  purpose: string
  department_name: string
  has_covid_symptoms: boolean
  patients: Patient | Patient[] | null
}

interface MedicalRecord {
  id: string
  visit_date: string
  department: string
  chief_complaint: string
  diagnosis: string
  present_illness: string | null
  past_history: string | null
  physical_exam: string | null
  temperature: number | null
  pulse: number | null
  respiration: number | null
  blood_pressure: string | null
  auxiliary_exam: string | null
  medication: string | null
  medical_advice: string | null
  rest_days: number | null
  doctor_name: string
}

const doctorId = ref<string | null>(null)
const doctorName = ref('')
const doctorDept = ref('')

const queue = ref<QueueItem[]>([])
const selectedId = ref<string | null>(null)
const loadingQueue = ref(false)
const queueError = ref('')

const medicalRecords = ref<MedicalRecord[]>([])
const loadingRecords = ref(false)
const recordsError = ref('')

const todayLabel = computed(() => {
  const d = new Date()
  return d.toLocaleDateString('zh-CN', { month: 'long', day: 'numeric', weekday: 'short' })
})

const selectedItem = computed(() =>
  queue.value.find(q => q.id === selectedId.value) || null,
)

const selectedPatient = computed(() => {
  const p = selectedItem.value?.patients
  if (!p) return null
  return Array.isArray(p) ? p[0] || null : p
})

function patientNameOf(item: QueueItem): string {
  const p = item.patients
  if (!p) return '未知患者'
  const patient = Array.isArray(p) ? p[0] : p
  return patient?.name || '未知患者'
}

function parseAppointmentTime(dateStr: string): Date | null {
  if (!dateStr) return null
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(dateStr) && !dateStr.includes('Z') && !/[+-]\d{2}:\d{2}$/.test(dateStr)) {
    const [datePart, timePart] = dateStr.split('T')
    const [y, m, d] = datePart.split('-').map(Number)
    const [hh, mm, ss = 0] = timePart.split(':').map(Number)
    return new Date(y, m - 1, d, hh, mm, ss)
  }
  const d = new Date(dateStr)
  return isNaN(d.getTime()) ? null : d
}

function isAppointmentToday(dateStr: string): boolean {
  const d = parseAppointmentTime(dateStr)
  if (!d) return false
  const now = new Date()
  return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate()
}

function formatTimeOnly(dateStr: string): string {
  const d = parseAppointmentTime(dateStr)
  if (!d) return '-'
  return d.toLocaleString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function calcAge(dateOfBirth: string | null): string {
  if (!dateOfBirth) return '-'
  const birth = new Date(dateOfBirth)
  if (isNaN(birth.getTime())) return '-'
  const today = new Date()
  let age = today.getFullYear() - birth.getFullYear()
  const m = today.getMonth() - birth.getMonth()
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--
  return `${age} 岁`
}

async function loadDoctorProfile() {
  const authId = user.value?.id
  if (!authId) throw new Error('未登录')

  const { data, error } = await runQuery(
    supabase.from('doctors').select('id, name, department').eq('auth_id', authId).maybeSingle(),
    '查询医生信息',
    10000,
  )
  if (error) throw error
  if (!data) throw new Error('无法识别医生身份')

  doctorId.value = data.id
  doctorName.value = data.name
  doctorDept.value = data.department || ''
}

async function loadTodayQueue() {
  if (!doctorId.value) return
  loadingQueue.value = true
  queueError.value = ''

  try {
    await ensureValidSession()
    const { data, error } = await runQuery(
      supabase
        .from('registrations')
        .select(`
          id, patient_id, appointment_time, chief_complaint, purpose,
          department_name, has_covid_symptoms,
          patients (
            id, name, gender, date_of_birth, phone, id_card_last5,
            past_history, allergy_history, surgery_history,
            emergency_contact_name, emergency_contact_phone, address
          )
        `)
        .eq('doctor_id', doctorId.value)
        .eq('status', 'waiting')
        .order('appointment_time', { ascending: true }),
      '加载今日挂号',
    )

    if (error) throw error

    const items = ((data as unknown as QueueItem[]) || []).filter(r => isAppointmentToday(r.appointment_time))
    queue.value = items

    if (items.length > 0) {
      const stillExists = items.some(i => i.id === selectedId.value)
      if (!stillExists) selectRegistration(items[0].id)
    } else {
      selectedId.value = null
      medicalRecords.value = []
    }
  } catch (e) {
    queue.value = []
    selectedId.value = null
    queueError.value = await resolveError(e, '加载今日挂号失败，请稍后重试')
  } finally {
    loadingQueue.value = false
  }
}

async function loadMedicalRecords(patientId: string) {
  loadingRecords.value = true
  recordsError.value = ''
  medicalRecords.value = []

  try {
    await ensureValidSession()
    const { data, error } = await runQuery(
      supabase
        .from('medical_records')
        .select('*')
        .eq('patient_id', patientId)
        .order('visit_date', { ascending: false })
        .limit(10),
      '加载以往病例',
    )
    if (error) throw error
    medicalRecords.value = (data as MedicalRecord[]) || []
  } catch (e) {
    recordsError.value = await resolveError(e, '加载以往病例失败')
  } finally {
    loadingRecords.value = false
  }
}

function selectRegistration(id: string) {
  selectedId.value = id
}

watch(selectedId, (id) => {
  const item = queue.value.find(q => q.id === id)
  if (item?.patient_id) loadMedicalRecords(item.patient_id)
  else medicalRecords.value = []
})

function startConsultation() {
  const p = selectedPatient.value
  const item = selectedItem.value
  if (!p || !item) return
  router.push({
    path: '/assistant',
    query: {
      patientId: p.id,
      patientName: p.name,
      patientGender: p.gender || '',
      patientDateOfBirth: p.date_of_birth || '',
      registrationId: item.id,
      from: 'doctor',
    },
  })
}

async function handleLogout() {
  await logout()
  router.replace('/login')
}

onMounted(async () => {
  try {
    await loadDoctorProfile()
    await loadTodayQueue()
  } catch (e) {
    queueError.value = await resolveError(e, '初始化失败')
  }
})
</script>

<template>
  <div class="doctor-page">
    <header class="top-bar">
      <div class="top-left">
        <h1 class="page-title">医生工作台</h1>
        <span v-if="doctorName" class="doctor-chip">
          {{ doctorName }}
          <template v-if="doctorDept"> · {{ doctorDept }}</template>
        </span>
      </div>
      <div class="top-right">
        <span class="date-chip">{{ todayLabel }}</span>
        <button class="refresh-btn" :disabled="loadingQueue" @click="loadTodayQueue">
          {{ loadingQueue ? '刷新中…' : '刷新' }}
        </button>
        <button class="logout-btn" @click="handleLogout">退出</button>
      </div>
    </header>

    <div class="workspace">
      <!-- 左侧：今日挂号列表 -->
      <aside class="queue-panel">
        <div class="panel-head">
          <h2>今日候诊</h2>
          <span class="count-badge">{{ queue.length }} 人</span>
        </div>

        <div v-if="loadingQueue && queue.length === 0" class="panel-status">加载中…</div>
        <div v-else-if="queueError" class="panel-error">{{ queueError }}</div>
        <div v-else-if="queue.length === 0" class="panel-empty">
          <span class="empty-icon">📋</span>
          <p>今日暂无挂号</p>
        </div>
        <ul v-else class="queue-list">
          <li
            v-for="item in queue"
            :key="item.id"
            class="queue-item"
            :class="{ active: selectedId === item.id }"
            @click="selectRegistration(item.id)"
          >
            <div class="queue-time">{{ formatTimeOnly(item.appointment_time) }}</div>
            <div class="queue-body">
              <span class="queue-name">{{ patientNameOf(item) }}</span>
              <span class="purpose-tag">{{ purposeLabels[item.purpose] || item.purpose }}</span>
            </div>
            <p v-if="item.chief_complaint" class="queue-complaint">{{ item.chief_complaint }}</p>
            <span v-if="item.has_covid_symptoms" class="covid-flag">发热/新冠症状</span>
          </li>
        </ul>
      </aside>

      <!-- 右侧：患者详情 -->
      <main class="detail-panel">
        <div v-if="!selectedItem" class="detail-empty">
          <span class="empty-icon">👈</span>
          <p>请从左侧选择一位患者查看详情</p>
        </div>

        <template v-else>
          <div class="detail-header">
            <div class="patient-hero">
              <div class="avatar">{{ (selectedPatient?.name || '?')[0] }}</div>
              <div>
                <h2 class="patient-name">{{ selectedPatient?.name || '-' }}</h2>
                <p class="patient-meta">
                  {{ selectedPatient?.gender || '-' }}
                  · {{ calcAge(selectedPatient?.date_of_birth ?? null) }}
                  <template v-if="selectedPatient?.phone"> · {{ selectedPatient.phone }}</template>
                </p>
              </div>
            </div>
            <button class="consult-btn" @click="startConsultation">开始问诊</button>
          </div>

          <!-- 患者信息 -->
          <section class="info-card">
            <h3 class="card-title">患者信息</h3>
            <div class="info-grid">
              <div class="info-item"><span class="label">出生日期</span><span>{{ selectedPatient?.date_of_birth || '-' }}</span></div>
              <div class="info-item"><span class="label">身份证后5位</span><span>{{ selectedPatient?.id_card_last5 || '-' }}</span></div>
              <div class="info-item"><span class="label">联系地址</span><span>{{ selectedPatient?.address || '-' }}</span></div>
              <div class="info-item"><span class="label">紧急联系人</span><span>{{ selectedPatient?.emergency_contact_name || '-' }} {{ selectedPatient?.emergency_contact_phone || '' }}</span></div>
              <div class="info-item full"><span class="label">既往病史</span><span>{{ selectedPatient?.past_history || '-' }}</span></div>
              <div class="info-item full"><span class="label">过敏史</span><span>{{ selectedPatient?.allergy_history || '-' }}</span></div>
              <div class="info-item full"><span class="label">手术/外伤史</span><span>{{ selectedPatient?.surgery_history || '-' }}</span></div>
            </div>
          </section>

          <!-- 挂号信息 -->
          <section class="info-card reg-card">
            <h3 class="card-title">本次挂号</h3>
            <div class="reg-highlight">
              <div class="reg-time-block">
                <span class="reg-time">{{ formatTimeOnly(selectedItem.appointment_time) }}</span>
                <span class="reg-date">今日就诊</span>
              </div>
              <div class="reg-fields">
                <div><span class="label">科室</span>{{ selectedItem.department_name || doctorDept || '-' }}</div>
                <div><span class="label">就诊目的</span>{{ purposeLabels[selectedItem.purpose] || selectedItem.purpose }}</div>
                <div class="full"><span class="label">主诉</span>{{ selectedItem.chief_complaint || '-' }}</div>
              </div>
            </div>
          </section>

          <!-- 以往病例 -->
          <section class="info-card records-card">
            <h3 class="card-title">以往病例</h3>
            <div v-if="loadingRecords" class="panel-status">加载中…</div>
            <div v-else-if="recordsError" class="panel-error sm">{{ recordsError }}</div>
            <div v-else-if="medicalRecords.length === 0" class="records-empty">暂无历史病历</div>
            <div v-else class="records-list">
              <div v-for="rec in medicalRecords" :key="rec.id" class="record-item">
                <div class="record-head">
                  <span class="record-date">{{ rec.visit_date || '-' }}</span>
                  <span class="record-dept">{{ rec.department || '-' }}</span>
                </div>
                <p class="record-line"><strong>主诉</strong> {{ rec.chief_complaint || '-' }}</p>
                <p class="record-line"><strong>诊断</strong> {{ rec.diagnosis || '-' }}</p>
                <p v-if="rec.medication" class="record-line muted"><strong>用药</strong> {{ rec.medication }}</p>
              </div>
            </div>
          </section>
        </template>
      </main>
    </div>
  </div>
</template>

<style scoped>
.doctor-page {
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 2rem;
  text-align: left;
  min-height: 100vh;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.top-left {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.page-title {
  font-size: 1.35rem;
  margin: 0;
  font-weight: 600;
  color: var(--text-h);
}

.doctor-chip {
  font-size: 0.85rem;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.08);
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-weight: 500;
}

.top-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.date-chip {
  font-size: 0.82rem;
  color: var(--text);
  padding: 0.3rem 0.65rem;
  background: var(--code-bg);
  border-radius: 6px;
}

.refresh-btn,
.logout-btn {
  padding: 0.35rem 0.85rem;
  font-size: 0.82rem;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text-h);
  transition: background 0.15s;
}

.refresh-btn:hover:not(:disabled),
.logout-btn:hover {
  background: var(--code-bg);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.workspace {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 1.25rem;
  align-items: start;
  min-height: calc(100vh - 120px);
}

@media (max-width: 900px) {
  .workspace {
    grid-template-columns: 1fr;
  }
}

/* ---- 左侧列表 ---- */
.queue-panel {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  position: sticky;
  top: 1rem;
  max-height: calc(100vh - 100px);
  display: flex;
  flex-direction: column;
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 1rem;
  background: var(--code-bg);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.panel-head h2 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-h);
}

.count-badge {
  font-size: 0.75rem;
  font-weight: 600;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.1);
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
}

.panel-status,
.panel-empty,
.panel-error {
  padding: 2rem 1rem;
  text-align: center;
  font-size: 0.88rem;
  color: var(--text);
}

.panel-error {
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.06);
}

.panel-error.sm {
  padding: 0.75rem;
  text-align: left;
  border-radius: 6px;
  margin: 0.5rem 0;
}

.empty-icon {
  font-size: 1.8rem;
  display: block;
  margin-bottom: 0.5rem;
}

.queue-list {
  list-style: none;
  margin: 0;
  padding: 0.5rem;
  overflow-y: auto;
  flex: 1;
}

.queue-item {
  padding: 0.75rem 0.85rem;
  border-radius: 10px;
  cursor: pointer;
  border: 1px solid transparent;
  margin-bottom: 0.35rem;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
}

.queue-item:hover {
  background: var(--code-bg);
}

.queue-item.active {
  background: rgba(37, 99, 235, 0.06);
  border-color: rgba(37, 99, 235, 0.35);
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
}

.queue-time {
  font-size: 0.78rem;
  font-weight: 700;
  color: #2563eb;
  font-variant-numeric: tabular-nums;
  margin-bottom: 0.25rem;
}

.queue-body {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.queue-name {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--text-h);
}

.purpose-tag {
  font-size: 0.68rem;
  font-weight: 600;
  color: #7c3aed;
  background: rgba(124, 58, 237, 0.1);
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
}

.queue-complaint {
  margin: 0.35rem 0 0;
  font-size: 0.78rem;
  color: var(--text);
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.covid-flag {
  display: inline-block;
  margin-top: 0.35rem;
  font-size: 0.68rem;
  color: #b45309;
  background: rgba(245, 158, 11, 0.12);
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
}

/* ---- 右侧详情 ---- */
.detail-panel {
  min-width: 0;
}

.detail-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 360px;
  border: 1px dashed var(--border);
  border-radius: 12px;
  color: var(--text);
  font-size: 0.9rem;
}

.detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.patient-hero {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.avatar {
  width: 3rem;
  height: 3rem;
  border-radius: 12px;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  font-weight: 700;
  flex-shrink: 0;
}

.patient-name {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--text-h);
}

.patient-meta {
  margin: 0.2rem 0 0;
  font-size: 0.85rem;
  color: var(--text);
}

.consult-btn {
  padding: 0.55rem 1.4rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
  transition: opacity 0.2s, transform 0.15s;
  white-space: nowrap;
}

.consult-btn:hover {
  opacity: 0.92;
  transform: translateY(-1px);
}

.info-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.15rem;
  margin-bottom: 1rem;
}

.card-title {
  margin: 0 0 0.75rem;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text-h);
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem 1.25rem;
}

@media (max-width: 600px) {
  .info-grid {
    grid-template-columns: 1fr;
  }
}

.info-item {
  font-size: 0.84rem;
  color: var(--text-h);
  line-height: 1.45;
}

.info-item.full {
  grid-column: 1 / -1;
}

.label {
  display: block;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.15rem;
  text-transform: none;
}

.reg-highlight {
  display: flex;
  gap: 1.25rem;
  flex-wrap: wrap;
}

.reg-time-block {
  flex-shrink: 0;
  text-align: center;
  padding: 0.75rem 1rem;
  background: rgba(37, 99, 235, 0.06);
  border-radius: 10px;
  border: 1px solid rgba(37, 99, 235, 0.15);
  min-width: 100px;
}

.reg-time {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: #2563eb;
  font-variant-numeric: tabular-nums;
}

.reg-date {
  font-size: 0.72rem;
  color: var(--text);
}

.reg-fields {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem 1rem;
  font-size: 0.84rem;
  color: var(--text-h);
  min-width: 200px;
}

.reg-fields .full {
  grid-column: 1 / -1;
}

.reg-fields .label {
  display: inline;
  margin-right: 0.35rem;
}

.records-empty {
  font-size: 0.85rem;
  color: var(--text);
  padding: 0.5rem 0;
}

.records-list {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.record-item {
  padding: 0.75rem 0.85rem;
  background: var(--code-bg);
  border-radius: 8px;
  border-left: 3px solid #6366f1;
}

.record-head {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
  flex-wrap: wrap;
}

.record-date {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-h);
}

.record-dept {
  font-size: 0.75rem;
  color: var(--text);
  padding: 0.05rem 0.4rem;
  background: var(--bg);
  border-radius: 4px;
}

.record-line {
  margin: 0.2rem 0;
  font-size: 0.8rem;
  color: var(--text-h);
  line-height: 1.4;
}

.record-line.muted {
  color: var(--text);
}

.record-line strong {
  font-weight: 600;
  color: var(--text);
  margin-right: 0.25rem;
}
</style>
