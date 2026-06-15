<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { supabase } from '../lib/supabase'
import { useAuth } from '../composables/useAuth'
import { useApiError } from '../composables/useApiError'
import { ensureValidSession, runQuery } from '../lib/session'
import {
  type RegistrationRow,
  purposeLabels,
  statusLabels,
  normalizeRegistrationStatus,
  isAppointmentToday,
  formatDateTime,
  formatTimeOnly,
  nowBeijingISO,
} from '../lib/registration'

const router = useRouter()
const route = useRoute()
const { user } = useAuth()
const { resolveError } = useApiError()

const patientId = (route.query.patientId as string) || ''
const patientName = (route.query.patientName as string) || '患者'

const list = ref<RegistrationRow[]>([])
const loading = ref(false)
const listError = ref('')
const cancellingId = ref<string | null>(null)

const showForm = ref(false)
const formError = ref('')
const saving = ref(false)
const editingId = ref<string | null>(null)
const doctors = ref<{ id: string; name: string; department: string }[]>([])

const form = reactive({
  doctorId: '',
  chiefComplaint: '',
  purpose: 'initial',
  hasCovidSymptoms: false,
})

const isEditing = computed(() => editingId.value !== null)

const todayLabel = computed(() =>
  new Date().toLocaleDateString('zh-CN', { month: 'long', day: 'numeric', weekday: 'short' }),
)

async function loadList() {
  if (!patientId) return
  loading.value = true
  listError.value = ''
  try {
    await ensureValidSession()
    const { data, error } = await runQuery(
      supabase
        .from('registrations')
        .select('id, patient_id, doctor_id, department_name, appointment_time, chief_complaint, purpose, has_covid_symptoms, status, doctors(name, department)')
        .eq('patient_id', patientId)
        .order('appointment_time', { ascending: true }),
      '加载挂号列表',
    )
    if (error) throw error
    list.value = ((data as unknown as RegistrationRow[]) || []).filter(r =>
      isAppointmentToday(r.appointment_time),
    )
  } catch (e) {
    list.value = []
    listError.value = await resolveError(e, '加载挂号列表失败')
  } finally {
    loading.value = false
  }
}

async function loadDoctors() {
  const { data, error } = await runQuery(
    supabase.from('doctors').select('id,name,department').order('name'),
    '加载医生列表',
    10000,
  )
  if (error) throw error
  doctors.value = (data as any) || []
}

function goBack() {
  router.push('/nurse')
}

function openCreate() {
  formError.value = ''
  editingId.value = null
  form.doctorId = ''
  form.chiefComplaint = ''
  form.purpose = 'initial'
  form.hasCovidSymptoms = false
  showForm.value = true
  loadDoctors()
}

function openEdit(reg: RegistrationRow) {
  if (normalizeRegistrationStatus(reg.status) !== 'waiting') return
  formError.value = ''
  editingId.value = reg.id
  form.doctorId = reg.doctor_id
  form.chiefComplaint = reg.chief_complaint
  form.purpose = reg.purpose
  form.hasCovidSymptoms = reg.has_covid_symptoms ?? false
  showForm.value = true
  loadDoctors()
}

function closeForm() {
  showForm.value = false
  editingId.value = null
  formError.value = ''
}

async function submitForm() {
  formError.value = ''
  if (!form.doctorId || !form.chiefComplaint.trim()) {
    formError.value = '请填写必填项：医生、主诉'
    return
  }
  saving.value = true
  const editId = editingId.value
  try {
    await ensureValidSession()
    const selectedDoctor = doctors.value.find(d => d.id === form.doctorId)
    const record = {
      doctor_id: form.doctorId,
      department_name: selectedDoctor?.department || '',
      appointment_time: nowBeijingISO(),
      chief_complaint: form.chiefComplaint.trim(),
      purpose: form.purpose,
      has_covid_symptoms: form.hasCovidSymptoms,
    }

    if (editId) {
      const { error } = await runQuery(
        supabase.from('registrations').update(record).eq('id', editId),
        '更新挂号',
      )
      if (error) throw error
    } else {
      const authId = user.value?.id
      if (!authId) throw new Error('未登录')
      const { data: nurseData, error: nurseErr } = await runQuery(
        supabase.from('nurses').select('id').eq('auth_id', authId).maybeSingle(),
        '查询护士身份',
        10000,
      )
      if (nurseErr || !nurseData) throw new Error('无法识别护士身份')

      const { error } = await runQuery(
        supabase.from('registrations').insert([{
          ...record,
          patient_id: patientId,
          nurse_id: nurseData.id,
          status: 'waiting',
        }]),
        '创建挂号',
      )
      if (error) throw error
    }
    closeForm()
    await loadList()
  } catch (e) {
    formError.value = await resolveError(e, editId ? '更新失败' : '挂号失败')
  } finally {
    saving.value = false
  }
}

async function cancelRegistration(reg: RegistrationRow) {
  if (normalizeRegistrationStatus(reg.status) !== 'waiting') return
  const time = formatTimeOnly(reg.appointment_time)
  const doctor = reg.doctors?.name || '未知医生'
  if (!confirm(`确定取消 ${patientName} 今日 ${time}（${doctor}）的挂号吗？`)) return

  cancellingId.value = reg.id
  try {
    await ensureValidSession()
    const { error } = await runQuery(
      supabase.from('registrations').update({ status: 'cancelled' }).eq('id', reg.id),
      '取消挂号',
      10000,
    )
    if (error) throw error
    await loadList()
  } catch (e) {
    alert(await resolveError(e, '取消失败'))
  } finally {
    cancellingId.value = null
  }
}

onMounted(() => {
  if (!patientId) {
    listError.value = '缺少患者信息'
    return
  }
  loadList()
})
</script>

<template>
  <div class="reg-page">
    <header class="top-bar">
      <button class="back-btn" @click="goBack">&larr; 返回查询</button>
      <div class="title-block">
        <h1>挂号处理</h1>
        <p class="subtitle">{{ patientName }} · {{ todayLabel }}</p>
      </div>
      <button class="add-btn" @click="openCreate">+ 新增挂号</button>
    </header>

    <div v-if="loading" class="status-box">加载中…</div>
    <div v-else-if="listError" class="error-box">{{ listError }}</div>
    <div v-else-if="list.length === 0" class="empty-box">
      <span class="empty-icon">📋</span>
      <p>今日暂无挂号记录</p>
      <button class="add-btn-inline" @click="openCreate">立即挂号</button>
    </div>

    <div v-else class="reg-list">
      <article
        v-for="(reg, idx) in list"
        :key="reg.id"
        class="reg-card"
        :class="`status-${normalizeRegistrationStatus(reg.status)}`"
      >
        <div class="reg-card-head">
          <span class="reg-index">{{ idx + 1 }}</span>
          <span class="reg-time">{{ formatDateTime(reg.appointment_time) }}</span>
          <span class="status-pill" :class="normalizeRegistrationStatus(reg.status)">
            {{ statusLabels[normalizeRegistrationStatus(reg.status)] }}
          </span>
        </div>
        <div class="reg-card-body">
          <div class="field-row">
            <span class="label">医生</span>
            <span>{{ reg.doctors?.name || '未知' }} · {{ reg.department_name || reg.doctors?.department || '-' }}</span>
          </div>
          <div class="field-row">
            <span class="label">目的</span>
            <span class="purpose-tag">{{ purposeLabels[reg.purpose] || reg.purpose }}</span>
          </div>
          <div class="field-row">
            <span class="label">主诉</span>
            <span>{{ reg.chief_complaint || '-' }}</span>
          </div>
          <div v-if="reg.has_covid_symptoms" class="covid-tag">发热/新冠相关症状</div>
        </div>
        <div v-if="normalizeRegistrationStatus(reg.status) === 'waiting'" class="reg-card-actions">
          <button class="btn-edit" @click="openEdit(reg)">编辑</button>
          <button
            class="btn-cancel"
            :disabled="cancellingId === reg.id"
            @click="cancelRegistration(reg)"
          >{{ cancellingId === reg.id ? '处理中…' : '取消' }}</button>
        </div>
      </article>
    </div>

    <!-- 挂号弹窗 -->
    <div v-if="showForm" class="modal-overlay" @click.self="closeForm">
      <div class="modal">
        <div class="modal-head">
          <h2>{{ isEditing ? '编辑挂号' : '新增挂号' }}</h2>
          <button class="close-btn" @click="closeForm">&times;</button>
        </div>
        <p class="modal-sub">{{ patientName }} · 就诊时间将自动记录为当前北京时间</p>
        <div v-if="formError" class="form-error">{{ formError }}</div>
        <div class="form-body">
          <label class="field">
            <span class="field-label">选择医生 *</span>
            <select v-model="form.doctorId" class="field-input">
              <option value="">请选择</option>
              <option v-for="d in doctors" :key="d.id" :value="d.id">{{ d.name }} ({{ d.department }})</option>
            </select>
          </label>
          <label class="field">
            <span class="field-label">主诉 *</span>
            <textarea v-model="form.chiefComplaint" class="field-textarea" rows="3" />
          </label>
          <label class="field">
            <span class="field-label">就诊目的 *</span>
            <select v-model="form.purpose" class="field-input">
              <option value="initial">初诊</option>
              <option value="follow_up">复查</option>
              <option value="medication">开药</option>
              <option value="consultation">咨询</option>
              <option value="chronic_follow_up">慢性病复诊</option>
            </select>
          </label>
          <label class="field check-row">
            <input v-model="form.hasCovidSymptoms" type="checkbox" />
            <span>有无发热/新冠相关症状</span>
          </label>
        </div>
        <div class="modal-foot">
          <button class="btn-ghost" @click="closeForm">取消</button>
          <button class="btn-primary" :disabled="saving" @click="submitForm">
            {{ saving ? '保存中…' : (isEditing ? '保存' : '确认挂号') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.reg-page {
  max-width: 720px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 2rem;
  text-align: left;
}

.top-bar {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.back-btn {
  padding: 0.35rem 0.75rem;
  font-size: 0.85rem;
  color: var(--text);
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
}
.back-btn:hover { background: var(--code-bg); }

.title-block { flex: 1; min-width: 160px; }
.title-block h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-h);
}
.subtitle {
  margin: 0.2rem 0 0;
  font-size: 0.85rem;
  color: var(--text);
}

.add-btn, .add-btn-inline {
  padding: 0.5rem 1.1rem;
  font-size: 0.88rem;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
}
.add-btn:hover, .add-btn-inline:hover { opacity: 0.92; }

.status-box, .empty-box {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--text);
  font-size: 0.9rem;
}
.empty-icon { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
.empty-box p { margin: 0 0 1rem; }

.error-box {
  padding: 0.85rem 1rem;
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 8px;
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.reg-list {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.reg-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  border-left: 4px solid #3b82f6;
  transition: box-shadow 0.15s;
}
.reg-card:hover { box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); }
.reg-card.status-cancelled {
  opacity: 0.72;
  border-left-color: #9ca3af;
}
.reg-card.status-completed {
  border-left-color: #10b981;
}

.reg-card-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.65rem;
}

.reg-index {
  width: 1.4rem;
  height: 1.4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 700;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.1);
  border-radius: 50%;
}

.reg-time {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text-h);
  font-variant-numeric: tabular-nums;
}

.status-pill {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
}
.status-pill.waiting { color: #2563eb; background: rgba(37, 99, 235, 0.1); }
.status-pill.cancelled { color: #6b7280; background: rgba(107, 114, 128, 0.12); }
.status-pill.completed { color: #059669; background: rgba(16, 185, 129, 0.12); }

.reg-card-body { font-size: 0.85rem; color: var(--text-h); }
.field-row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
  line-height: 1.4;
}
.field-row .label {
  flex-shrink: 0;
  width: 2.5rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text);
}

.purpose-tag {
  font-size: 0.75rem;
  font-weight: 600;
  color: #7c3aed;
  background: rgba(124, 58, 237, 0.1);
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
}

.covid-tag {
  display: inline-block;
  margin-top: 0.25rem;
  font-size: 0.72rem;
  color: #b45309;
  background: rgba(245, 158, 11, 0.12);
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
}

.reg-card-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
}

.btn-edit, .btn-cancel {
  padding: 0.35rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid transparent;
}
.btn-edit {
  color: #2563eb;
  background: rgba(37, 99, 235, 0.08);
  border-color: rgba(37, 99, 235, 0.25);
}
.btn-cancel {
  color: #dc2626;
  background: rgba(239, 68, 68, 0.06);
  border-color: rgba(239, 68, 68, 0.25);
}
.btn-cancel:disabled { opacity: 0.5; cursor: not-allowed; }

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
}
.modal {
  background: var(--bg);
  border-radius: 12px;
  width: 100%;
  max-width: 440px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2);
}
.modal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.2rem 0;
}
.modal-head h2 { margin: 0; font-size: 1.05rem; }
.close-btn { background: none; border: none; font-size: 1.4rem; cursor: pointer; color: var(--text); }
.modal-sub {
  margin: 0.25rem 1.2rem 0;
  font-size: 0.85rem;
  color: var(--text);
}
.form-error {
  margin: 0.75rem 1.2rem 0;
  padding: 0.55rem 0.75rem;
  font-size: 0.85rem;
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 6px;
}
.form-body { padding: 1rem 1.2rem; display: flex; flex-direction: column; gap: 0.75rem; }
.field { display: flex; flex-direction: column; gap: 0.3rem; }
.check-row { flex-direction: row; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
.field-label { font-size: 0.82rem; font-weight: 600; color: var(--text-h); }
.field-input, .field-textarea {
  padding: 0.5rem 0.7rem;
  font-size: 0.9rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-h);
}
.field-textarea { resize: vertical; font-family: inherit; }
.modal-foot {
  display: flex;
  justify-content: flex-end;
  gap: 0.6rem;
  padding: 0.85rem 1.2rem;
  border-top: 1px solid var(--border);
}
.btn-ghost {
  padding: 0.45rem 1rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  cursor: pointer;
  color: var(--text-h);
}
.btn-primary {
  padding: 0.45rem 1.1rem;
  font-weight: 600;
  color: #fff;
  background: #2563eb;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
.btn-primary:disabled { opacity: 0.5; }
</style>
