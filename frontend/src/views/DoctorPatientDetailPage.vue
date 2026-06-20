<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { supabase } from '../lib/supabase'
import { useApiError } from '../composables/useApiError'
import { ensureValidSession, runQuery } from '../lib/session'

const router = useRouter()
const route = useRoute()
const { resolveError } = useApiError()

const patientId = (route.query.patientId as string) || ''
const patientNameHint = (route.query.patientName as string) || ''

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
  doctor_name: string
}

const patient = ref<Patient | null>(null)
const medicalRecords = ref<MedicalRecord[]>([])
const loading = ref(false)
const errorMsg = ref('')
const expandedRecordId = ref<string | null>(null)

const displayName = computed(() => patient.value?.name || patientNameHint || '患者')

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

function toggleRecord(id: string) {
  expandedRecordId.value = expandedRecordId.value === id ? null : id
}

function goBack() {
  if (route.path.startsWith('/nurse')) {
    router.push('/nurse')
  } else {
    router.push('/doctor')
  }
}

async function loadData() {
  if (!patientId) {
    errorMsg.value = '缺少患者信息'
    return
  }
  loading.value = true
  errorMsg.value = ''
  try {
    await ensureValidSession()
    const [patientRes, recordsRes] = await Promise.all([
      runQuery(
        supabase.from('patients').select('*').eq('id', patientId).maybeSingle(),
        '加载患者信息',
      ),
      runQuery(
        supabase
          .from('medical_records')
          .select('*')
          .eq('patient_id', patientId)
          .order('visit_date', { ascending: false }),
        '加载以往病例',
      ),
    ])
    if (patientRes.error) throw patientRes.error
    if (recordsRes.error) throw recordsRes.error
    if (!patientRes.data) throw new Error('未找到该患者')
    patient.value = patientRes.data as Patient
    medicalRecords.value = (recordsRes.data as MedicalRecord[]) || []
  } catch (e) {
    patient.value = null
    medicalRecords.value = []
    errorMsg.value = await resolveError(e, '加载患者信息失败')
  } finally {
    loading.value = false
  }
}

onMounted(loadData)
</script>

<template>
  <div class="detail-page">
    <header class="top-bar">
      <button class="back-btn" @click="goBack">&larr; 返回工作台</button>
      <div v-if="patient" class="patient-hero">
        <div class="avatar">{{ patient.name[0] }}</div>
        <div>
          <h1 class="page-title">{{ patient.name }}</h1>
          <p class="patient-meta">
            {{ patient.gender || '-' }} · {{ calcAge(patient.date_of_birth) }}
            <template v-if="patient.phone"> · {{ patient.phone }}</template>
          </p>
        </div>
      </div>
      <h1 v-else class="page-title">{{ displayName }}</h1>
    </header>

    <div v-if="loading" class="status-box">加载中…</div>
    <div v-else-if="errorMsg" class="error-box">{{ errorMsg }}</div>
    <template v-else-if="patient">
      <section class="info-card">
        <h2 class="card-title">基本信息</h2>
        <div class="info-grid">
          <div class="info-item"><span class="label">出生日期</span><span>{{ patient.date_of_birth || '-' }}</span></div>
          <div class="info-item"><span class="label">身份证后5位</span><span>{{ patient.id_card_last5 || '-' }}</span></div>
          <div class="info-item"><span class="label">手机号</span><span>{{ patient.phone || '-' }}</span></div>
          <div class="info-item"><span class="label">联系地址</span><span>{{ patient.address || '-' }}</span></div>
          <div class="info-item"><span class="label">紧急联系人</span><span>{{ patient.emergency_contact_name || '-' }}</span></div>
          <div class="info-item"><span class="label">紧急联系电话</span><span>{{ patient.emergency_contact_phone || '-' }}</span></div>
          <div class="info-item full"><span class="label">既往病史</span><span>{{ patient.past_history || '-' }}</span></div>
          <div class="info-item full"><span class="label">过敏史</span><span>{{ patient.allergy_history || '-' }}</span></div>
          <div class="info-item full"><span class="label">手术/外伤史</span><span>{{ patient.surgery_history || '-' }}</span></div>
        </div>
      </section>

      <section class="info-card records-card">
        <h2 class="card-title">以往病例（{{ medicalRecords.length }}）</h2>
        <div v-if="medicalRecords.length === 0" class="records-empty">暂无历史病历</div>
        <div v-else class="records-list">
          <article v-for="rec in medicalRecords" :key="rec.id" class="record-item">
            <div class="record-head">
              <span class="record-date">{{ rec.visit_date || '-' }}</span>
              <span class="record-dept">{{ rec.department || '-' }}</span>
              <span class="record-doctor">{{ rec.doctor_name || '-' }}</span>
              <button class="expand-btn" @click="toggleRecord(rec.id)">
                {{ expandedRecordId === rec.id ? '收起' : '详情' }}
              </button>
            </div>
            <p class="record-line"><strong>主诉</strong> {{ rec.chief_complaint || '-' }}</p>
            <p class="record-line"><strong>诊断</strong> {{ rec.diagnosis || '-' }}</p>
            <div v-if="expandedRecordId === rec.id" class="record-detail">
              <p v-if="rec.present_illness"><strong>现病史</strong> {{ rec.present_illness }}</p>
              <p v-if="rec.past_history"><strong>既往史</strong> {{ rec.past_history }}</p>
              <p v-if="rec.physical_exam"><strong>体格检查</strong> {{ rec.physical_exam }}</p>
              <p v-if="rec.temperature || rec.pulse || rec.respiration || rec.blood_pressure">
                <strong>生命体征</strong>
                <template v-if="rec.temperature"> T {{ rec.temperature }}℃</template>
                <template v-if="rec.pulse"> P {{ rec.pulse }}次/分</template>
                <template v-if="rec.respiration"> R {{ rec.respiration }}次/分</template>
                <template v-if="rec.blood_pressure"> BP {{ rec.blood_pressure }}</template>
              </p>
              <p v-if="rec.auxiliary_exam"><strong>辅助检查</strong> {{ rec.auxiliary_exam }}</p>
              <p v-if="rec.medication"><strong>用药</strong> {{ rec.medication }}</p>
              <p v-if="rec.medical_advice"><strong>医嘱</strong> {{ rec.medical_advice }}</p>
            </div>
          </article>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.detail-page {
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 2rem;
  text-align: left;
  min-height: 100vh;
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
  padding: 0.35rem 0.85rem;
  font-size: 0.82rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-h);
  cursor: pointer;
}
.back-btn:hover { background: var(--code-bg); color: #2563eb; }

.patient-hero {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex: 1;
}

.avatar {
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 10px;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  font-weight: 700;
}

.page-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-h);
}

.patient-meta {
  margin: 0.2rem 0 0;
  font-size: 0.85rem;
  color: var(--text);
}

.status-box, .records-empty {
  text-align: center;
  padding: 2rem;
  color: var(--text);
  font-size: 0.9rem;
}

.error-box {
  padding: 0.85rem 1rem;
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 8px;
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.info-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
}

.card-title {
  margin: 0 0 0.85rem;
  font-size: 0.95rem;
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

@media (max-width: 700px) {
  .info-grid { grid-template-columns: 1fr; }
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  font-size: 0.85rem;
  color: var(--text-h);
}

.info-item.full { grid-column: 1 / -1; }

.label {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.records-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.record-item {
  padding: 0.85rem;
  background: var(--code-bg);
  border-radius: 8px;
  border-left: 3px solid #6366f1;
}

.record-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.4rem;
}

.record-date {
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--text-h);
}

.record-dept, .record-doctor {
  font-size: 0.72rem;
  padding: 0.1rem 0.4rem;
  background: var(--bg);
  border-radius: 4px;
  color: var(--text);
}

.expand-btn {
  margin-left: auto;
  padding: 0.15rem 0.55rem;
  font-size: 0.72rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg);
  color: #2563eb;
  cursor: pointer;
}

.record-line {
  margin: 0.25rem 0;
  font-size: 0.84rem;
  line-height: 1.45;
  color: var(--text-h);
}

.record-line strong {
  color: var(--text);
  font-weight: 600;
  margin-right: 0.25rem;
}

.record-detail {
  margin-top: 0.6rem;
  padding-top: 0.6rem;
  border-top: 1px dashed var(--border);
  font-size: 0.8rem;
  line-height: 1.5;
  color: var(--text);
}

.record-detail p { margin: 0.3rem 0; }
.record-detail strong { color: var(--text-h); margin-right: 0.25rem; }
</style>
