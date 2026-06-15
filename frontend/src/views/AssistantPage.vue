<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { supabase } from '../lib/supabase'
import { useApiError } from '../composables/useApiError'
import { useAuth } from '../composables/useAuth'
import { ensureValidSession, runQuery } from '../lib/session'
import WhisperTranscriber from '../components/WhisperTranscriber.vue'

const router = useRouter()
const route = useRoute()
const { resolveError } = useApiError()
const { role, user } = useAuth()

const patientId = (route.query.patientId as string) || ''
const registrationId = computed(() => (route.query.registrationId as string) || '')
const patientName = (route.query.patientName as string) || ''
const patientGender = (route.query.patientGender as string) || ''
const patientDateOfBirth = (route.query.patientDateOfBirth as string) || ''

/** 从出生日期计算年龄 */
function calcAge(dateOfBirth: string): number | null {
  if (!dateOfBirth) return null
  const birth = new Date(dateOfBirth)
  if (isNaN(birth.getTime())) return null
  const today = new Date()
  let age = today.getFullYear() - birth.getFullYear()
  const m = today.getMonth() - birth.getMonth()
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) {
    age--
  }
  return age
}

const patientAge = calcAge(patientDateOfBirth)

// ---- 以往病例 ----
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

const medicalRecords = ref<MedicalRecord[]>([])
const fetchingRecords = ref(false)
const recordsError = ref('')
const expandedRecordId = ref<string | null>(null)

async function fetchMedicalRecords() {
  if (!patientId) return
  fetchingRecords.value = true
  recordsError.value = ''
  try {
    await ensureValidSession()
    const { data, error } = await runQuery(
      supabase.from('medical_records').select('*').eq('patient_id', patientId).order('visit_date', { ascending: false }),
      '加载以往病例',
    )
    if (error) throw error
    medicalRecords.value = (data as MedicalRecord[]) || []
  } catch (e) {
    console.error('Fetch medical records error:', e)
    medicalRecords.value = []
    recordsError.value = await resolveError(e, '加载以往病例失败，请稍后重试')
  } finally {
    fetchingRecords.value = false
  }
}

function toggleRecordExpand(id: string) {
  expandedRecordId.value = expandedRecordId.value === id ? null : id
}

onMounted(() => {
  fetchMedicalRecords()
})

// ---- 转写组件引用 ----
const transcriberRef = ref<InstanceType<typeof WhisperTranscriber> | null>(null)

// 只有意见判定「信息充足」时才允许写入
const canWrite = computed(() => {
  const advice = transcriberRef.value?.adviceText || ''
  return advice.includes('信息充足')
})

// ---- 写入数据 ----
const extracting = ref(false)
const extractError = ref('')
const showReviewModal = ref(false)
const saving = ref(false)
const saveError = ref('')
const saveSuccess = ref(false)

interface MedicalRecordForm {
  department: string
  chief_complaint: string
  present_illness: string
  past_history: string
  physical_exam: string
  temperature: string
  pulse: string
  respiration: string
  blood_pressure: string
  auxiliary_exam: string
  diagnosis: string
  medication: string
  medical_advice: string
  rest_days: string
  doctor_name: string
}

const recordForm = ref<MedicalRecordForm>({
  department: '',
  chief_complaint: '',
  present_illness: '',
  past_history: '',
  physical_exam: '',
  temperature: '',
  pulse: '',
  respiration: '',
  blood_pressure: '',
  auxiliary_exam: '',
  diagnosis: '',
  medication: '',
  medical_advice: '',
  rest_days: '',
  doctor_name: '',
})

const DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions'

async function extractMedicalRecord() {
  const list = transcriberRef.value?.transcriptionList
  if (!list || list.length === 0) {
    extractError.value = '暂无转写内容可提取'
    return
  }

  extractError.value = ''
  extracting.value = true
  saveSuccess.value = false

  const transcript = list.map((t: any) => `${t.speaker}：${t.text}`).join('\n')
  const adviceRef = transcriberRef.value?.adviceText || ''

  const systemPrompt = `你是专业的病历结构化助手。根据医患对话转写文本及医生助手的分析意见，提取以下字段并以 JSON 返回。

优先从转写对话中提取客观事实；若对话未明确提及但医生助手分析意见（「治疗方案」「初步诊断」等章节）中有合理的临床建议，可将其作为参考填入对应字段。不编造信息，确实未提及的字段返回空字符串。

返回格式（严格 JSON，不要 markdown 代码块包裹）：
{
  "department": "就诊科室",
  "chief_complaint": "主诉（症状+持续时间，例如：腹痛2天）",
  "present_illness": "现病史（起病情况、症状演变、伴随症状、诊治经过）",
  "past_history": "既往史（慢性病、手术史、过敏史、家族史等）",
  "physical_exam": "体格检查（T、P、R、BP、心肺腹查体等发现）",
  "temperature": "体温数值（如 36.5，不要单位）",
  "pulse": "脉搏/心率数值",
  "respiration": "呼吸频率数值",
  "blood_pressure": "血压（如 120/80）",
  "auxiliary_exam": "辅助检查结果",
  "diagnosis": "诊断或初步诊断（优先取分析意见中的诊断）",
  "medication": "用药（药品名+用法用量；如分析意见的「治疗方案-药物治疗」中有推荐，务必提取）",
  "medical_advice": "医嘱/处置建议（含非药物治疗、随访建议等；优先综合「治疗方案」章节）",
  "rest_days": "建议休息天数（仅数字）",
  "doctor_name": "医生姓名"
}

要求：充分利用分析意见中的临床建议填充各字段，不要遗漏治疗方案中的用药和医嘱信息；数值类字段只写数字；体征未述及时留空。`

  const userContent = `以下为医患对话转写文本：\n\n${transcript}\n\n以下为医生助手的初步分析意见，可作为字段填充参考：\n\n${adviceRef}\n\n请从以上内容中提取病历信息。`

  try {
    const apiKey = import.meta.env.VITE_DEEPSEEK_API_KEY?.trim()
    if (!apiKey) throw new Error('未配置 DeepSeek API Key')

    const res = await fetch(DEEPSEEK_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'deepseek-chat',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
        ],
        temperature: 0.1,
      }),
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const data = await res.json()
    const raw = data.choices?.[0]?.message?.content?.trim()
    if (!raw) throw new Error('AI 未返回有效内容')

    // 解析 JSON（可能带有 markdown 代码块包裹）
    let jsonStr = raw
    const jsonMatch = raw.match(/\{[\s\S]*\}/)
    if (jsonMatch) jsonStr = jsonMatch[0]

    const parsed = JSON.parse(jsonStr)

    recordForm.value = {
      department: parsed.department || '',
      chief_complaint: parsed.chief_complaint || '',
      present_illness: parsed.present_illness || '',
      past_history: parsed.past_history || '',
      physical_exam: parsed.physical_exam || '',
      temperature: parsed.temperature != null ? String(parsed.temperature) : '',
      pulse: parsed.pulse != null ? String(parsed.pulse) : '',
      respiration: parsed.respiration != null ? String(parsed.respiration) : '',
      blood_pressure: parsed.blood_pressure || '',
      auxiliary_exam: parsed.auxiliary_exam || '',
      diagnosis: parsed.diagnosis || '',
      medication: parsed.medication || '',
      medical_advice: parsed.medical_advice || '',
      rest_days: parsed.rest_days != null ? String(parsed.rest_days) : '',
      doctor_name: parsed.doctor_name || '',
    }

    showReviewModal.value = true
  } catch (e: any) {
    extractError.value = e.message || '提取失败'
  } finally {
    extracting.value = false
  }
}

async function resolveRegistrationId(): Promise<string | null> {
  if (registrationId.value) return registrationId.value
  if (!patientId || role.value !== 'doctor' || !user.value?.id) return null

  const { data: doctorData, error: doctorErr } = await runQuery(
    supabase.from('doctors').select('id').eq('auth_id', user.value.id).maybeSingle(),
    '查询医生身份',
    10000,
  )
  if (doctorErr || !doctorData?.id) return null

  const { data: regs, error: regErr } = await runQuery(
    supabase
      .from('registrations')
      .select('id')
      .eq('patient_id', patientId)
      .eq('doctor_id', doctorData.id)
      .eq('status', 'waiting')
      .order('appointment_time', { ascending: false })
      .limit(1),
    '查询待完成挂号',
    10000,
  )
  if (regErr || !regs?.length) return null
  return (regs[0] as { id: string }).id
}

async function completeRegistration(regId: string): Promise<void> {
  const { data, error } = await runQuery(
    supabase
      .from('registrations')
      .update({ status: 'completed' })
      .eq('id', regId)
      .eq('status', 'waiting')
      .select('id, status')
      .maybeSingle(),
    '更新挂号状态',
    10000,
  )
  if (error) throw error
  if (!data) {
    throw new Error('挂号状态更新失败：记录不存在、已完成，或当前账号无权更新')
  }
}

async function confirmSave() {
  if (!patientId) {
    saveError.value = '缺少病人 ID，无法保存'
    return
  }
  if (!recordForm.value.chief_complaint.trim()) {
    saveError.value = '主诉为必填项'
    return
  }
  if (!recordForm.value.diagnosis.trim()) {
    saveError.value = '诊断为必填项'
    return
  }

  saveError.value = ''
  saving.value = true

  try {
    await ensureValidSession()
    const { error } = await runQuery(
      supabase.from('medical_records').insert({
        patient_id: patientId,
        department: recordForm.value.department || '未指定',
        visit_date: new Date().toISOString().split('T')[0],
        chief_complaint: recordForm.value.chief_complaint,
        present_illness: recordForm.value.present_illness || null,
        past_history: recordForm.value.past_history || null,
        physical_exam: recordForm.value.physical_exam || null,
        temperature: recordForm.value.temperature ? parseFloat(recordForm.value.temperature) : null,
        pulse: recordForm.value.pulse ? parseInt(recordForm.value.pulse) : null,
        respiration: recordForm.value.respiration ? parseInt(recordForm.value.respiration) : null,
        blood_pressure: recordForm.value.blood_pressure || null,
        auxiliary_exam: recordForm.value.auxiliary_exam || null,
        diagnosis: recordForm.value.diagnosis,
        medication: recordForm.value.medication || null,
        medical_advice: recordForm.value.medical_advice || null,
        rest_days: recordForm.value.rest_days ? parseInt(recordForm.value.rest_days) : null,
        doctor_name: recordForm.value.doctor_name || '未指定',
      }),
      '保存病历',
    )

    if (error) {
      saveError.value = await resolveError(error, '保存病历失败')
      return
    }

    const regId = await resolveRegistrationId()
    if (regId) {
      await completeRegistration(regId)
    }

    saveSuccess.value = true
    showReviewModal.value = false
    fetchMedicalRecords()

    const patient = await fetchPatientDetail()
    if (patient) {
      printMedicalRecord(patient, recordForm.value)
    }
  } catch (e) {
    saveError.value = await resolveError(e, '保存病历失败')
  } finally {
    saving.value = false
  }
}

function goBack() {
  if (route.query.from === 'doctor' || role.value === 'doctor') {
    router.push('/doctor')
  } else {
    router.push('/')
  }
}

// ---- 患者完整信息（用于打印病历）----
interface PatientDetail {
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

async function fetchPatientDetail(): Promise<PatientDetail | null> {
  if (!patientId) return null
  const { data, error } = await supabase
    .from('patients')
    .select('name, gender, date_of_birth, phone, id_card_last5, past_history, allergy_history, surgery_history, emergency_contact_name, emergency_contact_phone, address')
    .eq('id', patientId)
    .single()
  if (error || !data) return null
  return data as PatientDetail
}

function printMedicalRecord(patient: PatientDetail, record: MedicalRecordForm) {
  const age = patient.date_of_birth ? calcAge(patient.date_of_birth) : null
  const today = new Date().toISOString().split('T')[0]

  const vitals: string[] = []
  if (record.temperature) vitals.push('T ' + record.temperature + '℃')
  if (record.pulse) vitals.push('P ' + record.pulse + '次/分')
  if (record.respiration) vitals.push('R ' + record.respiration + '次/分')
  if (record.blood_pressure) vitals.push('BP ' + record.blood_pressure)

  const patientName = patient.name || '-'
  const patientGender = patient.gender || '-'
  const ageText = age != null ? age + ' 岁' : '-'
  const phone = patient.phone || '-'
  const idCard = patient.id_card_last5 || '-'
  const dob = patient.date_of_birth || '-'
  const addr = patient.address || '-'
  let emergency = '-'
  if (patient.emergency_contact_name) {
    emergency = patient.emergency_contact_name
    if (patient.emergency_contact_phone) emergency += ' / ' + patient.emergency_contact_phone
  }
  const pHistory = patient.past_history || '-'
  const allergy = patient.allergy_history || '-'
  const surgery = patient.surgery_history || '-'
  const dept = record.department || '未指定'
  const chief = record.chief_complaint || '-'
  const present = record.present_illness || '-'
  const past = record.past_history || '-'
  const exam = record.physical_exam || '-'
  const vitalsText = vitals.length > 0 ? vitals.join('  ') : '-'
  const aux = record.auxiliary_exam || '-'
  const diag = record.diagnosis || '-'
  const med = record.medication || '-'
  const advice = record.medical_advice || '-'
  const rest = record.rest_days ? record.rest_days + ' 天' : '-'

  const html = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n<title>病历单</title>\n<style>\n  * { margin: 0; padding: 0; box-sizing: border-box; }\n  body { font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif; color: #1a1a1a; line-height: 1.7; padding: 2rem; max-width: 800px; margin: 0 auto; }\n  h1 { text-align: center; font-size: 1.4rem; margin-bottom: 0.15rem; letter-spacing: 0.1em; }\n  .meta { text-align: center; font-size: 0.8rem; color: #666; margin-bottom: 1.2rem; }\n  h2 { font-size: 1rem; border-bottom: 2px solid #333; padding-bottom: 0.25rem; margin: 1.2rem 0 0.6rem; }\n  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }\n  th, td { padding: 0.35rem 0.5rem; border: 1px solid #ccc; text-align: left; vertical-align: top; }\n  th { background: #f5f5f5; font-weight: 600; width: 100px; white-space: nowrap; }\n  .info-table td { width: auto; }\n  .signature { margin-top: 2rem; text-align: right; font-size: 0.85rem; }\n  .signature span { display: inline-block; min-width: 80px; border-bottom: 1px solid #333; margin-left: 0.5rem; }\n  @media print {\n    body { padding: 1rem; }\n    @page { margin: 1.5cm; }\n  }\n</style>\n</head>\n<body>\n<h1>门诊病历</h1>\n<div class="meta">就诊日期：' + today + '　|　科室：' + dept + '</div>\n\n<h2>患者基本信息</h2>\n<table class="info-table">\n  <tr><th>姓名</th><td>' + patientName + '</td><th>性别</th><td>' + patientGender + '</td><th>年龄</th><td>' + ageText + '</td></tr>\n  <tr><th>手机号</th><td>' + phone + '</td><th>身份证后5位</th><td>' + idCard + '</td><th>出生日期</th><td>' + dob + '</td></tr>\n  <tr><th>联系地址</th><td colspan="3">' + addr + '</td><th>紧急联系人</th><td>' + emergency + '</td></tr>\n  <tr><th>既往病史</th><td colspan="5">' + pHistory + '</td></tr>\n  <tr><th>过敏史</th><td colspan="5">' + allergy + '</td></tr>\n  <tr><th>手术/外伤史</th><td colspan="5">' + surgery + '</td></tr>\n</table>\n\n<h2>病历记录</h2>\n<table>\n  <tr><th>主诉</th><td>' + chief + '</td></tr>\n  <tr><th>现病史</th><td>' + present + '</td></tr>\n  <tr><th>既往史</th><td>' + past + '</td></tr>\n  <tr><th>体格检查</th><td>' + exam + '</td></tr>\n  <tr><th>生命体征</th><td>' + vitalsText + '</td></tr>\n  <tr><th>辅助检查</th><td>' + aux + '</td></tr>\n  <tr><th>诊断</th><td>' + diag + '</td></tr>\n  <tr><th>用药</th><td>' + med + '</td></tr>\n  <tr><th>医嘱</th><td>' + advice + '</td></tr>\n  <tr><th>休息天数</th><td>' + rest + '</td></tr>\n</table>\n\n<div class="signature">\n  医生签名：<span></span>\n</div>\n</body>\n</html>'

  const w = window.open('', '_blank', 'width=800,height=600')
  if (!w) return
  w.document.write(html)
  w.document.close()
  w.focus()
  setTimeout(() => w.print(), 300)
}
</script>

<template>
  <div class="assistant-page">
    <header class="top-bar">
      <button class="back-btn" @click="goBack">
        <span class="back-arrow">&larr;</span> 返回工作台
      </button>
      <div v-if="patientName" class="patient-hero">
        <div class="avatar">{{ patientName[0] }}</div>
        <div>
          <h1 class="patient-name">{{ patientName }}</h1>
          <p class="patient-meta">
            {{ patientGender || '-' }}
            <template v-if="patientAge != null"> · {{ patientAge }} 岁</template>
          </p>
        </div>
      </div>
    </header>

    <div class="content-grid">
      <!-- 以往病例 -->
      <section class="panel records-panel">
        <h3 class="panel-title">以往病例</h3>
        <div v-if="fetchingRecords" class="panel-status">加载中…</div>
        <div v-else-if="recordsError" class="panel-error">{{ recordsError }}</div>
        <div v-else-if="medicalRecords.length === 0" class="panel-empty">暂无历史病历</div>
        <div v-else class="records-scroll">
          <div v-for="rec in medicalRecords" :key="rec.id" class="record-card">
            <div class="record-head">
              <span class="record-date">{{ rec.visit_date }}</span>
              <span class="record-dept">{{ rec.department }}</span>
              <button class="expand-btn" @click="toggleRecordExpand(rec.id)">
                {{ expandedRecordId === rec.id ? '收起' : '详情' }}
              </button>
            </div>
            <p class="record-brief"><strong>主诉</strong> {{ rec.chief_complaint || '-' }}</p>
            <p class="record-brief"><strong>诊断</strong> {{ rec.diagnosis || '-' }}</p>
            <div v-if="expandedRecordId === rec.id" class="record-detail">
              <p v-if="rec.present_illness"><strong>现病史</strong> {{ rec.present_illness }}</p>
              <p v-if="rec.medication"><strong>用药</strong> {{ rec.medication }}</p>
              <p v-if="rec.medical_advice"><strong>医嘱</strong> {{ rec.medical_advice }}</p>
            </div>
          </div>
        </div>
      </section>

      <!-- 问诊转写 -->
      <section class="panel transcribe-panel">
        <h3 class="panel-title">问诊助手</h3>
        <div class="transcribe-area">
          <WhisperTranscriber ref="transcriberRef" :patient-gender="patientGender" :patient-age="patientAge">
            <template #extra-controls>
              <button class="write-btn" :disabled="extracting || !canWrite" @click="extractMedicalRecord">
                {{ extracting ? '提取中…' : '写入病历' }}
              </button>
              <span v-if="extractError" class="error-msg-inline">{{ extractError }}</span>
            </template>
          </WhisperTranscriber>
        </div>
      </section>
    </div>

    <!-- 审核弹窗 -->
    <div v-if="showReviewModal" class="modal-overlay" @click.self="showReviewModal = false">
      <div class="review-modal">
        <div class="modal-header">
          <h2>审核病历数据</h2>
          <span class="hint">AI 从对话中提取，请审核修改后确认写入</span>
        </div>

        <div v-if="saveError" class="error-msg modal-error">{{ saveError }}</div>

        <div class="review-form">
          <div class="form-col">
            <label class="field">
              <span class="field-label">就诊科室</span>
              <input v-model="recordForm.department" class="field-input" />
            </label>
            <label class="field">
              <span class="field-label">主诉 *</span>
              <textarea v-model="recordForm.chief_complaint" class="field-textarea" rows="2"></textarea>
            </label>
            <label class="field">
              <span class="field-label">现病史</span>
              <textarea v-model="recordForm.present_illness" class="field-textarea" rows="3"></textarea>
            </label>
            <label class="field">
              <span class="field-label">既往史</span>
              <textarea v-model="recordForm.past_history" class="field-textarea" rows="2"></textarea>
            </label>
            <label class="field">
              <span class="field-label">体格检查</span>
              <textarea v-model="recordForm.physical_exam" class="field-textarea" rows="2"></textarea>
            </label>
          </div>

          <div class="form-col">
            <div class="vital-row">
              <label class="field vital-field">
                <span class="field-label">体温 (℃)</span>
                <input v-model="recordForm.temperature" class="field-input" type="number" step="0.1" />
              </label>
              <label class="field vital-field">
                <span class="field-label">脉搏 (次/分)</span>
                <input v-model="recordForm.pulse" class="field-input" type="number" />
              </label>
              <label class="field vital-field">
                <span class="field-label">呼吸 (次/分)</span>
                <input v-model="recordForm.respiration" class="field-input" type="number" />
              </label>
              <label class="field vital-field">
                <span class="field-label">血压</span>
                <input v-model="recordForm.blood_pressure" class="field-input" placeholder="120/80" />
              </label>
            </div>
            <label class="field">
              <span class="field-label">辅助检查</span>
              <textarea v-model="recordForm.auxiliary_exam" class="field-textarea" rows="2"></textarea>
            </label>
            <label class="field">
              <span class="field-label">诊断 *</span>
              <textarea v-model="recordForm.diagnosis" class="field-textarea" rows="2"></textarea>
            </label>
            <label class="field">
              <span class="field-label">用药</span>
              <textarea v-model="recordForm.medication" class="field-textarea" rows="2"></textarea>
            </label>
            <label class="field">
              <span class="field-label">医嘱</span>
              <textarea v-model="recordForm.medical_advice" class="field-textarea" rows="2"></textarea>
            </label>
            <div class="vital-row">
              <label class="field vital-field">
                <span class="field-label">休息天数</span>
                <input v-model="recordForm.rest_days" class="field-input" type="number" />
              </label>
              <label class="field vital-field">
                <span class="field-label">医生姓名</span>
                <input v-model="recordForm.doctor_name" class="field-input" />
              </label>
            </div>
          </div>
        </div>

        <div class="modal-footer">
          <button class="cancel-btn" @click="showReviewModal = false">取消</button>
          <button class="confirm-btn" :disabled="saving" @click="confirmSave">
            {{ saving ? '写入中…' : '确认写入数据库' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 写入成功提示 -->
    <div v-if="saveSuccess" class="success-toast">
      病历已成功写入数据库
    </div>
  </div>
</template>

<style scoped>
.assistant-page {
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 1rem 1.25rem 1.25rem;
  min-height: 100vh;
  text-align: left;
  box-sizing: border-box;
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
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.4rem 0.85rem;
  font-size: 0.85rem;
  color: var(--text);
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px;
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

.patient-name {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-h);
}

.patient-meta {
  margin: 0.15rem 0 0;
  font-size: 0.85rem;
  color: var(--text);
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  align-items: stretch;
  min-height: calc(100vh - 110px);
}

@media (max-width: 900px) {
  .content-grid {
    grid-template-columns: 1fr;
    min-height: auto;
  }
}

.panel {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel-title {
  margin: 0;
  padding: 0.85rem 1rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-h);
  background: var(--code-bg);
  border-bottom: 1px solid var(--border);
}

.panel-status, .panel-empty {
  padding: 1.5rem 1rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--text);
}

.panel-error {
  margin: 0.75rem;
  padding: 0.65rem 0.85rem;
  font-size: 0.85rem;
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 6px;
}

.records-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 0.65rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.record-card {
  padding: 0.75rem 0.85rem;
  background: var(--code-bg);
  border-radius: 8px;
  border-left: 3px solid #6366f1;
}

.record-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.35rem;
}

.record-date {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-h);
}

.record-dept {
  font-size: 0.72rem;
  padding: 0.1rem 0.4rem;
  background: var(--bg);
  border-radius: 4px;
  color: var(--text);
}

.expand-btn {
  margin-left: auto;
  padding: 0.15rem 0.5rem;
  font-size: 0.72rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg);
  cursor: pointer;
  color: #2563eb;
}

.record-brief {
  margin: 0.2rem 0;
  font-size: 0.8rem;
  color: var(--text-h);
  line-height: 1.4;
}

.record-brief strong {
  color: var(--text);
  font-weight: 600;
  margin-right: 0.25rem;
}

.record-detail {
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px dashed var(--border);
  font-size: 0.78rem;
  color: var(--text);
  line-height: 1.45;
}

.record-detail p { margin: 0.25rem 0; }

.transcribe-panel .transcribe-area {
  padding: 0.75rem;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.transcribe-area :deep(.transcription-container) {
  max-width: 100%;
  margin: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.transcribe-area :deep(.transcript-box) {
  flex: 1;
  min-height: 200px;
}

.records-panel {
  min-height: 300px;
}

.transcribe-panel {
  min-height: 500px;
}

.write-btn {
  padding: 0.55rem 1.2rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
  white-space: nowrap;
}
.write-btn:hover:not(:disabled) { opacity: 0.92; }
.write-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.error-msg-inline { color: #ef4444; font-size: 0.8rem; }
.error-msg { color: #ef4444; font-size: 0.9rem; }

/* 审核弹窗 */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 1rem;
}

.review-modal {
  background: var(--bg);
  border-radius: 12px;
  width: 95%;
  max-width: 900px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-header {
  padding: 1.2rem 1.5rem;
  border-bottom: 1px solid var(--border);
}
.modal-header h2 {
  margin: 0 0 0.3rem;
  font-size: 1.2rem;
}
.hint {
  font-size: 0.82rem;
  color: var(--text);
}

.modal-error {
  margin: 0.75rem 1.5rem 0;
  padding: 0.6rem 0.9rem;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 6px;
}

.review-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 1.5rem;
  padding: 1.2rem 1.5rem;
}

@media (max-width: 700px) {
  .review-form {
    grid-template-columns: 1fr;
  }
}

.form-col {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.field-label {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-h);
}

.field-input {
  padding: 0.5rem 0.7rem;
  font-size: 0.88rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-h);
  outline: none;
  transition: border-color 0.2s;
}
.field-input:focus { border-color: #3b82f6; }

.field-textarea {
  padding: 0.5rem 0.7rem;
  font-size: 0.88rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-h);
  outline: none;
  resize: vertical;
  font-family: inherit;
  transition: border-color 0.2s;
}
.field-textarea:focus { border-color: #3b82f6; }

.vital-row {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 0.6rem;
}

.vital-field {
  min-width: 0;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border);
  background: var(--code-bg);
  border-radius: 0 0 12px 12px;
}

.cancel-btn {
  padding: 0.5rem 1.2rem;
  font-size: 0.9rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-h);
  cursor: pointer;
}

.confirm-btn {
  padding: 0.5rem 1.5rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.confirm-btn:hover:not(:disabled) { opacity: 0.9; }
.confirm-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.success-toast {
  position: fixed;
  bottom: 2rem;
  left: 50%;
  transform: translateX(-50%);
  padding: 0.8rem 2rem;
  background: #2563eb;
  color: #fff;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  z-index: 3000;
  animation: toast-in 0.3s ease;
}

@keyframes toast-in {
  from { opacity: 0; transform: translateX(-50%) translateY(20px); }
  to { opacity: 1; transform: translateX(-50%) translateY(0); }
}
</style>
