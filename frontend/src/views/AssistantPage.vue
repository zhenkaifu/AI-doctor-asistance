<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { supabase } from '../lib/supabase'
import WhisperTranscriber from '../components/WhisperTranscriber.vue'

const router = useRouter()
const route = useRoute()

const patientId = (route.query.patientId as string) || ''
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

  const systemPrompt = `你是专业的病历结构化助手。根据医患对话转写文本，提取以下字段并以 JSON 返回。不编造信息，对话中未提及的字段返回空字符串。

返回格式（严格 JSON，不要 markdown 代码块包裹）：
{
  "department": "就诊科室",
  "chief_complaint": "主诉（症状+持续时间，例如：腹痛2天）",
  "present_illness": "现病史（起病情况、症状演变、伴随症状、诊治经过）",
  "past_history": "既往史（慢性病、手术史、过敏史、家族史等）",
  "physical_exam": "体格检查（T、P、R、BP、心肺腹查体等发现，可从对话中推测但明确标注为'未述及'如无）",
  "temperature": "体温数值（如 36.5，不要单位）",
  "pulse": "脉搏/心率数值",
  "respiration": "呼吸频率数值",
  "blood_pressure": "血压（如 120/80）",
  "auxiliary_exam": "辅助检查结果",
  "diagnosis": "诊断或初步诊断",
  "medication": "用药（药品名+用法用量）",
  "medical_advice": "医嘱/处置建议",
  "rest_days": "建议休息天数（仅数字）",
  "doctor_name": "医生姓名"
}

要求：仅从对话中提取，不编造；数值类字段只写数字；体征未述及时留空。`

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

  const { error } = await supabase.from('medical_records').insert({
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
  })

  saving.value = false

  if (error) {
    saveError.value = error.message
    return
  }

  saveSuccess.value = true
  showReviewModal.value = false
}

function goBack() {
  router.push('/')
}
</script>

<template>
  <div class="assistant-page">
    <!-- 顶部导航 -->
    <header class="top-bar">
      <button class="back-btn" @click="goBack">
        <span class="back-arrow">&larr;</span> 返回
      </button>
      <div class="patient-tag" v-if="patientName">
        <span class="tag-icon">👤</span>
        <span class="tag-name">{{ patientName }}</span>
      </div>
    </header>

    <!-- 转写区域 -->
    <div class="transcribe-area">
      <WhisperTranscriber ref="transcriberRef" :patient-gender="patientGender" :patient-age="patientAge">
        <template #extra-controls>
          <button
            class="write-btn"
            :disabled="extracting || !canWrite"
            @click="extractMedicalRecord"
          >
            {{ extracting ? '提取中…' : '📝 写入' }}
          </button>
          <span v-if="extractError" class="error-msg-inline">{{ extractError }}</span>
        </template>
      </WhisperTranscriber>
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
  max-width: 100%;
  padding: 0 1rem 2rem;
  min-height: 100vh;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.35rem 0.8rem;
  font-size: 0.85rem;
  color: var(--text);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}
.back-btn:hover {
  background: var(--code-bg);
  color: var(--accent);
}
.back-arrow {
  font-size: 0.85rem;
}

.patient-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.8rem;
  background: var(--accent-bg);
  border: 1px solid var(--accent-border);
  border-radius: 20px;
  font-size: 0.85rem;
}
.tag-icon {
  font-size: 0.9rem;
}
.tag-name {
  font-weight: 600;
  color: var(--accent);
}

/* 转写区域 */
.transcribe-area {
  width: 100%;
}
.transcribe-area :deep(.transcription-container) {
  max-width: 100%;
  margin: 0.5rem 0 0;
}

.write-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: #fff;
  background: #10b981;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity 0.2s, transform 0.15s;
  white-space: nowrap;
}
.write-btn:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}
.write-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-msg-inline {
  color: #ef4444;
  font-size: 0.8rem;
}

.error-msg {
  color: #ef4444;
  font-size: 0.9rem;
}

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
.field-input:focus {
  border-color: var(--accent);
}

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
.field-textarea:focus {
  border-color: var(--accent);
}

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
  background: var(--accent);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.confirm-btn:hover {
  opacity: 0.85;
}
.confirm-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.success-toast {
  position: fixed;
  bottom: 2rem;
  left: 50%;
  transform: translateX(-50%);
  padding: 0.8rem 2rem;
  background: #10b981;
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
