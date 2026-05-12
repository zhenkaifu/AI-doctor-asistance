<script setup lang="ts">
import { ref, reactive } from 'vue'
import { supabase } from '../lib/supabase'

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

const props = defineProps<{
  patient: Patient | null
}>()

const emit = defineEmits<{
  close: []
  saved: []
}>()

const saving = ref(false)
const errorMsg = ref('')

const form = reactive({
  name: props.patient?.name || '',
  gender: props.patient?.gender || '',
  date_of_birth: props.patient?.date_of_birth || '',
  phone: props.patient?.phone || '',
  id_card_last5: props.patient?.id_card_last5 || '',
  past_history: props.patient?.past_history || '',
  allergy_history: props.patient?.allergy_history || '',
  surgery_history: props.patient?.surgery_history || '',
  emergency_contact_name: props.patient?.emergency_contact_name || '',
  emergency_contact_phone: props.patient?.emergency_contact_phone || '',
  address: props.patient?.address || '',
})

const isEdit = !!props.patient

async function submitForm() {
  errorMsg.value = ''

  if (!form.name.trim()) {
    errorMsg.value = '姓名为必填项'
    return
  }
  if (!form.gender) {
    errorMsg.value = '性别为必填项'
    return
  }
  if (!form.date_of_birth) {
    errorMsg.value = '出生日期为必填项'
    return
  }
  if (!form.phone.trim()) {
    errorMsg.value = '手机号为必填项'
    return
  }
  if (!form.past_history.trim()) {
    errorMsg.value = '既往病史为必填项'
    return
  }
  if (!form.allergy_history.trim()) {
    errorMsg.value = '过敏史为必填项'
    return
  }
  if (!form.surgery_history.trim()) {
    errorMsg.value = '手术/外伤史为必填项'
    return
  }

  saving.value = true

  const payload = {
    name: form.name.trim(),
    gender: form.gender,
    date_of_birth: form.date_of_birth,
    phone: form.phone.trim(),
    id_card_last5: form.id_card_last5 || null,
    past_history: form.past_history.trim(),
    allergy_history: form.allergy_history.trim(),
    surgery_history: form.surgery_history.trim(),
    emergency_contact_name: form.emergency_contact_name.trim() || null,
    emergency_contact_phone: form.emergency_contact_phone.trim() || null,
    address: form.address.trim() || null,
  }

  let error = null

  if (isEdit && props.patient) {
    const { error: err } = await supabase
      .from('patients')
      .update(payload)
      .eq('id', props.patient.id)
    error = err
  } else {
    const { error: err } = await supabase
      .from('patients')
      .insert(payload)
    error = err
  }

  saving.value = false

  if (error) {
    errorMsg.value = error.message
    return
  }

  emit('saved')
}
</script>

<template>
  <div class="modal-overlay" @click.self="emit('close')">
    <div class="modal-content">
      <div class="modal-header">
        <h2>{{ isEdit ? '编辑病人' : '新增病人' }}</h2>
        <button class="close-btn" @click="emit('close')">&times;</button>
      </div>

      <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>

      <div class="form-body">
        <label class="field">
          <span class="field-label">姓名 *</span>
          <input v-model="form.name" type="text" class="field-input" placeholder="请输入姓名" />
        </label>

        <label class="field">
          <span class="field-label">性别 *</span>
          <select v-model="form.gender" class="field-input">
            <option value="">请选择</option>
            <option value="男">男</option>
            <option value="女">女</option>
          </select>
        </label>

        <label class="field">
          <span class="field-label">出生日期 *</span>
          <input v-model="form.date_of_birth" type="date" class="field-input" />
        </label>

        <label class="field">
          <span class="field-label">手机号 *</span>
          <input v-model="form.phone" type="text" class="field-input" placeholder="请输入手机号" />
        </label>

        <label class="field">
          <span class="field-label">既往病史 *</span>
          <textarea v-model="form.past_history" class="field-textarea" rows="1" placeholder="如高血压、糖尿病等慢性病史"></textarea>
        </label>

        <label class="field">
          <span class="field-label">过敏史 *</span>
          <textarea v-model="form.allergy_history" class="field-textarea" rows="1" placeholder="如青霉素过敏、海鲜过敏等"></textarea>
        </label>

        <label class="field">
          <span class="field-label">手术/外伤史 *</span>
          <textarea v-model="form.surgery_history" class="field-textarea" rows="1" placeholder="如阑尾切除、骨折史等"></textarea>
        </label>

        <label class="field">
          <span class="field-label">紧急联系人</span>
          <input v-model="form.emergency_contact_name" type="text" class="field-input" placeholder="请输入紧急联系人姓名" />
        </label>

        <label class="field">
          <span class="field-label">紧急联系人电话</span>
          <input v-model="form.emergency_contact_phone" type="text" class="field-input" placeholder="请输入紧急联系人电话" />
        </label>

        <label class="field">
          <span class="field-label">联系地址</span>
          <input v-model="form.address" type="text" class="field-input" placeholder="请输入联系地址" />
        </label>

        <label class="field">
          <span class="field-label">身份证后5位</span>
          <input
            v-model="form.id_card_last5"
            type="text"
            maxlength="5"
            class="field-input"
            placeholder="请输入身份证后5位"
          />
        </label>
      </div>

      <div class="modal-footer">
        <button class="cancel-btn" @click="emit('close')">取消</button>
        <button class="submit-btn" :disabled="saving" @click="submitForm">
          {{ saving ? '保存中…' : '保存' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--bg);
  border-radius: 12px;
  width: 90%;
  max-width: 420px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  padding: 0;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 1.3rem;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.modal-header h2 {
  margin: 0;
  font-size: 1.05rem;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: var(--text);
  cursor: pointer;
  padding: 0;
  line-height: 1;
}
.close-btn:hover {
  color: var(--text-h);
}

.error-msg {
  margin: 0.75rem 1.5rem 0;
  padding: 0.6rem 0.9rem;
  font-size: 0.85rem;
  color: #ef4444;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 6px;
}

.form-body {
  padding: 1rem 1.3rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  overflow-y: auto;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.field-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-h);
}

.field-input {
  padding: 0.5rem 0.7rem;
  font-size: 0.9rem;
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
  font-size: 0.9rem;
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

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding: 0.8rem 1.3rem;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}

.cancel-btn {
  padding: 0.45rem 1.1rem;
  font-size: 0.9rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-h);
  cursor: pointer;
}
.cancel-btn:hover {
  background: var(--code-bg);
}

.submit-btn {
  padding: 0.45rem 1.3rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: var(--accent);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.submit-btn:hover {
  opacity: 0.85;
}
.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
