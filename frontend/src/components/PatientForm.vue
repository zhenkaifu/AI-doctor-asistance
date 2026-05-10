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
})

const isEdit = !!props.patient

async function submitForm() {
  errorMsg.value = ''

  if (!form.name.trim()) {
    errorMsg.value = '姓名为必填项'
    return
  }

  saving.value = true

  const payload = {
    name: form.name.trim(),
    gender: form.gender || null,
    date_of_birth: form.date_of_birth || null,
    phone: form.phone || null,
    id_card_last5: form.id_card_last5 || null,
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
          <span class="field-label">性别</span>
          <select v-model="form.gender" class="field-input">
            <option value="">请选择</option>
            <option value="男">男</option>
            <option value="女">女</option>
          </select>
        </label>

        <label class="field">
          <span class="field-label">出生日期</span>
          <input v-model="form.date_of_birth" type="date" class="field-input" />
        </label>

        <label class="field">
          <span class="field-label">手机号</span>
          <input v-model="form.phone" type="text" class="field-input" placeholder="请输入手机号" />
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
  max-width: 480px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  padding: 0;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.2rem 1.5rem;
  border-bottom: 1px solid var(--border);
}

.modal-header h2 {
  margin: 0;
  font-size: 1.15rem;
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
  padding: 1.2rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.field-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-h);
}

.field-input {
  padding: 0.55rem 0.75rem;
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

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border);
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
.cancel-btn:hover {
  background: var(--code-bg);
}

.submit-btn {
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
.submit-btn:hover {
  opacity: 0.85;
}
.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
