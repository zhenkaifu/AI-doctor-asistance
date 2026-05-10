<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { supabase } from '../lib/supabase'
import PatientForm from '../components/PatientForm.vue'

interface Patient {
  id: string
  name: string
  gender: string | null
  date_of_birth: string | null
  phone: string | null
  id_card_last5: string | null
  created_at: string
}

const router = useRouter()
const patients = ref<Patient[]>([])
const loading = ref(false)
const searched = ref(false)

const searchForm = reactive({
  name: '',
  phone: '',
  id_card_last5: '',
})

const showForm = ref(false)
const editingPatient = ref<Patient | null>(null)

async function doSearch() {
  loading.value = true
  searched.value = true

  let query = supabase.from('patients').select('*')

  const filters: string[] = []
  if (searchForm.name.trim()) {
    filters.push(`name.ilike.%${searchForm.name.trim()}%`)
  }
  if (searchForm.phone.trim()) {
    filters.push(`phone.eq.${searchForm.phone.trim()}`)
  }
  if (searchForm.id_card_last5.trim()) {
    filters.push(`id_card_last5.eq.${searchForm.id_card_last5.trim()}`)
  }

  if (filters.length > 0) {
    query = query.or(filters.join(','))
  }

  const { data, error } = await query.order('created_at', { ascending: false })

  if (error) {
    console.error('Search error:', error)
    patients.value = []
  } else {
    patients.value = (data as Patient[]) || []
  }

  loading.value = false
}

function openAddPatient() {
  editingPatient.value = null
  showForm.value = true
}

function onFormSaved() {
  showForm.value = false
  doSearch()
}

function goToAssistant(patient: Patient) {
  router.push({ path: '/assistant', query: { patientId: patient.id, patientName: patient.name } })
}
</script>

<template>
  <div class="patient-page">
    <!-- 顶部导航 -->
    <header class="top-bar">
      <div class="spacer"></div>
      <h1 class="page-title">病人查询</h1>
      <button class="add-btn" @click="openAddPatient">+ 新增病人</button>
    </header>

    <!-- 搜索区域 -->
    <div class="search-area">
      <div class="search-row">
        <input
          v-model="searchForm.name"
          type="text"
          placeholder="姓名（支持模糊搜索）"
          class="search-input"
          @keyup.enter="doSearch"
        />
        <input
          v-model="searchForm.phone"
          type="text"
          placeholder="手机号（精确匹配）"
          class="search-input"
          @keyup.enter="doSearch"
        />
        <input
          v-model="searchForm.id_card_last5"
          type="text"
          placeholder="身份证后5位"
          maxlength="5"
          class="search-input"
          @keyup.enter="doSearch"
        />
        <button class="search-btn" :disabled="loading" @click="doSearch">
          {{ loading ? '查询中…' : '查询' }}
        </button>
      </div>
    </div>

    <!-- 结果表格 -->
    <div v-if="searched" class="result-area">
      <div v-if="loading" class="loading-text">查询中…</div>
      <div v-else-if="patients.length === 0" class="empty-text">
        未找到匹配的病人记录
      </div>
      <table v-else class="patient-table">
        <thead>
          <tr>
            <th>姓名</th>
            <th>性别</th>
            <th>出生日期</th>
            <th>手机号</th>
            <th>身份证后5位</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="p in patients" :key="p.id">
            <td>{{ p.name }}</td>
            <td>{{ p.gender || '-' }}</td>
            <td>{{ p.date_of_birth || '-' }}</td>
            <td>{{ p.phone || '-' }}</td>
            <td>{{ p.id_card_last5 || '-' }}</td>
            <td>
              <button class="consult-btn" @click="goToAssistant(p)">开始问诊</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 新增/编辑弹窗 -->
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
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.5rem;
  text-align: left;
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

.add-btn {
  padding: 0.5rem 1.2rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity 0.2s;
  white-space: nowrap;
}
.add-btn:hover {
  opacity: 0.85;
}

.search-area {
  margin-bottom: 1.5rem;
}

.search-row {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

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
  transition: border-color 0.2s;
}
.search-input:focus {
  border-color: var(--accent);
}

.search-btn {
  padding: 0.6rem 1.5rem;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.search-btn:hover {
  opacity: 0.85;
}
.search-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.result-area {
  margin-top: 0.5rem;
}

.loading-text,
.empty-text {
  text-align: center;
  padding: 2rem;
  color: var(--text);
  font-size: 0.95rem;
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
}

.patient-table th {
  font-weight: 600;
  color: var(--text-h);
  background: var(--code-bg);
  font-size: 0.85rem;
}

.patient-table td {
  color: var(--text-h);
}

.consult-btn {
  padding: 0.35rem 0.9rem;
  font-size: 0.82rem;
  font-weight: 600;
  border: 1px solid var(--accent);
  border-radius: 4px;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  transition: opacity 0.2s;
}
.consult-btn:hover {
  opacity: 0.85;
}

.spacer {
  width: 100px;
}
</style>
