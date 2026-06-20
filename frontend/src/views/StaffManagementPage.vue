<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useApiError } from '../composables/useApiError'
import { callEdgeFunction } from '../lib/session'

const router = useRouter()
const { resolveError } = useApiError()

interface Staff {
  id: string
  email: string
  name: string
  role: string
  department: string | null
  phone: string | null
  created_at: string
}

// ---- 列表 ----
const staffList = ref<Staff[]>([])
const loading = ref(false)
const listError = ref('')

async function fetchStaff() {
  loading.value = true
  listError.value = ''
  try {
    staffList.value = await callEdgeFunction<Staff[]>('list-all-staff')
  } catch (e: any) {
    staffList.value = []
    listError.value = await resolveError(e, '加载员工列表失败，请稍后重试')
  } finally {
    loading.value = false
  }
}

// ---- 新增弹窗 ----
const showAddModal = ref(false)
const addError = ref('')
const adding = ref(false)
const addSuccess = ref('')

const addForm = ref({ email: '', password: '', name: '', role: 'doctor' as 'doctor' | 'nurse', department: '', phone: '' })

function openAddModal() {
  addError.value = ''
  addSuccess.value = ''
  addForm.value = { email: '', password: '', name: '', role: 'doctor', department: '', phone: '' }
  showAddModal.value = true
}

async function handleAdd() {
  addError.value = ''
  addSuccess.value = ''
  const { email, password, name, role, department, phone } = addForm.value
  if (!email.trim() || !password || !name.trim() || !phone.trim()) {
    addError.value = '邮箱、密码、姓名、手机号为必填'
    return
  }
  if (password.length < 6) { addError.value = '密码至少 6 位'; return }

  adding.value = true
  try {
    await callEdgeFunction('create-staff-user', {
      body: {
        email: email.trim(),
        password,
        name: name.trim(),
        role,
        department: role === 'doctor' ? (department.trim() || null) : null,
        phone: phone.trim(),
      },
    })
    showAddModal.value = false
    await fetchStaff()
  } catch (e: any) { addError.value = await resolveError(e, '创建失败') } finally { adding.value = false }
}

// ---- 编辑弹窗 ----
const showEditModal = ref(false)
const editingStaff = ref<Staff | null>(null)
const editError = ref('')
const editing = ref(false)
const editSuccess = ref('')
const editForm = ref({ name: '', phone: '', department: '', role: 'doctor' as 'doctor' | 'nurse' })

function openEditModal(s: Staff) {
  editError.value = ''
  editSuccess.value = ''
  editingStaff.value = s
  editForm.value = { name: s.name, phone: s.phone || '', department: s.department || '', role: (s.role as 'doctor' | 'nurse') || 'doctor' }
  showEditModal.value = true
}

async function handleEdit() {
  editError.value = ''
  editSuccess.value = ''
  const { name, phone, department, role } = editForm.value
  if (!name.trim() || !phone.trim()) {
    editError.value = '姓名和手机号为必填'
    return
  }
  if (!editingStaff.value) return

  editing.value = true
  try {
    await callEdgeFunction('update-staff-user', {
      body: {
        userId: editingStaff.value.id,
        name: name.trim(),
        phone: phone.trim(),
        department: role === 'doctor' ? (department.trim() || null) : null,
        role,
      },
    })
    showEditModal.value = false
    await fetchStaff()
  } catch (e: any) { editError.value = await resolveError(e, '更新失败') } finally { editing.value = false }
}

// ---- 删除 ----
const deleteConfirmId = ref<string | null>(null)
const deleteError = ref('')
const deleting = ref(false)

async function handleDelete(userId: string) {
  deleteError.value = ''
  deleting.value = true
  try {
    await callEdgeFunction('delete-staff-user', { body: { userId } })
    deleteConfirmId.value = null
    await fetchStaff()
  } catch (e: any) { deleteError.value = await resolveError(e, '删除失败') } finally { deleting.value = false }
}

const ROLE_LABELS: Record<string, string> = { admin: '管理员', doctor: '医生', nurse: '护士' }
function roleLabel(role: string) { return ROLE_LABELS[role] || role }

function openStats(s: Staff) {
  router.push({
    name: 'adminStaffStats',
    query: { userId: s.id, role: s.role, name: s.name },
  })
}

onMounted(() => { fetchStaff() })
</script>

<template>
  <div class="staff-page">
    <header class="top-bar">
      <button class="back-btn" @click="router.push('/admin')">&larr; 返回</button>
      <h1 class="page-title">医护人员管理</h1>
      <button class="add-btn" @click="openAddModal">+ 新增员工</button>
    </header>

    <div v-if="listError" class="error-msg">{{ listError }}</div>
    <div v-if="deleteError" class="error-msg">{{ deleteError }}</div>

    <div class="table-wrap">
      <table class="staff-table" v-if="staffList.length > 0">
        <thead>
          <tr>
            <th>姓名</th>
            <th>邮箱</th>
            <th>角色</th>
            <th>科室</th>
            <th>手机号</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in staffList" :key="s.id">
            <td>{{ s.name }}</td>
            <td>{{ s.email }}</td>
            <td><span :class="['role-tag', s.role]">{{ roleLabel(s.role) }}</span></td>
            <td>{{ s.department || '-' }}</td>
            <td>{{ s.phone || '-' }}</td>
            <td>{{ s.created_at ? new Date(s.created_at).toLocaleDateString('zh-CN') : '-' }}</td>
            <td class="action-cell">
              <template v-if="s.role !== 'admin'">
                <div class="action-group">
                  <button class="action-btn edit" @click="openEditModal(s)">编辑</button>
                  <button
                    v-if="s.role === 'nurse' || s.role === 'doctor'"
                    class="action-btn stats"
                    @click="openStats(s)"
                  >数据</button>
                  <button v-if="deleteConfirmId !== s.id" class="action-btn del" @click="deleteConfirmId = s.id">删除</button>
                  <span v-else class="confirm-del">
                    <button class="action-btn del" :disabled="deleting" @click="handleDelete(s.id)">{{ deleting ? '…' : '确认' }}</button>
                    <button class="action-btn cancel" @click="deleteConfirmId = null">取消</button>
                  </span>
                </div>
              </template>
              <span v-else class="muted">—</span>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else-if="!loading && !listError" class="empty-msg">暂无医护人员</div>
      <div v-if="loading" class="loading-msg">加载中…</div>
    </div>

    <!-- 新增弹窗 -->
    <div v-if="showAddModal" class="modal-overlay" @click.self="showAddModal = false">
      <div class="modal-card">
        <div class="modal-header"><h2>新增员工</h2></div>
        <div v-if="addError" class="error-msg">{{ addError }}</div>
        <div v-if="addSuccess" class="success-msg">{{ addSuccess }}</div>
        <div class="form-body">
          <label class="field"><span class="field-label">邮箱（登录账号）*</span><input v-model="addForm.email" type="email" class="field-input" placeholder="staff@hospital.com" /></label>
          <label class="field"><span class="field-label">初始密码 *</span><input v-model="addForm.password" type="password" class="field-input" placeholder="至少 6 位" /></label>
          <label class="field"><span class="field-label">姓名 *</span><input v-model="addForm.name" type="text" class="field-input" placeholder="真实姓名" /></label>
          <label class="field"><span class="field-label">角色 *</span>
            <select v-model="addForm.role" class="field-input">
              <option value="doctor">医生</option>
              <option value="nurse">护士</option>
            </select>
          </label>
          <label v-if="addForm.role === 'doctor'" class="field"><span class="field-label">科室</span><input v-model="addForm.department" type="text" class="field-input" placeholder="如：心内科" /></label>
          <label class="field"><span class="field-label">手机号 *</span><input v-model="addForm.phone" type="text" class="field-input" placeholder="手机号" /></label>
        </div>
        <div class="modal-footer">
          <button class="cancel-btn" @click="showAddModal = false">取消</button>
          <button class="confirm-btn" :disabled="adding" @click="handleAdd">{{ adding ? '创建中…' : '确认创建' }}</button>
        </div>
      </div>
    </div>

    <!-- 编辑弹窗 -->
    <div v-if="showEditModal" class="modal-overlay" @click.self="showEditModal = false">
      <div class="modal-card">
        <div class="modal-header"><h2>编辑员工</h2></div>
        <div v-if="editError" class="error-msg">{{ editError }}</div>
        <div v-if="editSuccess" class="success-msg">{{ editSuccess }}</div>
        <div class="form-body">
          <label class="field"><span class="field-label">姓名 *</span><input v-model="editForm.name" type="text" class="field-input" /></label>
          <label class="field"><span class="field-label">角色 *</span>
            <select v-model="editForm.role" class="field-input">
              <option value="doctor">医生</option>
              <option value="nurse">护士</option>
            </select>
          </label>
          <label v-if="editForm.role === 'doctor'" class="field"><span class="field-label">科室</span><input v-model="editForm.department" type="text" class="field-input" placeholder="如：心内科" /></label>
          <label class="field"><span class="field-label">手机号 *</span><input v-model="editForm.phone" type="text" class="field-input" /></label>
        </div>
        <div class="modal-footer">
          <button class="cancel-btn" @click="showEditModal = false">取消</button>
          <button class="confirm-btn" :disabled="editing" @click="handleEdit">{{ editing ? '保存中…' : '保存' }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.staff-page { padding: 1.5rem; text-align: left; }
.top-bar { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.2rem; }
.back-btn { padding: 0.35rem 0.8rem; font-size: 0.85rem; color: var(--text); background: transparent; border: 1px solid transparent; border-radius: 6px; cursor: pointer; white-space: nowrap; }
.back-btn:hover { background: var(--code-bg); color: var(--accent); }
.page-title { font-size: 1.2rem; margin: 0; flex: 1; color: var(--text-h); }
.add-btn { padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 600; color: #fff; background: var(--accent); border: none; border-radius: 6px; cursor: pointer; }
.add-btn:hover { opacity: 0.85; }
.error-msg { margin-bottom: 1rem; padding: 0.6rem 0.9rem; color: #ef4444; background: rgba(239, 68, 68, 0.08); border-radius: 6px; font-size: 0.85rem; }
.success-msg { margin-bottom: 1rem; padding: 0.6rem 0.9rem; color: #10b981; background: rgba(16, 185, 129, 0.08); border-radius: 6px; font-size: 0.85rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
.staff-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.staff-table th, .staff-table td { padding: 0.65rem 0.8rem; border-bottom: 1px solid var(--border); text-align: left; }
.staff-table th { font-weight: 600; color: var(--text); background: var(--code-bg); font-size: 0.82rem; white-space: nowrap; }
.staff-table td { color: var(--text-h); }
.action-cell { white-space: nowrap; }
.action-group { display: flex; flex-direction: column; gap: 0.35rem; align-items: flex-start; }
.action-btn { padding: 0.2rem 0.6rem; font-size: 0.78rem; font-weight: 600; border: none; border-radius: 4px; cursor: pointer; }
.action-btn.edit { color: #3b82f6; background: rgba(59, 130, 246, 0.1); }
.action-btn.edit:hover { background: rgba(59, 130, 246, 0.2); }
.action-btn.stats { color: #8b5cf6; background: rgba(139, 92, 246, 0.1); }
.action-btn.stats:hover { background: rgba(139, 92, 246, 0.2); }
.action-btn.del { color: #ef4444; background: rgba(239, 68, 68, 0.1); }
.action-btn.del:hover { background: rgba(239, 68, 68, 0.2); }
.action-btn.cancel { color: var(--text); background: var(--code-bg); }
.confirm-del { display: inline-flex; gap: 0.3rem; }
.muted { color: var(--text); font-size: 0.82rem; }
.role-tag { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
.role-tag.admin { background: rgba(239, 68, 68, 0.1); color: #ef4444; }
.role-tag.doctor { background: rgba(59, 130, 246, 0.1); color: #3b82f6; }
.role-tag.nurse { background: rgba(16, 185, 129, 0.1); color: #10b981; }
.empty-msg, .loading-msg { text-align: center; padding: 2rem; color: var(--text); font-size: 0.9rem; }
.modal-overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.45); display: flex; align-items: center; justify-content: center; z-index: 2000; padding: 1rem; }
.modal-card { background: var(--bg); border-radius: 12px; width: 100%; max-width: 480px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3); }
.modal-header { padding: 1rem 1.5rem; border-bottom: 1px solid var(--border); }
.modal-header h2 { margin: 0; font-size: 1.05rem; }
.form-body { padding: 1rem 1.5rem; display: flex; flex-direction: column; gap: 0.8rem; }
.field { display: flex; flex-direction: column; gap: 0.3rem; }
.field-label { font-size: 0.82rem; font-weight: 600; color: var(--text-h); }
.field-input { padding: 0.5rem 0.7rem; font-size: 0.88rem; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--text-h); outline: none; }
.field-input:focus { border-color: var(--accent); }
.modal-footer { display: flex; justify-content: flex-end; gap: 0.75rem; padding: 1rem 1.5rem; border-top: 1px solid var(--border); background: var(--code-bg); border-radius: 0 0 12px 12px; }
.cancel-btn { padding: 0.45rem 1.1rem; font-size: 0.9rem; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--text-h); cursor: pointer; }
.confirm-btn { padding: 0.45rem 1.3rem; font-size: 0.9rem; font-weight: 600; color: #fff; background: var(--accent); border: none; border-radius: 6px; cursor: pointer; }
.confirm-btn:hover { opacity: 0.85; }
.confirm-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
