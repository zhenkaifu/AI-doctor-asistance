import { createRouter, createWebHistory } from 'vue-router'
import { useAuth } from '../composables/useAuth'

// 扩展路由 meta 类型
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    guest?: boolean
    adminOnly?: boolean
    nurseOnly?: boolean
    doctorOnly?: boolean
  }
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/LoginPage.vue'),
      meta: { guest: true },
    },
    {
      path: '/',
      name: 'patients',
      component: () => import('../views/PatientSearch.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/nurse',
      name: 'nurse',
      component: () => import('../views/NurseDashboardPage.vue'),
      meta: { requiresAuth: true, nurseOnly: true },
    },
    {
      path: '/nurse/registration',
      name: 'nurseRegistration',
      component: () => import('../views/NurseRegistrationPage.vue'),
      meta: { requiresAuth: true, nurseOnly: true },
    },
    {
      path: '/nurse/patient',
      name: 'nursePatient',
      component: () => import('../views/DoctorPatientDetailPage.vue'),
      meta: { requiresAuth: true, nurseOnly: true },
    },
    {
      path: '/doctor',
      name: 'doctor',
      component: () => import('../views/DoctorDashboardPage.vue'),
      meta: { requiresAuth: true, doctorOnly: true },
    },
    {
      path: '/doctor/patient',
      name: 'doctorPatient',
      component: () => import('../views/DoctorPatientDetailPage.vue'),
      meta: { requiresAuth: true, doctorOnly: true },
    },
    {
      path: '/assistant',
      name: 'assistant',
      component: () => import('../views/AssistantPage.vue'),
      meta: { requiresAuth: true },
    },
    // ---- 管理员路由 ----
    {
      path: '/admin',
      name: 'admin',
      component: () => import('../views/AdminDashboardPage.vue'),
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      path: '/admin/staff',
      name: 'adminStaff',
      component: () => import('../views/StaffManagementPage.vue'),
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      path: '/admin/staff/stats',
      name: 'adminStaffStats',
      component: () => import('../views/StaffStatsPage.vue'),
      meta: { requiresAuth: true, adminOnly: true },
    },
  ],
})

// 鉴权守卫
router.beforeEach(async (to, _from) => {
  const { isAuthenticated, role, refreshRole } = useAuth()

  console.log(
    '[router.beforeEach] to.name=',
    to.name,
    'to.meta=',
    JSON.stringify(to.meta),
    'isAuthenticated=',
    isAuthenticated.value,
    'role=',
    role.value
  )

  // 未登录访问需鉴权页面 → 跳转登录
  if (to.meta.requiresAuth && !isAuthenticated.value) {
    console.log('[router.beforeEach] 未登录 → 跳转 /login')
    return { name: 'login' }
  }

  // 已登录访问登录页 → 按角色跳转
  if (to.meta.guest && isAuthenticated.value) {
    const target =
      role.value === 'admin' ? 'admin'
      : role.value === 'nurse' ? 'nurse'
      : role.value === 'doctor' ? 'doctor'
      : 'patients'
    console.log('[router.beforeEach] 已登录用户访问登录页 → 跳转', target)
    return { name: target }
  }

  // 医生访问通用病人查询页 → 跳转医生工作台
  if (to.name === 'patients' && role.value === 'doctor') {
    return { name: 'doctor' }
  }

  // 进入管理员路由前，刷新权限（后台静默执行，不阻塞路由）
  if (to.meta.adminOnly && isAuthenticated.value) {
    console.log('[router.beforeEach] 进入管理员路由，后台刷新权限...')
    // 不使用 await，让权限刷新在后台进行，不阻塞路由
    refreshRole().catch(e => console.warn('[router.beforeEach] 权限刷新失败:', e))
  }

  // 进入护士路由前，后台刷新权限（不阻塞）
  if (to.meta.nurseOnly && isAuthenticated.value) {
    console.log('[router.beforeEach] 进入护士路由，后台刷新权限...')
    refreshRole().catch(e => console.warn('[router.beforeEach] 权限刷新失败:', e))
  }

  // 进入医生路由前，后台刷新权限（不阻塞）
  if (to.meta.doctorOnly && isAuthenticated.value) {
    refreshRole().catch(e => console.warn('[router.beforeEach] 权限刷新失败:', e))
  }

  // 非管理员访问管理员页面 → 跳转首页
  if (to.meta.adminOnly && role.value !== 'admin') {
    console.log('[router.beforeEach] 非管理员访问 admin 页 → 跳转 /patients')
    return { name: 'patients' }
  }

  // 非护士访问护士页面 → 跳转首页
  if (to.meta.nurseOnly && role.value !== 'nurse') {
    console.log('[router.beforeEach] 非护士访问 nurse 页 → 跳转 /patients')
    return { name: 'patients' }
  }

  // 非医生访问医生页面 → 跳转首页
  if (to.meta.doctorOnly && role.value !== 'doctor') {
    console.log('[router.beforeEach] 非医生访问 doctor 页 → 跳转 /patients')
    return { name: 'patients' }
  }

  console.log('[router.beforeEach] 放行')
})

export default router
