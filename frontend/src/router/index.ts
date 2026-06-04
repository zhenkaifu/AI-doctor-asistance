import { createRouter, createWebHistory } from 'vue-router'
import { useAuth } from '../composables/useAuth'

// 扩展路由 meta 类型
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    guest?: boolean
    adminOnly?: boolean
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
    const target = role.value === 'admin' ? 'admin' : 'patients'
    console.log('[router.beforeEach] 已登录用户访问登录页 → 跳转', target)
    return { name: target }
  }

  // 进入管理员路由前，刷新权限（后台静默执行，不阻塞路由）
  if (to.meta.adminOnly && isAuthenticated.value) {
    console.log('[router.beforeEach] 进入管理员路由，后台刷新权限...')
    // 不使用 await，让权限刷新在后台进行，不阻塞路由
    refreshRole().catch(e => console.warn('[router.beforeEach] 权限刷新失败:', e))
  }

  // 非管理员访问管理员页面 → 跳转首页
  if (to.meta.adminOnly && role.value !== 'admin') {
    console.log('[router.beforeEach] 非管理员访问 admin 页 → 跳转 /patients')
    return { name: 'patients' }
  }

  console.log('[router.beforeEach] 放行')
})

export default router
