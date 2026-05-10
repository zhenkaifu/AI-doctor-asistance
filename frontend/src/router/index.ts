import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'patients',
      component: () => import('../views/PatientSearch.vue'),
    },
    {
      path: '/assistant',
      name: 'assistant',
      component: () => import('../views/AssistantPage.vue'),
    },
  ],
})

export default router
