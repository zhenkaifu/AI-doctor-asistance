import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'
import { useAuth } from './composables/useAuth'

async function bootstrap() {
  const app = createApp(App)
  app.use(router)

  // 在挂载前初始化鉴权（恢复会话 + 查询角色）
  const { init } = useAuth()
  await init()

  app.mount('#app')
}

bootstrap()
