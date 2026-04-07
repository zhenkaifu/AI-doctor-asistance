<script setup lang="ts">
import WhisperTranscriber from './components/WhisperTranscriber.vue';
import viteLogo from './assets/vite.svg';
import vueLogo from './assets/vue.svg';
</script>

<template>
  <header class="app-header">
    <div class="logos">
      <img :src="viteLogo" class="logo vite" alt="Vite logo" />
      <img :src="vueLogo" class="logo vue" alt="Vue logo" />
    </div>
    <h1>语音实时转写</h1>
    <p>本地 Whisper 引擎 · NestJS 中继 · 浏览器录音</p>
  </header>

  <main>
    <WhisperTranscriber />
  </main>

  <footer class="app-footer">
    <p>© 2026 AI Project · 本地 Whisper 语音识别</p>
  </footer>
</template>

<style>
#app {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

.app-header {
  margin-bottom: 3rem;
}

.logos {
  display: flex;
  justify-content: center;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.logo {
  height: 3.5rem;
  will-change: filter;
  transition: filter 300ms;
}

.logo.vite:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}

.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}

h1 {
  font-size: 3rem;
  margin: 0.5rem 0;
  background: linear-gradient(45deg, var(--accent), #42b883);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.app-header p {
  color: var(--text);
  font-size: 1.1rem;
}

.app-footer {
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid var(--border);
  color: var(--text);
  font-size: 0.9rem;
}

main {
  min-height: 60vh;
}
</style>
