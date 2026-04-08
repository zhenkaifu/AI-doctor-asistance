<script setup lang="ts">
import { ref, onUnmounted, reactive, nextTick, computed } from 'vue';
import { io, Socket } from 'socket.io-client';

interface TranscriptionResult {
  speaker: string;
  text: string;
  start: number;
  end: number;
  isFinal: boolean;
  engine_latency?: number;
  backend_latency?: number;
}

const transcriptionList = reactive<TranscriptionResult[]>([]);
const currentLatency = ref({ engine: 0, backend: 0, total: 0 });
const isRecording = ref(false);
const statusMessage = ref('就绪');
const statusTone = ref<'idle' | 'progress' | 'ok' | 'rec' | 'err'>('idle');

const generatingAdvice = ref(false);
const adviceText = ref('');
const adviceError = ref('');

let socket: Socket | null = null;
let audioContext: AudioContext | null = null;
let processor: ScriptProcessorNode | null = null;
let stream: MediaStream | null = null;
let lastChunkSentAt = 0;
let adviceAbort: AbortController | null = null;

const canGenerateAdvice = computed(
  () => transcriptionList.length > 0 && !generatingAdvice.value
);

const DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions';

type AdviceBlock =
  | { kind: 'h2'; title: string }
  | { kind: 'list'; items: string[] }
  | { kind: 'p'; text: string };

/** 去掉简单加粗标记，避免渲染难看 */
function stripInlineBold(s: string): string {
  return s.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1');
}

/**
 * 解析助手输出：优先识别 ## 标题、- 列表；否则整段作为段落并保留换行。
 * 仅作展示用，不使用 v-html，避免 XSS。
 */
function parseAdviceContent(raw: string): AdviceBlock[] {
  const text = raw.trim();
  if (!text) return [];

  const lines = text.split(/\r?\n/);
  const blocks: AdviceBlock[] = [];
  let paraBuf: string[] = [];
  let listBuf: string[] = [];

  const flushPara = () => {
    const t = stripInlineBold(paraBuf.join('\n').trim());
    if (t) blocks.push({ kind: 'p', text: t });
    paraBuf = [];
  };
  const flushList = () => {
    if (listBuf.length) {
      blocks.push({
        kind: 'list',
        items: listBuf.map((x) => stripInlineBold(x.trim())),
      });
      listBuf = [];
    }
  };

  let sawStructured = false;
  for (const line of lines) {
    const h = line.match(/^#{2,3}\s+(.+)$/);
    if (h) {
      sawStructured = true;
      flushList();
      flushPara();
      blocks.push({ kind: 'h2', title: stripInlineBold(h[1].trim()) });
      continue;
    }
    const li =
      line.match(/^\s*[-*•]\s+(.+)$/) ||
      line.match(/^\s*\d+[\.\)、]\s*(.+)$/);
    if (li) {
      sawStructured = true;
      flushPara();
      listBuf.push(li[1].trim());
      continue;
    }
    if (line.trim() === '') {
      flushList();
      flushPara();
      continue;
    }
    flushList();
    paraBuf.push(line);
  }
  flushList();
  flushPara();

  if (!sawStructured && blocks.length <= 1 && blocks[0]?.kind === 'p') {
    return blocks;
  }
  if (blocks.length === 0) {
    return [{ kind: 'p', text: stripInlineBold(text) }];
  }
  return blocks;
}

const adviceBlocks = computed(() => parseAdviceContent(adviceText.value));

function buildTranscriptRaw(): string {
  return transcriptionList.map((t) => `${t.speaker}：${t.text}`).join('\n');
}

async function generateDoctorAdvice() {
  if (!canGenerateAdvice.value) return;

  const apiKey = import.meta.env.VITE_DEEPSEEK_API_KEY?.trim();
  if (!apiKey) {
    adviceError.value =
      '未配置 DeepSeek 密钥：请在 frontend 目录创建 .env 并设置 VITE_DEEPSEEK_API_KEY。';
    adviceText.value = '';
    return;
  }

  adviceError.value = '';
  adviceText.value = '';
  generatingAdvice.value = true;
  adviceAbort = new AbortController();

const systemPrompt = `你是严谨、果断的内科医生助手。用户提供一段语音识别转写文本，其中可能混杂医生和病人的对话。
执行步骤：
0. 句子拼接：将语音识别转写中明显断开、但语义连贯的短句或碎片化片段合理连接成完整句子（例如“我肚子 疼”拼成“我肚子疼”）。不改变原意，不添加新信息。
1. 角色区分：先识别文本中哪些话是医生说的（通常是问句或引导性语句），哪些是病人说的（主诉、症状描述、回答、既往史等）。忽略医生说的话，只提取病人说的内容用于后续处理。
2. 文本清洗：对病人说的内容进行清洗——修正明显错别字和同音误识；删除与病情无关或重复的信息；不编造未出现的病史与检查结果。
3. 基于清洗后的患者信息，判断是否**必须追问**才能做出合理临床决策。追问的门槛：当前信息缺失会直接影响诊断方向或患者安全（如危险信号、鉴别诊断的关键依据）。常见病、典型表现或已可合理推断的情况，不应追问，直接进入“不需要继续追问”分支。
4. 输出格式（使用普通 Markdown 标题，不要代码块和反引号）：

## 清洗后病情摘要
（只写从病人话语中提取的客观事实，一段话概括）

## 判断结论
需要继续追问 / 不需要继续追问

如果是「需要继续追问」：
## 建议追问问题
- （只写3条最重要的、具体、可直接对患者提问的问题，每条一行）
## 追问目的
（简要说明每类问题对诊断或安全的意义）

如果是「不需要继续追问」：
## 专业判断与初步诊断
- 诊断1（依据：……）
- 诊断2（依据：……）
（列出1-3个鉴别诊断，每个一行，写清关键依据）
## 下一步建议
- 检查（如必要）
- 处理（如用药、生活方式调整）
- 随访建议
- 警示症状（出现哪些情况需立即就医）

要求：全程中文；不要输出与任务无关的客套话；不要习惯性追问；信息足够时请直接给出诊断和建议。`;

  const userContent = `以下为语音识别转写（可能杂乱），请先清洗后给出“医生助手意见”：\n\n${buildTranscriptRaw()}`;

  try {
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
        temperature: 0.3,
      }),
      signal: adviceAbort.signal,
    });

    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(errBody || `HTTP ${res.status}`);
    }

    const data = (await res.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const text = data.choices?.[0]?.message?.content?.trim();
    if (!text) {
      throw new Error('接口未返回有效内容');
    }
    adviceText.value = text;
  } catch (e) {
    if ((e as Error).name === 'AbortError') return;
    adviceError.value =
      e instanceof Error ? e.message : '生成失败，请稍后重试';
  } finally {
    generatingAdvice.value = false;
    adviceAbort = null;
  }
}

const startTranscription = async () => {
  try {
    adviceText.value = '';
    adviceError.value = '';
    statusMessage.value = '正在连接后端…';
    statusTone.value = 'progress';
    socket = io('http://localhost:3000');

    socket.on('connect', () => {
      statusMessage.value = '正在初始化语音识别…';
      statusTone.value = 'progress';
    });

    socket.on('ready', async () => {
      statusMessage.value = '引擎已就绪';
      statusTone.value = 'ok';
      await startMicrophone();
    });

    socket.on('transcription', (result: TranscriptionResult) => {
      const now = Date.now();
      currentLatency.value = {
        engine: result.engine_latency || 0,
        backend: result.backend_latency || 0,
        total: (now - lastChunkSentAt) / 1000,
      };

      transcriptionList.push(result);

      if (transcriptionList.length > 50) transcriptionList.shift();

      nextTick(() => {
        const box = document.querySelector('.transcript-box');
        if (box) box.scrollTop = box.scrollHeight;
      });
    });

    socket.on('error', (msg: string) => {
      statusMessage.value = '错误：' + msg;
      statusTone.value = 'err';
      stopTranscription();
    });
  } catch (err) {
    console.error('Failed to start transcription:', err);
    statusMessage.value = '错误：' + (err as Error).message;
    statusTone.value = 'err';
  }
};

const startMicrophone = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (socket?.connected && isRecording.value) {
        const inputData = e.inputBuffer.getChannelData(0);
        const buffer = float32ToInt16(inputData);
        lastChunkSentAt = Date.now();
        socket.emit('audio-chunk', buffer);
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    isRecording.value = true;
    statusMessage.value = '录音中…';
    statusTone.value = 'rec';
  } catch (err) {
    console.error('Microphone access failed:', err);
    statusMessage.value = '无法访问麦克风';
    statusTone.value = 'err';
  }
};

const stopTranscription = () => {
  isRecording.value = false;
  if (socket) {
    socket.disconnect();
    socket = null;
  }
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  statusMessage.value =
    transcriptionList.length > 0 ? '实时转写已结束' : '就绪';
  statusTone.value = transcriptionList.length > 0 ? 'idle' : 'idle';
};

const float32ToInt16 = (buffer: Float32Array): ArrayBuffer => {
  let l = buffer.length;
  const buf = new Int16Array(l);
  while (l--) {
    const s = Math.max(-1, Math.min(1, buffer[l]));
    buf[l] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return buf.buffer;
};

onUnmounted(() => {
  adviceAbort?.abort();
  stopTranscription();
});
</script>

<template>
  <div class="transcription-container">
    <div class="controls">
      <button
        :class="['record-btn', { recording: isRecording }]"
        @click="isRecording ? stopTranscription() : startTranscription()"
      >
        <span class="icon">{{ isRecording ? '⏹' : '🎤' }}</span>
        {{ isRecording ? '停止录音' : '开始实时转写' }}
      </button>
      <button
        type="button"
        class="case-btn"
        :disabled="!canGenerateAdvice"
        @click="generateDoctorAdvice"
      >
        <span class="icon">📋</span>
        {{ generatingAdvice ? '生成中…' : '意见' }}
      </button>
      <div class="status-badge" :class="`tone-${statusTone}`">
        {{ statusMessage }}
      </div>

      <div v-if="isRecording" class="latency-info">
        <span>引擎：{{ currentLatency.engine }}s</span>
        <span>桥接：{{ currentLatency.backend }}ms</span>
        <span>端到端：{{ currentLatency.total.toFixed(2) }}s</span>
      </div>
    </div>

    <div class="transcript-box">
      <div v-if="transcriptionList.length === 0" class="empty-state">
        开始说话后，转写内容将显示在这里…
      </div>
      <div
        v-for="(item, index) in transcriptionList"
        :key="index"
        :class="['transcript-item', { 'is-final': item.isFinal }]"
      >
        <span class="speaker">{{ item.speaker }}：</span>
        <span class="text">{{ item.text }}</span>
        <span v-if="!item.isFinal" class="cursor">|</span>
      </div>
    </div>

    <div v-if="adviceError" class="case-sheet case-sheet--error">
      {{ adviceError }}
    </div>
    <div v-else-if="adviceText" class="advice-panel">
      <header class="advice-panel__head">
        <span class="advice-panel__icon" aria-hidden="true">🩺</span>
        <div class="advice-panel__head-text">
          <h3 class="advice-panel__title">医生助手意见</h3>
          <p class="advice-panel__subtitle">基于当前转写内容的辅助参考，不能替代面诊与检查</p>
        </div>
      </header>
      <div class="advice-panel__body">
        <template v-for="(block, idx) in adviceBlocks" :key="idx">
          <h4 v-if="block.kind === 'h2'" class="advice-block-title">
            {{ block.title }}
          </h4>
          <ul
            v-else-if="block.kind === 'list'"
            class="advice-list"
          >
            <li v-for="(item, j) in block.items" :key="j" class="advice-list__item">
              {{ item }}
            </li>
          </ul>
          <p v-else class="advice-paragraph">{{ block.text }}</p>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.transcription-container {
  max-width: 800px;
  margin: 2rem auto;
  padding: 1.5rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow);
}

.controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.latency-info {
  font-size: 0.75rem;
  display: flex;
  gap: 0.8rem;
  color: var(--text);
  background: var(--social-bg);
  padding: 0.3rem 0.8rem;
  border-radius: 4px;
}

.record-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: white;
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.record-btn.recording {
  background: #ef4444;
  animation: pulse 2s infinite;
}

.case-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-h);
  background: var(--social-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.case-btn:hover:not(:disabled) {
  border-color: var(--accent);
  background: var(--accent-bg);
}

.case-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.status-badge {
  font-size: 0.85rem;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  background: var(--social-bg);
  color: var(--text);
}

.status-badge.tone-ok {
  color: #10b981;
  font-weight: bold;
}

.status-badge.tone-rec {
  color: #ef4444;
  font-weight: bold;
}

.status-badge.tone-err {
  color: #ef4444;
}

.status-badge.tone-progress {
  color: var(--accent);
}

.transcript-box {
  background: var(--code-bg);
  border-radius: 8px;
  padding: 1.5rem;
  min-height: 300px;
  max-height: 500px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  text-align: left;
}

.case-sheet {
  margin-top: 1.25rem;
  text-align: left;
}

.case-sheet--error {
  padding: 1rem 1.25rem;
  border-radius: 10px;
  border: 1px solid rgba(239, 68, 68, 0.35);
  background: rgba(239, 68, 68, 0.06);
  color: #ef4444;
  font-size: 0.9rem;
}

.advice-panel {
  margin-top: 1.25rem;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: linear-gradient(
    145deg,
    var(--code-bg) 0%,
    color-mix(in srgb, var(--code-bg) 92%, var(--accent)) 100%
  );
  box-shadow: var(--shadow);
  overflow: hidden;
  text-align: left;
}

.advice-panel__head {
  display: flex;
  align-items: flex-start;
  gap: 0.85rem;
  padding: 1.1rem 1.25rem;
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  border-bottom: 1px solid var(--border);
}

.advice-panel__icon {
  font-size: 1.75rem;
  line-height: 1;
}

.advice-panel__head-text {
  min-width: 0;
}

.advice-panel__title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--text-h);
  letter-spacing: 0.02em;
}

.advice-panel__subtitle {
  margin: 0.35rem 0 0;
  font-size: 0.78rem;
  color: var(--text);
  opacity: 0.88;
  line-height: 1.45;
}

.advice-panel__body {
  padding: 1.15rem 1.35rem 1.35rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.advice-block-title {
  margin: 0;
  padding-bottom: 0.35rem;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--accent);
  border-bottom: 2px solid color-mix(in srgb, var(--accent) 35%, transparent);
}

.advice-paragraph {
  margin: 0;
  font-size: 0.92rem;
  line-height: 1.75;
  color: var(--text-h);
  white-space: pre-wrap;
  word-break: break-word;
}

.advice-list {
  margin: 0;
  padding: 0 0 0 1.1rem;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.advice-list__item {
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--text-h);
  padding-left: 0.15rem;
}

.advice-list__item::marker {
  color: var(--accent);
}

.empty-state {
  color: var(--text);
  opacity: 0.6;
  text-align: center;
  margin-top: 5rem;
}

.transcript-item {
  line-height: 1.6;
  padding: 0.5rem;
  border-radius: 4px;
}

.transcript-item.is-final {
  border-left: 3px solid var(--accent);
  background: rgba(170, 59, 255, 0.05);
}

.speaker {
  font-weight: bold;
  margin-right: 0.25rem;
  color: var(--accent);
}

.text {
  color: var(--text-h);
}

.cursor {
  display: inline-block;
  width: 2px;
  background: var(--accent);
  animation: blink 1s step-end infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
  }
}

@keyframes blink {
  from,
  to {
    opacity: 1;
  }
  50% {
    opacity: 0;
  }
}
</style>
