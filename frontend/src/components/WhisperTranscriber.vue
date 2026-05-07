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
const currentLatency = ref({ engine: 0, backend: 0 });
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

const systemPrompt = `你是内科临床决策支持助手。医生正在接诊患者，你会收到实时语音转写的对话文本。你的任务是辅助医生判断：当前采集的信息是否足以形成初步诊断，或还需要追问哪些关键问题。
工作流程：

1. 信息提取与整理
- 从患者话语中提取主诉、现病史要素（症状的部位、性质、程度、起病与持续时间、诱发与缓解因素、伴随症状）。
- 同步提取个体化风险背景：既往病史（慢性病、手术史）、家族史（遗传病、肿瘤、心脑血管疾病）、个人史（吸烟、饮酒、职业暴露、近期旅行或疫区停留）、过敏史（药物、食物）、近期用药情况（处方药、非处方药、保健品）。
- 忽略医生的话语，仅以患者陈述为依据。
- ASR 转写可能有碎片化或同音错字，结合上下文合理推断并修正，但不编造未出现的信息。

2. 信息充足性判断（关键原则：不是做完美病历，而是辅助决策）

当以下核心条件满足时，即判定为「信息充足」，应进入诊断阶段：
- 主诉的核心特征已明确（部位 + 性质 + 时间线索）
- 已排除需要紧急处理的危险信号（如胸痛、剧烈头痛、高热不退、意识改变、急性腹痛、咯血、呕血等）

以下情况不应判定为「信息不足」，不要因此持续追问：
- 风险背景（既往史、家族史、个人史、用药史）有缺失 → 可在诊断的「下一步建议」中提醒医生补充，不必作为追问的拦路虎
- 鉴别诊断存在不确定性 → 这正是需要你给出建议的场景，临床决策本就伴随不确定性
- 信息不完美 → 有初步方向远好于无方向

只有当主诉本身模糊不清（如患者只说"不舒服"但未描述任何具体症状）或危险信号未被排除时，才判定为「信息不足」并追问。

3. 输出格式（使用普通 Markdown 标题，不要代码块和反引号）：

## 病史摘要
（一段话概括从患者话语中提取的客观病史，按现病史逻辑组织）

## 已明确的关键信息
- （要点1）
- （要点2）

## 关键信息缺口
- （缺口1：缺少什么信息，对诊断的意义）
- （缺口2）

## 综合判断
信息充足，可形成初步诊断 / 信息不足，建议补充问诊

若是「信息不足」：
## 建议追问
- 问题1（针对哪个信息缺口，按优先级排序）
- 问题2
- 问题3
（最多 3 条，精准具体，可直接向患者提问）

若是「信息充足」：
## 初步诊断与鉴别诊断
- 诊断1（依据：……）
- 诊断2（依据：……）
（1-3 个鉴别诊断，按可能性排序）
## 下一步建议
- 辅助检查
- 初步处理
- 随访建议
- 需警惕的危险信号（出现哪些情况应立即就医）

要求：全程中文；只基于患者已说出的信息分析，不编造病史；你是在辅助临床决策，不是在参加医学考试——有初步方向就果断给出诊断，缺失的风险背景在「下一步建议」中提醒医生补充即可，不必追问到完美才下结论；追问最多 1-3 条，精准针对最核心的信息缺口；切忌连续追问多轮而不给诊断建议。`;

  const userContent = `以下为语音识别转写（可能杂乱），请按工作流程分析并给出建议：\n\n${buildTranscriptRaw()}`;

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
    transcriptionList.splice(0); // 新会话清空历史转写
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

      // 替换原 socket.on('transcription', ...) 部分的逻辑
      socket.on('transcription', (result: TranscriptionResult) => {
        currentLatency.value = {
          engine: result.engine_latency || 0,
          backend: result.backend_latency || 0,
        };

        // ========== 修改后的拼接逻辑 ==========
        if (!result.isFinal) {
          // 未完成句：先移除之前的临时句（如果有），再新增临时句
          // 找到最后一个未完成的项并删除
          const tempItemIndex = transcriptionList.findIndex(item => !item.isFinal);
          if (tempItemIndex > -1) {
            transcriptionList.splice(tempItemIndex, 1);
          }
          // 添加新的临时句
          transcriptionList.push({
            ...result
          });
        } else {
          // 完成句：先移除临时句（如果有），再添加最终句
          const tempItemIndex = transcriptionList.findIndex(item => !item.isFinal);
          if (tempItemIndex > -1) {
            transcriptionList.splice(tempItemIndex, 1);
          }
          // 新增最终句（确保每句独立）
          transcriptionList.push({
            ...result,
            isFinal: true
          });
        }

        // 限制最大条数（可选，保持原逻辑）
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
  // 将最后一个未完成句转为完成句
  const tempIdx = transcriptionList.findIndex(item => !item.isFinal);
  if (tempIdx > -1) {
    transcriptionList[tempIdx].isFinal = true;
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
        <span>引擎 {{ currentLatency.engine }}ms</span>
        <span>后端 {{ currentLatency.backend }}ms</span>
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
        <span v-if="item.isFinal && item.engine_latency" class="item-latency">{{ item.engine_latency }}ms</span>
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

.item-latency {
  font-size: 0.7rem;
  color: var(--text);
  opacity: 0.5;
  margin-left: 0.5rem;
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
