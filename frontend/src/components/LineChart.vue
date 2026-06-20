<script setup lang="ts">
import { computed } from 'vue'

export interface ChartPoint {
  label: string
  value: number
}

const props = withDefaults(
  defineProps<{
    data: ChartPoint[]
    height?: number
    color?: string
    unit?: string
  }>(),
  {
    height: 260,
    color: '#3b82f6',
    unit: '次',
  },
)

const PAD = { top: 24, right: 16, bottom: 36, left: 44 }
const W = 640

const maxValue = computed(() => {
  const m = Math.max(...props.data.map((d) => d.value), 0)
  return m === 0 ? 1 : m
})

const chartHeight = computed(() => props.height - PAD.top - PAD.bottom)

const points = computed(() => {
  const n = props.data.length
  if (n === 0) return []
  const innerW = W - PAD.left - PAD.right
  return props.data.map((d, i) => {
    const x = PAD.left + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW)
    const y = PAD.top + chartHeight.value * (1 - d.value / maxValue.value)
    return { ...d, x, y }
  })
})

const linePath = computed(() => {
  const pts = points.value
  if (pts.length === 0) return ''
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
})

const areaPath = computed(() => {
  const pts = points.value
  if (pts.length === 0) return ''
  const baseY = PAD.top + chartHeight.value
  const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
  const last = pts[pts.length - 1]
  const first = pts[0]
  return `${line} L ${last.x.toFixed(1)} ${baseY} L ${first.x.toFixed(1)} ${baseY} Z`
})

const yTicks = computed(() => {
  const m = maxValue.value
  const step = m <= 5 ? 1 : Math.ceil(m / 4)
  const ticks: number[] = []
  for (let v = 0; v <= m; v += step) ticks.push(v)
  if (ticks[ticks.length - 1] < m) ticks.push(m)
  return [...new Set(ticks)]
})

const xLabelIndices = computed(() => {
  const n = props.data.length
  if (n <= 7) return props.data.map((_, i) => i)
  const indices: number[] = [0]
  for (let i = 5; i < n - 1; i += 5) indices.push(i)
  if (!indices.includes(n - 1)) indices.push(n - 1)
  return indices
})

function formatLabel(label: string): string {
  const parts = label.split('-')
  if (parts.length === 3) return `${Number(parts[1])}/${Number(parts[2])}`
  return label
}
</script>

<template>
  <div class="line-chart">
    <svg
      :viewBox="`0 0 ${W} ${height}`"
      class="chart-svg"
      role="img"
      :aria-label="`近30日折线图，单位${unit}`"
    >
      <defs>
        <linearGradient :id="`area-${color.replace('#', '')}`" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" :stop-color="color" stop-opacity="0.18" />
          <stop offset="100%" :stop-color="color" stop-opacity="0.02" />
        </linearGradient>
      </defs>

      <g class="grid">
        <line
          v-for="tick in yTicks"
          :key="tick"
          :x1="PAD.left"
          :y1="PAD.top + chartHeight * (1 - tick / maxValue)"
          :x2="W - PAD.right"
          :y2="PAD.top + chartHeight * (1 - tick / maxValue)"
        />
      </g>

      <g class="y-labels">
        <text
          v-for="tick in yTicks"
          :key="'y' + tick"
          :x="PAD.left - 8"
          :y="PAD.top + chartHeight * (1 - tick / maxValue) + 4"
          text-anchor="end"
        >{{ tick }}</text>
      </g>

      <path
        v-if="areaPath"
        :d="areaPath"
        :fill="`url(#area-${color.replace('#', '')})`"
      />
      <path
        v-if="linePath"
        :d="linePath"
        fill="none"
        :stroke="color"
        stroke-width="2.5"
        stroke-linejoin="round"
        stroke-linecap="round"
      />

      <g class="dots">
        <circle
          v-for="(p, i) in points"
          :key="i"
          :cx="p.x"
          :cy="p.y"
          r="3.5"
          :fill="color"
        />
      </g>

      <g class="x-labels">
        <text
          v-for="idx in xLabelIndices"
          :key="'x' + idx"
          :x="points[idx]?.x ?? 0"
          :y="height - 8"
          text-anchor="middle"
        >{{ formatLabel(data[idx]?.label ?? '') }}</text>
      </g>
    </svg>

    <div v-if="data.length === 0" class="empty-overlay">暂无数据</div>
  </div>
</template>

<style scoped>
.line-chart {
  position: relative;
  width: 100%;
}
.chart-svg {
  width: 100%;
  height: auto;
  display: block;
}
.grid line {
  stroke: var(--border);
  stroke-width: 1;
  stroke-dasharray: 4 4;
}
.y-labels text,
.x-labels text {
  font-size: 11px;
  fill: var(--text);
}
.empty-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text);
  font-size: 0.9rem;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 8px;
}
</style>
