export type RegistrationStatus = 'waiting' | 'cancelled' | 'completed'

export const purposeLabels: Record<string, string> = {
  initial: '初诊',
  follow_up: '复查',
  medication: '开药',
  consultation: '咨询',
  chronic_follow_up: '慢性病复诊',
}

export const statusLabels: Record<RegistrationStatus, string> = {
  waiting: '待就诊',
  cancelled: '已取消',
  completed: '已完成',
}

/** 将数据库中的 status 规范为 waiting | cancelled | completed */
export function normalizeRegistrationStatus(status: string | null | undefined): RegistrationStatus {
  if (status === 'cancelled' || status === 'completed') return status
  if (status === 'waiting' || status === 'active') return 'waiting'
  return 'waiting'
}

function getBeijingDateParts(d: Date): { y: number; m: number; day: number } {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
  }).formatToParts(d)
  const get = (type: string) => Number(parts.find(p => p.type === type)?.value ?? 0)
  return { y: get('year'), m: get('month'), day: get('day') }
}

export function parseAppointmentTime(dateStr: string): Date | null {
  if (!dateStr) return null
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(dateStr) && !dateStr.includes('Z') && !/[+-]\d{2}:\d{2}$/.test(dateStr)) {
    const [datePart, timePart] = dateStr.split('T')
    const [y, m, d] = datePart.split('-').map(Number)
    const [hh, mm, ss = 0] = timePart.split(':').map(Number)
    return new Date(y, m - 1, d, hh, mm, ss)
  }
  const d = new Date(dateStr)
  return isNaN(d.getTime()) ? null : d
}

/** 当前北京时间，写入数据库用（ISO +08:00） */
export function nowBeijingISO(): string {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(new Date())
  const get = (type: string) => parts.find(p => p.type === type)?.value ?? '00'
  return `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}:${get('second')}+08:00`
}

export function isAppointmentToday(dateStr: string): boolean {
  const d = parseAppointmentTime(dateStr)
  if (!d) return false
  const a = getBeijingDateParts(d)
  const n = getBeijingDateParts(new Date())
  return a.y === n.y && a.m === n.m && a.day === n.day
}

export function isAppointmentThisMonth(dateStr: string): boolean {
  const d = parseAppointmentTime(dateStr)
  if (!d) return false
  const a = getBeijingDateParts(d)
  const n = getBeijingDateParts(new Date())
  return a.y === n.y && a.m === n.m
}

export function formatTimeOnly(dateStr: string): string {
  const d = parseAppointmentTime(dateStr)
  if (!d) return '-'
  return d.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour: '2-digit', minute: '2-digit' })
}

export function formatDateTime(dateStr: string): string {
  const d = parseAppointmentTime(dateStr)
  if (!d) return '-'
  return d.toLocaleString('zh-CN', {
    timeZone: 'Asia/Shanghai',
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function toDatetimeLocal(dateStr: string): string {
  const d = parseAppointmentTime(dateStr)
  if (!d) return ''
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(d)
  const get = (type: string) => parts.find(p => p.type === type)?.value ?? '00'
  return `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}`
}

export function fromDatetimeLocal(local: string): string {
  const d = parseAppointmentTime(local)
  if (!d) throw new Error('无效的就诊时间')
  return d.toISOString()
}

export interface RegistrationRow {
  id: string
  patient_id: string
  doctor_id: string
  department_name: string
  appointment_time: string
  chief_complaint: string
  purpose: string
  has_covid_symptoms: boolean
  status: RegistrationStatus
  doctors?: { name: string; department: string } | null
}

export interface RegStats {
  todayCount: number
  monthCount: number
}

export function computeRegStats(regs: { appointment_time: string; status: string }[]): RegStats {
  let todayCount = 0
  let monthCount = 0
  for (const r of regs) {
    if (normalizeRegistrationStatus(r.status) === 'cancelled') continue
    if (isAppointmentThisMonth(r.appointment_time)) monthCount++
    if (isAppointmentToday(r.appointment_time)) todayCount++
  }
  return { todayCount, monthCount }
}
