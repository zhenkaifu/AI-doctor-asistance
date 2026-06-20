// @ts-nocheck
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers':
    'authorization, x-client-info, apikey, content-type',
}

function beijingDateKey(iso: string): string | null {
  const d = new Date(iso)
  if (isNaN(d.getTime())) return null
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai' }).format(d)
}

function last30DaysKeys(): string[] {
  const formatter = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai' })
  const keys: string[] = []
  const now = new Date()
  for (let i = 29; i >= 0; i--) {
    const d = new Date(now.getTime() - i * 86400000)
    keys.push(formatter.format(d))
  }
  return keys
}

function buildDailySeries(
  keys: string[],
  rows: { dateKey: string | null }[],
): { date: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const k of keys) counts.set(k, 0)
  for (const r of rows) {
    if (!r.dateKey || !counts.has(r.dateKey)) continue
    counts.set(r.dateKey, (counts.get(r.dateKey) ?? 0) + 1)
  }
  return keys.map((date) => ({ date, count: counts.get(date) ?? 0 }))
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: CORS })
  }

  try {
    const { userId, role } = await req.json()
    if (!userId || !role) throw new Error('缺少 userId 或 role')
    if (role !== 'nurse' && role !== 'doctor') {
      throw new Error('role 须为 nurse 或 doctor')
    }

    const authHeader = req.headers.get('Authorization')
    if (!authHeader) throw new Error('Missing Authorization header')

    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      { global: { headers: { Authorization: authHeader } } },
    )

    const {
      data: { user: caller },
      error: callerError,
    } = await supabaseClient.auth.getUser()
    if (callerError || !caller) throw new Error('Unauthorized')

    const { data: adminRow } = await supabaseClient
      .from('admins')
      .select('auth_id')
      .eq('auth_id', caller.id)
      .maybeSingle()
    if (!adminRow) throw new Error('Forbidden')

    const serviceClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    const dateKeys = last30DaysKeys()
    const rangeStart = `${dateKeys[0]}T00:00:00+08:00`

    let staffName = ''
    let daily: { date: string; count: number }[] = []

    if (role === 'nurse') {
      const { data: nurseRow, error: nurseErr } = await serviceClient
        .from('nurses')
        .select('id, name')
        .eq('auth_id', userId)
        .maybeSingle()
      if (nurseErr) throw nurseErr
      if (!nurseRow) throw new Error('未找到该护士')

      staffName = nurseRow.name || ''

      const { data: regs, error: regErr } = await serviceClient
        .from('registrations')
        .select('appointment_time, status')
        .eq('nurse_id', nurseRow.id)
        .gte('appointment_time', rangeStart)
        .neq('status', 'cancelled')
      if (regErr) throw regErr

      daily = buildDailySeries(
        dateKeys,
        (regs || []).map((r) => ({
          dateKey: beijingDateKey(r.appointment_time),
        })),
      )
    } else {
      const { data: doctorRow, error: doctorErr } = await serviceClient
        .from('doctors')
        .select('name')
        .eq('auth_id', userId)
        .maybeSingle()
      if (doctorErr) throw doctorErr
      if (!doctorRow) throw new Error('未找到该医生')

      staffName = doctorRow.name || ''

      const { data: records, error: recErr } = await serviceClient
        .from('medical_records')
        .select('visit_date')
        .eq('doctor_id', userId)
        .gte('visit_date', rangeStart)
      if (recErr) throw recErr

      daily = buildDailySeries(
        dateKeys,
        (records || []).map((r) => ({
          dateKey: beijingDateKey(r.visit_date),
        })),
      )
    }

    const total = daily.reduce((s, d) => s + d.count, 0)

    return new Response(
      JSON.stringify({
        staffName,
        role,
        daily,
        total,
        average: Math.round((total / 30) * 10) / 10,
      }),
      { headers: { 'Content-Type': 'application/json', ...CORS } },
    )
  } catch (e) {
    return new Response(
      JSON.stringify({ error: e.message || 'Internal error' }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json', ...CORS },
      },
    )
  }
})
