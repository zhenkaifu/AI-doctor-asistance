// @ts-nocheck
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const TABLE_MAP: Record<string, string> = {
  admin: 'admins',
  doctor: 'doctors',
  nurse: 'nurses',
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers':
          'authorization, x-client-info, apikey, content-type',
      },
    })
  }

  try {
    const { userId, name, phone, department, role } = await req.json()
    if (!userId) throw new Error('Missing userId')

    // ── 1. 验证管理员身份 ──────────────────────────────────
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

    if (!adminRow) throw new Error('Forbidden: only admin can update staff')

    // ── 2. 使用 service_role 操作 ──────────────────────────
    const serviceClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    // ── 3. 禁止修改管理员 ──────────────────────────────────
    const { data: targetAdmin } = await serviceClient
      .from('admins')
      .select('auth_id')
      .eq('auth_id', userId)
      .maybeSingle()

    if (targetAdmin) {
      throw new Error('不能修改管理员账号')
    }

    // ── 4. 更新 auth 用户 metadata ─────────────────────────
    if (name) {
      await serviceClient.auth.admin.updateUserById(userId, {
        user_metadata: { name: name.trim() },
      })
    }

    // ── 5. 构建更新数据（null 值跳过，保留原值）────────────
    const row: any = {}
    if (name != null) row.name = name.trim()
    if (phone != null) row.phone = phone.trim()
    if (department != null) row.department = department.trim()

    // 如果指定了 role 且与当前不同，则需要迁移表
    if (role && role !== 'admin') {
      const newTable = TABLE_MAP[role]

      for (const oldTable of ['doctors', 'nurses']) {
        if (oldTable === newTable) continue
        await serviceClient.from(oldTable).delete().eq('auth_id', userId)
      }

      const { data: existing } = await serviceClient
        .from(newTable)
        .select('auth_id')
        .eq('auth_id', userId)
        .maybeSingle()

      if (existing) {
        const { error: updateErr } = await serviceClient
          .from(newTable)
          .update(row)
          .eq('auth_id', userId)
        if (updateErr) throw updateErr
      } else {
        const { error: insertErr } = await serviceClient
          .from(newTable)
          .insert({ auth_id: userId, ...row })
        if (insertErr) throw insertErr
      }
    } else {
      for (const table of ['doctors', 'nurses']) {
        const { data: exist } = await serviceClient
          .from(table)
          .select('auth_id')
          .eq('auth_id', userId)
          .maybeSingle()
        if (exist) {
          const { error: updateErr } = await serviceClient
            .from(table)
            .update(row)
            .eq('auth_id', userId)
          if (updateErr) throw updateErr
          break
        }
      }
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    })
  } catch (e) {
    return new Response(
      JSON.stringify({ error: e.message || 'Internal error' }),
      {
        status: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      },
    )
  }
})
