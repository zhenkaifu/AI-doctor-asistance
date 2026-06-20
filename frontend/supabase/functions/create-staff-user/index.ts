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
    const { email, password, name, role, department, phone } = await req.json()

    // ── 1. 验证 JWT ────────────────────────────────────────
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
    if (callerError || !caller) throw new Error('Unauthorized: invalid token')

    // ── 2. 确认调用者是管理员（查 admins 表）───────────────
    const { data: adminRow } = await supabaseClient
      .from('admins')
      .select('auth_id')
      .eq('auth_id', caller.id)
      .maybeSingle()

    if (!adminRow) {
      throw new Error('Forbidden: only admin can create staff users')
    }

    // ── 3. 使用 service_role 创建 auth 用户 ─────────────────
    const serviceClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    const { data: newAuth, error: createError } =
      await serviceClient.auth.admin.createUser({
        email: email.trim(),
        password,
        email_confirm: true,
        user_metadata: { name: name.trim() },
      })

    if (createError || !newAuth.user) throw createError

    const userId = newAuth.user.id
    const tableName = TABLE_MAP[role] || 'doctors'

    // ── 4. 插入对应角色表 ──────────────────────────────────
    const row: any = {
      auth_id: userId,
      name: name.trim(),
      phone: phone?.trim() || null,
    }
    if (role === 'doctor') {
      row.department = department?.trim() || null
    }

    const { error: insertError } = await serviceClient
      .from(tableName)
      .insert(row)

    // ── 5. 插入失败则回滚删除 auth 用户 ─────────────────────
    if (insertError) {
      await serviceClient.auth.admin.deleteUser(userId)
      throw insertError
    }

    return new Response(
      JSON.stringify({ success: true, userId, email: newAuth.user.email }),
      {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      },
    )
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
