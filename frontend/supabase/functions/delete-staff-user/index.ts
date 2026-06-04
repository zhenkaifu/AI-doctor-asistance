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
    const { userId } = await req.json()
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

    if (!adminRow) throw new Error('Forbidden: only admin can delete staff')

    // ── 2. 禁止删除管理员 ──────────────────────────────────
    const serviceClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    // 检查目标是不是管理员
    const { data: targetAdmin } = await serviceClient
      .from('admins')
      .select('auth_id')
      .eq('auth_id', userId)
      .maybeSingle()

    if (targetAdmin) {
      throw new Error('不能删除管理员账号')
    }

    // ── 3. 从角色表中删除 ──────────────────────────────────
    for (const table of ['doctors', 'nurses']) {
      const { error: delErr } = await serviceClient
        .from(table)
        .delete()
        .eq('auth_id', userId)
      if (delErr) throw delErr
    }

    // ── 4. 删除 auth 用户 ──────────────────────────────────
    const { error: authErr } = await serviceClient.auth.admin.deleteUser(userId)
    if (authErr) throw authErr

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
