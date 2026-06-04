// @ts-nocheck
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

interface StaffMember {
  id: string
  email: string
  name: string
  role: string
  department: string | null
  phone: string | null
  created_at: string
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
    if (callerError || !caller) throw new Error('Unauthorized')

    // ── 2. 确认调用者是管理员 ─────────────────────────────────
    const { data: adminRow } = await supabaseClient
      .from('admins')
      .select('auth_id')
      .eq('auth_id', caller.id)
      .maybeSingle()

    if (!adminRow) throw new Error('Forbidden')

    // ── 3. 使用 service_role 获取数据 ───────────────────────
    const serviceClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    // 3a. 获取所有 auth 用户
    const { data: usersData, error: listError } =
      await serviceClient.auth.admin.listUsers()
    if (listError) throw listError

    // 3b. 并行查询三张表
    const [adminsRes, doctorsRes, nursesRes] = await Promise.all([
      serviceClient.from('admins').select('*'),
      serviceClient.from('doctors').select('*'),
      serviceClient.from('nurses').select('*'),
    ])

    // ── 4. 用 auth_id 匹配 auth.users，合并数据 ────────────
    const userMap = new Map<string, any>(
      (usersData?.users || []).map((u) => [u.id, u]),
    )

    const buildStaff = (rows: any[], role: string): StaffMember[] =>
      (rows || [])
        .filter((r) => r.auth_id && userMap.has(r.auth_id))
        .map((r) => {
          const u = userMap.get(r.auth_id)!
          return {
            id: u.id,
            email: u.email ?? '',
            name: r.name || u.email,
            role,
            department: r.department || null,
            phone: r.phone || null,
            created_at: u.created_at,
          }
        })

    const staff: StaffMember[] = [
      ...buildStaff(adminsRes.data || [], 'admin'),
      ...buildStaff(doctorsRes.data || [], 'doctor'),
      ...buildStaff(nursesRes.data || [], 'nurse'),
    ].sort(
      (a, b) =>
        new Date(b.created_at || 0).getTime() -
        new Date(a.created_at || 0).getTime(),
    )

    return new Response(JSON.stringify(staff), {
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
