已将敏感信息替换为占位符，并添加了安全提醒。以下是修改后的 `README.md` 内容，可直接用于 GitHub 公开仓库：

---

```markdown
# 🏥 AI 医生助手 —— 医院后台管理系统

一个基于 **Vue 3 + Supabase + Edge Functions** 的医护人员管理平台，支持管理员增删改查医生/护士账号，并与 Supabase 认证系统深度集成。

> 🔑 本项目依赖 Supabase 服务，运行前请自行配置项目地址、密钥并部署 Edge Functions。

---

## 🚀 快速开始（本地运行）

### 1. 克隆项目
```bash
git clone https://github.com/your-username/AI-doctor-asistance.git
cd AI-doctor-asistance
```

### 2. 安装依赖（前后端）
```bash
# 安装前端依赖（frontend）
cd frontend && npm install

# 安装后端依赖（backend，如需启动 Nest.js API）
cd ../backend && npm install
```

### 3. 配置 Supabase 环境变量（关键！）
在 `frontend/.env.local` 中填入你的 Supabase 项目信息：
```env
# Supabase 项目配置（必填，请替换为你的项目信息）
VITE_SUPABASE_URL=https://<YOUR_SUPABASE_PROJECT>.supabase.co
VITE_SUPABASE_ANON_KEY=<YOUR_SUPABASE_ANON_KEY>

# DeepSeek API（用于生成病历单，可选）
VITE_DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
```

> ⚠️ **安全警告**：`VITE_SUPABASE_ANON_KEY` 是公开密钥，但仍应避免直接提交到公开仓库。建议将 `.env.local` 加入 `.gitignore`，并参考 `.env.example` 提供模板。

### 4. 启动前端开发服务器
```bash
cd frontend
npm run dev
```
访问 `http://localhost:5173` 即可使用！

---

## ⚙️ Supabase 初始化（首次部署必读）

本项目使用 Supabase 提供的数据库表结构：
- `admins`（管理员）
- `doctors`（医生）
- `nurses`（护士）

### ✅ 数据库初始化（只需一次）
1. 登录 [Supabase Dashboard](https://supabase.com/dashboard)
2. 进入 **SQL Editor**，执行以下建表语句（若表不存在）：

```sql
-- 创建 admins 表（管理员）
CREATE TABLE IF NOT EXISTS admins (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_id UUID UNIQUE NOT NULL,
  name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 创建 doctors 表（医生）
CREATE TABLE IF NOT EXISTS doctors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_id UUID UNIQUE NOT NULL,
  name TEXT NOT NULL,
  department TEXT,
  phone TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 创建 nurses 表（护士）
CREATE TABLE IF NOT EXISTS nurses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_id UUID UNIQUE NOT NULL,
  name TEXT NOT NULL,
  phone TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

> ✅ 提示：如果你的项目中已有这些表，可跳过此步。

---

## 🌐 Edge Functions 部署（核心！必须操作）

所有员工管理逻辑由 Supabase Edge Functions 实现，需手动部署：

### ✅ 前提：安装 Supabase CLI
```bash
brew install supabase/tap/supabase
```

### ✅ 步骤（在 `frontend/` 目录下执行）
```bash
cd frontend

# 1. 登录 Supabase（会自动打开浏览器）
supabase login

# 2. 关联你的项目（请将 <YOUR_PROJECT_REF> 替换为实际项目引用 ID）
supabase link --project-ref <YOUR_PROJECT_REF>

# 3. 部署全部函数（共 4 个）
supabase functions deploy list-all-staff
supabase functions deploy create-staff-user
supabase functions deploy update-staff-user
supabase functions deploy delete-staff-user
```

> ✅ 成功标志：终端显示 `Deployed Functions on project <YOUR_PROJECT_REF>`

---

## 👥 使用流程

| 角色 | 操作 |
|------|------|
| **管理员** | 1. 访问 `http://localhost:5173/login`，用邮箱密码登录<br>2. 跳转至 `/admin` → 点击「医护人员管理」→ `/admin/staff`<br>3. 可新增、编辑、删除医生/护士（⚠️ 不可操作管理员账号） |
| **医生/护士** | 1. 收到管理员创建的账号邮件<br>2. 访问 `http://localhost:5173/login` 登录<br>3. 自动跳转至病人查询页 `/`，进入诊疗助手 |

> 🔐 权限保障：所有敏感操作（删账号、跨表查询）均由 Edge Functions 强制校验 JWT + `admins` 表身份，前端无权限绕过。

---

## 📁 项目结构概览
```
AI-doctor-asistance/
├── frontend/          # Vue 3 前端（Vite）
│   ├── src/views/StaffManagementPage.vue    ← 员工管理主页面
│   ├── supabase/functions/                 ← 所有 Edge Functions 源码
│   └── .env.local                        ← Supabase 密钥配置（需自行创建）
├── backend/           # Nest.js 后端（ASR/语音识别服务，可选）
├── python-asr/       # Python ASR 推理脚本（可选）
└── README.md         ← 本文件
```

---

## 🧩 技术栈
- **前端**：Vue 3 + TypeScript + Vite + Tailwind CSS  
- **后端能力**：Supabase Auth + PostgreSQL + Edge Functions（Deno）  
- **认证**：Supabase Email/Password + RLS（行级安全）  
- **扩展性**：所有业务逻辑封装在 Edge Functions，便于后续对接 LLM 或审计日志

---

## 📬 贡献 & 支持
欢迎提交 Issue 或 PR！  
如有部署问题，请检查：
- ✅ `supabase link` 是否成功  
- ✅ Edge Functions 是否全部 `deploy` 成功  
- ✅ 浏览器控制台是否报 `403 Forbidden`（说明管理员身份未通过校验）

> 🛡️ **安全提醒**：请勿将真实的 `service_role_key`、`ANON_KEY` 或 API 密钥提交到 Git 仓库。建议使用环境变量并在 `.gitignore` 中排除敏感文件。

---

© 2026 AI Doctor Assistant — Designed for Hospital Admins, Built with ❤️ and Supabase
```

所有敏感信息（Supabase URL、anon key、DeepSeek API key、project ref）均已替换为占位符 `<YOUR_...>`，并添加了安全提醒，可安全发布到 GitHub。
