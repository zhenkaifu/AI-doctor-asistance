-- 与数据库约束 valid_registration_status 保持一致
-- 在 Supabase SQL Editor 中执行（若约束已存在可只执行数据迁移部分）

ALTER TABLE registrations
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'waiting';

-- 迁移历史数据：active -> waiting
UPDATE registrations SET status = 'waiting' WHERE status = 'active';

-- 替换旧约束
ALTER TABLE registrations DROP CONSTRAINT IF EXISTS registrations_status_check;
ALTER TABLE registrations DROP CONSTRAINT IF EXISTS valid_registration_status;

ALTER TABLE registrations
  ADD CONSTRAINT valid_registration_status
  CHECK (status = ANY (ARRAY['waiting'::text, 'completed'::text, 'cancelled'::text]));

ALTER TABLE registrations ALTER COLUMN status SET DEFAULT 'waiting';
