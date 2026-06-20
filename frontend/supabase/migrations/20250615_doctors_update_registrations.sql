-- 医生完成挂号：允许医生更新分配给自己的 registrations（如 status -> completed）
-- 原因：原 RLS 仅授予医生 SELECT，UPDATE 被静默拒绝（0 行更新）

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policy
    WHERE polrelid = 'public.registrations'::regclass
      AND polname = 'Doctors can complete own registrations'
  ) THEN
    CREATE POLICY "Doctors can complete own registrations"
    ON public.registrations
    FOR UPDATE
    TO authenticated
    USING (
      EXISTS (
        SELECT 1 FROM public.doctors d
        WHERE d.auth_id = auth.uid()
          AND d.id = registrations.doctor_id
      )
    )
    WITH CHECK (
      EXISTS (
        SELECT 1 FROM public.doctors d
        WHERE d.auth_id = auth.uid()
          AND d.id = registrations.doctor_id
      )
    );
  END IF;
END $$;
