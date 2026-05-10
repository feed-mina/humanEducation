-- V4: ui_metadata RBAC 컬럼 추가
-- RBAC(Role-Based Access Control) 지원을 위한 컬럼을 추가합니다.

-- allowed_roles 컬럼 추가 (없을 경우)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ui_metadata' AND column_name = 'allowed_roles'
    ) THEN
        ALTER TABLE ui_metadata
        ADD COLUMN allowed_roles character varying(255) DEFAULT NULL;

        COMMENT ON COLUMN ui_metadata.allowed_roles IS '접근 가능한 역할 목록 (예: "ROLE_USER,ROLE_ADMIN"). NULL이면 모두 허용';
    END IF;
END $$;

-- label_text_overrides 컬럼 추가 (없을 경우)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ui_metadata' AND column_name = 'label_text_overrides'
    ) THEN
        ALTER TABLE ui_metadata
        ADD COLUMN label_text_overrides jsonb DEFAULT NULL;

        COMMENT ON COLUMN ui_metadata.label_text_overrides IS '역할별 label_text 오버라이드 (예: {"ROLE_ADMIN":"관리자용","ROLE_USER":"사용자용"})';
    END IF;
END $$;

-- css_class_overrides 컬럼 추가 (없을 경우)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ui_metadata' AND column_name = 'css_class_overrides'
    ) THEN
        ALTER TABLE ui_metadata
        ADD COLUMN css_class_overrides jsonb DEFAULT NULL;

        COMMENT ON COLUMN ui_metadata.css_class_overrides IS '역할별 css_class 오버라이드 (예: {"ROLE_ADMIN":"btn-danger","ROLE_USER":"btn-primary"})';
    END IF;
END $$;

DO $$ BEGIN RAISE NOTICE 'V4: ui_metadata RBAC 컬럼 추가 완료'; END $$;
