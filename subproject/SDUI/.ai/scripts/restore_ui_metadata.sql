-- ============================================================
-- ui_metadata 복구 스크립트 (V8 롤백 시 실행)
-- 실행 방법:
--   docker exec -i sdui-db psql -U mina -d SDUI_TD < scripts/restore_ui_metadata.sql
-- ============================================================

-- 백업 테이블 존재 확인
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'ui_metadata_backup_v7'
    ) THEN
        RAISE EXCEPTION '❌ ui_metadata_backup_v7 테이블이 없습니다. 백업을 먼저 실행하세요.';
    END IF;
END $$;

BEGIN;

-- 현재 ui_metadata 전체 삭제 후 백업 데이터로 복구
TRUNCATE TABLE ui_metadata;

INSERT INTO ui_metadata
SELECT * FROM ui_metadata_backup_v7;

-- 복구 확인
SELECT COUNT(*) AS restored_rows FROM ui_metadata;

RAISE NOTICE '✅ 복구 완료: ui_metadata_backup_v7 → ui_metadata';

COMMIT;
