-- ============================================================
-- ui_metadata 백업 스크립트 (V8 실행 전 실행)
-- 실행 방법:
--   docker exec -i sdui-db psql -U mina -d SDUI_TD < scripts/backup_ui_metadata.sql
-- ============================================================

-- 기존 백업 테이블이 있으면 덮어쓰기
DROP TABLE IF EXISTS ui_metadata_backup_v7;

-- 현재 ui_metadata 전체를 백업 테이블로 복사 (구조 + 데이터 모두)
CREATE TABLE ui_metadata_backup_v7 AS
SELECT * FROM ui_metadata;

-- 백업 확인
SELECT COUNT(*) AS backed_up_rows FROM ui_metadata_backup_v7;

DO $$
BEGIN
    RAISE NOTICE '✅ 백업 완료: ui_metadata_backup_v7 생성됨';
    RAISE NOTICE '   복구 시 scripts/restore_ui_metadata.sql 실행';
END $$;