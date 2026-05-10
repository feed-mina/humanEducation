-- V7: diary 테이블 삭제
-- 데이터 마이그레이션이 완료되고 diary_backup이 있으므로 원본 diary 테이블을 삭제합니다.
-- 주의: 이 마이그레이션은 프로덕션에서 신중히 실행해야 합니다.

-- diary 테이블 존재 여부 확인 및 삭제
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'diary') THEN
        -- 최종 확인: diary_backup에 데이터가 있는지 확인
        IF (SELECT COUNT(*) FROM diary_backup) > 0 THEN
            RAISE NOTICE 'diary 테이블 삭제 - backup에 % 건의 데이터가 안전하게 저장되어 있습니다.', (SELECT COUNT(*) FROM diary_backup);
            DROP TABLE diary CASCADE;
            RAISE NOTICE 'diary 테이블 삭제 완료';
        ELSIF (SELECT COUNT(*) FROM diary) = 0 THEN
            RAISE NOTICE 'diary 테이블에 데이터가 없으므로 삭제합니다.';
            DROP TABLE diary CASCADE;
        ELSE
            RAISE EXCEPTION 'diary 테이블에 데이터가 있지만 backup이 비어있습니다. 마이그레이션을 중단합니다.';
        END IF;
    ELSE
        RAISE NOTICE 'diary 테이블이 이미 삭제되었거나 존재하지 않습니다.';
    END IF;
END $$;

-- 최종 검증: content 테이블이 존재하고 데이터가 있는지 확인
DO $$
DECLARE
    content_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO content_count FROM content;

    IF content_count = 0 THEN
        RAISE WARNING 'content 테이블에 데이터가 없습니다. 마이그레이션을 확인하세요.';
    ELSE
        RAISE NOTICE 'V7 완료 - content 테이블에 % 건의 데이터가 있습니다.', content_count;
    END IF;
END $$;
