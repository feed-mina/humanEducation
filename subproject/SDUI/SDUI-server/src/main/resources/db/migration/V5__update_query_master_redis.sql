-- V5: query_master Redis 캐싱 컬럼 추가
-- Redis 캐싱 제어를 위한 컬럼을 추가합니다.

-- use_redis_yn 컬럼 추가 (없을 경우)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'query_master' AND column_name = 'use_redis_yn'
    ) THEN
        ALTER TABLE query_master
        ADD COLUMN use_redis_yn character(1) DEFAULT 'N';

        COMMENT ON COLUMN query_master.use_redis_yn IS 'Redis 캐싱 사용 여부 (Y/N)';
    END IF;
END $$;

-- redis_ttl_sec 컬럼 추가 (없을 경우)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'query_master' AND column_name = 'redis_ttl_sec'
    ) THEN
        ALTER TABLE query_master
        ADD COLUMN redis_ttl_sec integer DEFAULT 3600;

        COMMENT ON COLUMN query_master.redis_ttl_sec IS 'Redis 캐시 만료 시간 (초 단위, 기본 3600초=1시간)';
    END IF;
END $$;

-- param_mapping 컬럼 추가 (없을 경우)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'query_master' AND column_name = 'param_mapping'
    ) THEN
        ALTER TABLE query_master
        ADD COLUMN param_mapping jsonb DEFAULT NULL;

        COMMENT ON COLUMN query_master.param_mapping IS '파라미터 매핑 정보 (JSON 형식)';
    END IF;
END $$;

DO $$ BEGIN RAISE NOTICE 'V5: query_master Redis 캐싱 컬럼 추가 완료'; END $$;
