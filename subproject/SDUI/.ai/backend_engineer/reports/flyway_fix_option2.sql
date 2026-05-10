-- Flyway checksum 문제 해결 SQL (Option 2)
-- 작성일: 2026-03-03
-- 방법: V3-V7 레코드 삭제 후 Flyway baseline을 V7로 재설정

-- Step 1: 현재 상태 확인
SELECT installed_rank, version, description, checksum, success
FROM flyway_schema_history
ORDER BY installed_rank;

-- Step 2: V3-V7 레코드 삭제 (데이터는 이미 마이그레이션되어 있음)
DELETE FROM flyway_schema_history WHERE version IN ('3', '4', '5', '6', '7');

-- Step 3: V7을 새로운 baseline으로 설정
INSERT INTO flyway_schema_history (
    installed_rank,
    version,
    description,
    type,
    script,
    checksum,
    installed_by,
    installed_on,
    execution_time,
    success
) VALUES (
    3,  -- installed_rank (V0=1, V1=2이므로 V7=3)
    '7',  -- version
    'baseline at version 7 after manual migration',  -- description
    'BASELINE',  -- type
    'V7__drop_diary_table.sql',  -- script
    NULL,  -- checksum (baseline은 NULL)
    'mina',  -- installed_by
    NOW(),  -- installed_on
    0,  -- execution_time
    true  -- success
);

-- Step 4: 결과 확인
SELECT installed_rank, version, description, type, success
FROM flyway_schema_history
ORDER BY installed_rank;

-- 참고: 이 방법을 사용하면 Flyway는 V3-V7이 이미 적용된 것으로 간주합니다.
-- 컨테이너 시작 시 더 이상 validation 에러가 발생하지 않습니다.
