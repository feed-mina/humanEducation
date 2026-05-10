-- V15__add_required_role_to_query_master.sql
-- [P0 Security Fix] query_master 테이블에 required_role 컬럼 추가
--
-- 목적: /api/execute/{sqlKey} 엔드포인트의 무인증 SQL 실행 취약점 해소
-- 방식: 각 쿼리에 실행 권한 요구사항을 명시 → CommonQueryController에서 검증
--
-- required_role 값 규칙:
--   NULL        → 누구나 실행 가능 (기존 동작 유지)
--   'ROLE_USER' → 로그인한 사용자만 실행 가능
--   'ROLE_ADMIN'→ ADMIN 권한 보유자만 실행 가능

ALTER TABLE query_master
    ADD COLUMN IF NOT EXISTS required_role VARCHAR(50) DEFAULT NULL;

COMMENT ON COLUMN query_master.required_role
    IS '실행 권한 요구사항: NULL=공개, ROLE_USER=로그인 필요, ROLE_ADMIN=관리자 전용';

-- 기존 등록된 쿼리들의 required_role 설정
-- 사용자별 데이터를 조회하는 쿼리는 ROLE_USER로 보호
UPDATE query_master SET required_role = 'ROLE_USER'
WHERE sql_key IN (
    'GET_MY_GOAL_TIME',
    'GET_GOAL_LIST',
    'GET_CONTENT_LIST',
    'GET_MY_CONTENT'
);

-- 관리자 전용 쿼리 (V14에서 등록됨)
UPDATE query_master SET required_role = 'ROLE_ADMIN'
WHERE sql_key IN (
    'GET_SYSTEM_LOGS'
);

-- ※ sql_key 목록은 실제 query_master 데이터에 맞게 조정 필요
--   (현재 등록된 키를 확인 후 적용: SELECT sql_key FROM query_master;)
