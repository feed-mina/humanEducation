-- =================================================================
-- 이 파일은 관리자 페이지의 회원 권한 관리를 위해 'query_master' 테이블에 추가될 쿼리 예시입니다.
-- 실제 환경에 맞게 테이블명, 컬럼명 등을 수정하여 사용하세요.
-- =================================================================

-- [1] 회원 목록 조회 및 검색 쿼리
-- query_id: find_users_for_admin
-- 설명: 관리자 페이지에서 사용할 회원 목록을 조회합니다. 이름 또는 이메일로 검색하는 기능을 포함합니다.
--      검색어가 없는 경우 전체 목록을 반환합니다.

SELECT
    user_id,
    username,
    email,
    role
FROM
    users  -- 'users' 테이블은 실제 사용하는 회원 테이블명으로 변경해야 합니다.
WHERE
    (LOWER(username) LIKE LOWER('%${searchTerm}%') OR LOWER(email) LIKE LOWER('%${searchTerm}%'))
ORDER BY
    created_at DESC;

-- 파라미터:
-- ${searchTerm}: 프론트엔드에서 전달받는 검색어. MyBatis/JPA 등에서 동적 쿼리로 처리하거나, 값이 없을 경우 빈 문자열 ''을 전달합니다.


-- [2] 선택된 회원의 권한 일괄 변경 쿼리
-- query_id: update_user_roles
-- 설명: 관리자가 선택한 여러 회원의 권한(role)을 한번에 변경합니다.
-- 중요: 이 쿼리를 실행하기 전, 서비스 계층(Service Layer)에서 반드시 비즈니스 로직을 처리해야 합니다.
--      - 요청된 회원의 수가 1명 이상 5명 이하인지 확인해야 합니다.

UPDATE
    users  -- 'users' 테이블은 실제 사용하는 회원 테이블명으로 변경해야 합니다.
SET
    role = #{newRole}
WHERE
    user_id IN
        <foreach item="userId" collection="userIds" open="(" separator="," close=")">
            #{userId}
        </foreach>

-- 파라미터 (MyBatis 예시):
-- #{newRole}: 변경할 새로운 권한 (e.g., 'USER', 'MANAGER', 'ADMIN')
-- collection="userIds": 권한을 변경할 회원 ID 목록 (List<String> 또는 String[]). 서비스 로직에서 1~5개로 필터링된 목록이어야 합니다.

