-- V25: query_master 핵심 SQL 키 시드
-- AWS DB에는 존재하지만 로컬 Flyway 마이그레이션에 누락된 쿼리들을 추가합니다.
-- WHERE NOT EXISTS로 중복 방지 (이미 존재하면 SKIP)
-- V13(GET_ADMIN_STATS), V14(GET_SYSTEM_LOGS), V23(목표시간 3개) 는 제외

-- ─────────────────────────────────────────────────────────
-- 콘텐츠 목록 페이징 조회
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_CONTENT_LIST_PAGE',
       'SELECT d.content_id AS "content_id", d.user_sqno AS "user_sqno", d.title AS "title", d.date AS "date", d.emotion AS "emotion", d.day_tag1 AS "tag1", d.day_tag2 AS "tag2", d.day_tag3 AS "tag3", d.content_status AS "content_status", d.reg_dt AS "reg_dt", CASE WHEN u.del_yn = ''Y'' THEN CONCAT(''del_'', d.user_id) ELSE d.user_id END AS "user_id"
          FROM content d
          LEFT JOIN users u ON d.user_sqno = u.user_sqno
         WHERE d.del_yn = ''N''
           AND (NULLIF(CAST(:filterId AS VARCHAR), '''') IS NULL OR d.user_id = CAST(:filterId AS VARCHAR))
         ORDER BY d.reg_dt DESC
         LIMIT  CAST(COALESCE(NULLIF(CAST(:pageSize AS VARCHAR), ''''), ''5'') AS INTEGER)
         OFFSET CAST(COALESCE(NULLIF(CAST(:offset  AS VARCHAR), ''''), ''0'') AS INTEGER)',
       'MULTI', '콘텐츠 목록 페이징 조회', 'Y', 300, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_CONTENT_LIST_PAGE');

-- ─────────────────────────────────────────────────────────
-- 전체 콘텐츠 개수 조회 (페이징 카운터)
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'COUNT_CONTENT_LIST',
       'SELECT COUNT(*) AS total_count
          FROM content d
         WHERE d.del_yn = ''N''
           AND (:userId IS NULL OR :userId = '''' OR d.user_id = :userId)',
       'SINGLE', '전체 콘텐츠 개수 조회', 'N', 0, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'COUNT_CONTENT_LIST');

-- ─────────────────────────────────────────────────────────
-- 콘텐츠 상세 정보 조회
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_CONTENT_DETAIL',
       'SELECT d.content_id, d.user_id, d.title, d.content, d.date, d.emotion,
               d.day_tag1, d.day_tag2, d.day_tag3, d.content_status, d.role_nm,
               d.selected_times, d.daily_slots, d.reg_dt
          FROM content d
         WHERE d.content_id = CAST(:contentId AS BIGINT)
           AND d.del_yn = ''N''',
       'SINGLE', '콘텐츠 상세 정보 조회 (JSON 데이터 포함)', 'N', 3600, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_CONTENT_DETAIL');

-- ─────────────────────────────────────────────────────────
-- 로그인 사용자의 콘텐츠 목록 조회
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_MEMBER_CONTENT_LIST',
       'SELECT content_id AS "content_id", title AS "title", content AS "content",
               reg_dt AS "reg_dt", img_url AS "img_url", user_id AS "user_id"
          FROM content
         WHERE user_sqno = :userSqno
           AND del_yn = ''N''
         ORDER BY reg_dt DESC
         LIMIT :pageSize OFFSET :offset',
       'MULTI', '로그인한 사용자의 콘텐츠 목록 조회', 'Y', 600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_MEMBER_CONTENT_LIST');

-- ─────────────────────────────────────────────────────────
-- 신규 콘텐츠 작성
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'INSERT_CONTENT',
       'INSERT INTO content (
            user_sqno, user_id, title, content, emotion,
            selected_times, daily_slots, day_tag1, day_tag2, day_tag3,
            reg_dt
        ) VALUES (
            :userSqno, :userId, :title, :content, CAST(:emotion AS INTEGER),
            CAST(:selected_times AS jsonb), CAST(:daily_slots AS jsonb),
            :day_tag1, :day_tag2, :day_tag3,
            NOW()
        )',
       'COMMAND', '신규 콘텐츠 작성 (JSONB 타입 및 day_tag 반영)', 'N', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'INSERT_CONTENT');

-- ─────────────────────────────────────────────────────────
-- 콘텐츠 수정
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'UPDATE_CONTENT_DETAIL',
       'UPDATE content
           SET title = :title,
               content = :content,
               emotion = CAST(:emotion AS INTEGER),
               selected_times = CAST(:selected_times AS jsonb),
               daily_slots = CAST(:daily_slots AS jsonb),
               day_tag1 = :day_tag1,
               day_tag2 = :day_tag2,
               day_tag3 = :day_tag3
         WHERE content_id = CAST(:content_id AS BIGINT)
           AND user_sqno = :userSqno',
       'COMMAND', '콘텐츠 내용 수정 쿼리', 'N', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'UPDATE_CONTENT_DETAIL');

-- ─────────────────────────────────────────────────────────
-- 콘텐츠 삭제 (soft delete)
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'UPDATE_CONTENT_DELETE',
       'UPDATE content
           SET del_yn = ''Y'',
               del_dt = NOW(),
               last_updt_ip = :lastUpdtIp,
               last_updt_usps_sqno = :lastUpdtUspsSqno
         WHERE content_id = ANY(:contentIdList)
           AND (:userSqno IS NULL OR user_sqno = :userSqno)',
       'COMMAND', '선택한 콘텐츠 일괄 삭제 처리 (본인 확인 포함)', 'N', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'UPDATE_CONTENT_DELETE');

-- ─────────────────────────────────────────────────────────
-- 회원가입 (users INSERT)
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'INSERT_USER',
       'INSERT INTO users (user_id, password, hashed_password, email, phone, nickname, role, username, created_at, updated_at, social_type)
        VALUES (:userId, :password, :hashedPassword, :email, :phone,
                COALESCE(:nickname, :userId), :role, :username, NOW(), NOW(), :socialType)',
       'COMMAND', '신규 회원 가입 (닉네임 없을 시 아이디로 대체)', 'N', 3600, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'INSERT_USER');

-- ─────────────────────────────────────────────────────────
-- 탈퇴 후 재가입 여부 확인
-- ─────────────────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'FIND_WITHDRAW_USER',
       'SELECT user_sqno AS "userSqno", user_id AS "userId", email
          FROM users
         WHERE email = :email
           AND del_yn = ''Y''
           AND withdraw_at > NOW() - INTERVAL ''7 days''',
       'SINGLE', '7일 이내 탈퇴한 회원 조회', 'N', 3600, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'FIND_WITHDRAW_USER');
