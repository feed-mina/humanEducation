-- V35: 팬 게시판(FanBoard) 시스템 구축 및 고도화 (통합 마이그레이션)
-- 2026-03-21

-- (0) missing columns in users & content table (fix Flyway error)
ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(255);
ALTER TABLE content ADD COLUMN IF NOT EXISTS img_url TEXT;

-- (0.1) resync user_sqno sequence (fix duplicate key violation)
SELECT setval('users_user_sqno_seq', COALESCE((SELECT MAX(user_sqno) FROM users), 1), true);

-- (1) 익명 작성을 위한 GUEST 시스템 계정 추가
INSERT INTO users (user_id, password, hashed_password, email, nickname, role, username, created_at, updated_at, social_type)
SELECT 'GUEST', 'GUEST_PW_DUMMY', 'GUEST_HASH_DUMMY', 'guest@sdui.com', '방탄팬(ARMY)', 'ROLE_USER', 'GUEST_USER', NOW(), NOW(), 'system'
WHERE NOT EXISTS (SELECT 1 FROM users WHERE user_id = 'GUEST');

-- (2) 팬 게시글 전용 테이블 fan_board 생성 (기존 content 테이블 구조 복사)
-- INCLUDING DEFAULTS만 사용: content_pkey / idx_content_* 인덱스 이름 충돌 방지
CREATE TABLE IF NOT EXISTS fan_board (LIKE content INCLUDING DEFAULTS);
-- 별도 primary key 지정 (content_id를 PK로)
ALTER TABLE fan_board ADD PRIMARY KEY (content_id);
-- fan_board 전용 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_fan_board_user_sqno ON fan_board(user_sqno);
CREATE INDEX IF NOT EXISTS idx_fan_board_day_tag1 ON fan_board(day_tag1);
CREATE INDEX IF NOT EXISTS idx_fan_board_reg_dt ON fan_board(reg_dt DESC);

-- (3) 기존 content 테이블에서 팬 게시판 관련 데이터 이전
INSERT INTO fan_board
SELECT * FROM content
 WHERE (day_tag1 IN ('REPORT', 'LOST', 'CHEER', 'FEED') OR user_id = 'GUEST')
   AND del_yn = 'N'
   AND NOT EXISTS (SELECT 1 FROM fan_board fb WHERE fb.content_id = content.content_id);

-- (4) query_master 신규 추가 (기존 SDUI 쿼리 수정 안 함)
-- NOTE: GET_CONTENT_LIST_PAGE는 V31에서 올바르게 정의됨 (filterId=userId 필터).
--       FanBoard는 GET_FANBOARD_LIST를 사용하므로 SDUI 쿼리 변경 불필요.

-- 4-1. 팬 게시판 전용 리스트 조회 (GET_FANBOARD_LIST)
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_FANBOARD_LIST',
       'SELECT d.content_id, d.user_id, d.title, d.content, d.date, d.emotion, 
               d.day_tag1, d.day_tag2, d.day_tag3, d.content_status, d.selected_times, d.reg_dt
          FROM fan_board d
         WHERE d.del_yn = ''N''
           AND (:filterId IS NULL OR :filterId = '''' OR d.day_tag1 = :filterId)
         ORDER BY d.reg_dt DESC
         LIMIT  CAST(COALESCE(NULLIF(CAST(:pageSize AS VARCHAR), ''''), ''50'') AS INTEGER)
         OFFSET CAST(COALESCE(NULLIF(CAST(:offset  AS VARCHAR), ''''), ''0'')  AS INTEGER)',
       'LIST', '팬 게시판 전용 리스트 조회 (fan_board 테이블)', 'N', 0, NULL
 WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_FANBOARD_LIST');

-- 4-3. 팬 게시판 전용 상세 조회 (GET_FANBOARD_DETAIL)
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_FANBOARD_DETAIL',
       'SELECT d.content_id, d.user_id, d.title, d.content, d.date, d.emotion, 
               d.day_tag1, d.day_tag2, d.day_tag3, d.content_status, d.selected_times, d.reg_dt
          FROM fan_board d
         WHERE d.content_id = :contentId',
       'SINGLE', '팬 게시판 전용 상세 조회 (fan_board 테이블)', 'N', 0, NULL
 WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_FANBOARD_DETAIL');

-- 4-4. 익명 글쓰기 전용 쿼리 (INSERT_FANBOARD)
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'INSERT_FANBOARD',
       'INSERT INTO fan_board (
            user_sqno, user_id, title, content, emotion,
            selected_times, daily_slots, day_tag1, day_tag2, day_tag3,
            reg_dt
        ) 
        SELECT 
            user_sqno, user_id, :title, :content, CAST(COALESCE(CAST(:emotion AS VARCHAR), ''0'') AS INTEGER),
            CAST(:selected_times AS jsonb), CAST(:daily_slots AS jsonb),
            :day_tag1, :day_tag2, :day_tag3,
            NOW()
          FROM users 
         WHERE user_id = ''GUEST''
         LIMIT 1',
       'COMMAND', '팬 게시판용 익명 게시글 작성 (로그인 불필요)', 'N', 0, NULL
 WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'INSERT_FANBOARD');

-- 4-5. 팬 게시판 전용 수정 쿼리 (UPDATE_FANBOARD) - fan_board 테이블 대상
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'UPDATE_FANBOARD',
       'UPDATE fan_board
           SET title = :title,
               content = :content,
               emotion = CAST(COALESCE(CAST(:emotion AS VARCHAR), ''0'') AS INTEGER),
               selected_times = CAST(:selected_times AS jsonb),
               day_tag1 = :day_tag1,
               day_tag2 = :day_tag2,
               day_tag3 = :day_tag3
         WHERE content_id = CAST(:content_id AS BIGINT)',
       'COMMAND', '팬 게시판 게시글 수정 (fan_board 테이블)', 'N', 0, NULL
 WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'UPDATE_FANBOARD');

DO $$ BEGIN RAISE NOTICE 'V35 완료 - 팬 게시판 시스템 통합 구축 성공'; END $$;
