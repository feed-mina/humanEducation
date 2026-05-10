-- V23: 핵심 테이블 생성 + query_master 시드 + MAIN_PAGE 수정
-- 통합 범위: 구 V23(목표 쿼리) + V24(MAIN_PAGE 수정) + V25(핵심 쿼리) + V28(memberships) + V29(user_memberships)
-- 2026-03-13

-- ───────────────────────────────────────────────
-- 1. memberships 테이블 생성 + 시드
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memberships (
    id            BIGSERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL UNIQUE,
    can_learn     BOOLEAN NOT NULL DEFAULT FALSE,
    can_converse  BOOLEAN NOT NULL DEFAULT FALSE,
    can_analyze   BOOLEAN NOT NULL DEFAULT FALSE,
    duration_days INTEGER NOT NULL,
    price_cents   INTEGER NOT NULL,
    description   TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO memberships (name, can_learn, can_converse, can_analyze, duration_days, price_cents, description)
VALUES
  ('베이직',   TRUE,  FALSE, FALSE, 30, 129000, 'AI 학습 기능만 이용 가능한 기본 멤버십'),
  ('프리미엄', TRUE,  TRUE,  TRUE,  30, 219000, 'AI 학습 + 음성 대화 + 분석 모두 이용 가능')
ON CONFLICT (name) DO NOTHING;

-- ───────────────────────────────────────────────
-- 2. user_memberships 테이블 생성
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_memberships (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    membership_id BIGINT NOT NULL REFERENCES memberships(id),
    started_at    TIMESTAMP NOT NULL,
    expires_at    TIMESTAMP NOT NULL,
    status        VARCHAR(20) NOT NULL DEFAULT 'active',
    granted_by    VARCHAR(20) NOT NULL DEFAULT 'purchase',
    created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_memberships_user_id
    ON user_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memberships_user_status
    ON user_memberships(user_id, status);
CREATE INDEX IF NOT EXISTS idx_user_memberships_expires_at
    ON user_memberships(expires_at);

-- ───────────────────────────────────────────────
-- 3. query_master: 목표시간 관련 SQL 키
-- ───────────────────────────────────────────────
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_USER_GOAL_TIME',
       'SELECT target_time AS "targetTime"
          FROM goal_settings
         WHERE user_sqno = :userSqno
           AND status IS NULL
           AND target_time >= CURRENT_DATE
         ORDER BY target_time ASC
         LIMIT 1',
       'SINGLE', '메인 화면용: 가장 가까운 미래 목표 1건 조회', 'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_USER_GOAL_TIME');

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_USER_GOAL_LIST',
       'SELECT target_time AS "targetTime"
          FROM goal_settings
         WHERE user_sqno = :userSqno
           AND status IS NULL
           AND target_time >= CURRENT_DATE
         ORDER BY target_time ASC
         LIMIT 3 OFFSET 1',
       'MULTI', '팝업 리스트용: 메인 목표 이후의 다음 3건 조회', 'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_USER_GOAL_LIST');

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'UPDATE_GOAL_RESULT',
       'UPDATE goal_settings
           SET status = :status, recorded_time = :recordedTime
         WHERE id = (
             SELECT id
               FROM goal_settings
              WHERE user_sqno = :userSqno
                AND status IS NULL
                AND target_time >= CURRENT_DATE
              ORDER BY target_time ASC
              LIMIT 1
         )',
       'COMMAND', '가장 가까운 미완료 목표의 도착 결과(성공/실패) 업데이트', 'N', 0, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'UPDATE_GOAL_RESULT');

-- ───────────────────────────────────────────────
-- 4. query_master: 핵심 SQL 키 시드
-- ───────────────────────────────────────────────
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

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'COUNT_CONTENT_LIST',
       'SELECT COUNT(*) AS total_count
          FROM content d
         WHERE d.del_yn = ''N''
           AND (:userId IS NULL OR :userId = '''' OR d.user_id = :userId)',
       'SINGLE', '전체 콘텐츠 개수 조회', 'N', 0, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'COUNT_CONTENT_LIST');

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

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'INSERT_USER',
       'INSERT INTO users (user_id, password, hashed_password, email, phone, nickname, role, username, created_at, updated_at, social_type)
        VALUES (:userId, :password, :hashedPassword, :email, :phone,
                COALESCE(:nickname, :userId), :role, :username, NOW(), NOW(), :socialType)',
       'COMMAND', '신규 회원 가입 (닉네임 없을 시 아이디로 대체)', 'N', 3600, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'INSERT_USER');

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'FIND_WITHDRAW_USER',
       'SELECT user_sqno AS "userSqno", user_id AS "userId", email
          FROM users
         WHERE email = :email
           AND del_yn = ''Y''
           AND withdraw_at > NOW() - INTERVAL ''7 days''',
       'SINGLE', '7일 이내 탈퇴한 회원 조회', 'N', 3600, NULL
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'FIND_WITHDRAW_USER');

-- ───────────────────────────────────────────────
-- 5. MAIN_PAGE 중복 콘텐츠 카드 정리
-- ───────────────────────────────────────────────
UPDATE ui_metadata
   SET parent_group_id = 'main_bento_content_grp'
 WHERE screen_id = 'MAIN_PAGE'
   AND component_id = 'main_bento_content_body';

UPDATE ui_metadata
   SET parent_group_id = 'main_bento_content_body'
 WHERE screen_id = 'MAIN_PAGE'
   AND component_id IN ('main_bento_content_icon', 'main_bento_content_title', 'main_bento_content_desc');

UPDATE ui_metadata
   SET parent_group_id = 'main_bento_content_grp'
 WHERE screen_id = 'MAIN_PAGE'
   AND component_id = 'main_bento_content_btn';

DELETE FROM ui_metadata
 WHERE screen_id = 'MAIN_PAGE'
   AND component_id IN (
       'main_bento_diary_grp', 'main_bento_diary_body', 'main_bento_diary_icon',
       'main_bento_diary_title', 'main_bento_diary_desc', 'main_bento_diary_btn'
   );

DO $$ BEGIN RAISE NOTICE 'V23 완료 - memberships/user_memberships 테이블, 핵심 query_master, MAIN_PAGE 수정'; END $$;
