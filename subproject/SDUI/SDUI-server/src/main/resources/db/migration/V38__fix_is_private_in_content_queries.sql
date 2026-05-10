-- V38: 모든 content 관련 쿼리에 is_private 필드 추가
-- 적용 대상: INSERT_CONTENT(WRITE), GET_CONTENT_DETAIL(DETAIL/MODIFY 조회),
--            UPDATE_CONTENT_DETAIL(MODIFY 저장), GET_CONTENT_LIST_PAGE(목록 조회)

-- 1. INSERT_CONTENT: 신규 작성 시 is_private 저장 (CONTENT_WRITE)
UPDATE query_master
SET query_text =
  'INSERT INTO content (
       user_sqno, user_id, title, content, emotion,
       selected_times, daily_slots, day_tag1, day_tag2, day_tag3,
       is_private, reg_dt
   ) VALUES (
       :userSqno, :userId, :title, :content, CAST(:emotion AS INTEGER),
       CAST(:selected_times AS jsonb), CAST(:daily_slots AS jsonb),
       :day_tag1, :day_tag2, :day_tag3,
       CAST(:is_private AS BOOLEAN), NOW()
   )'
WHERE sql_key = 'INSERT_CONTENT';

-- 2. GET_CONTENT_DETAIL: 상세 조회 시 is_private 포함 (CONTENT_DETAIL, CONTENT_MODIFY 체크박스 활성화)
UPDATE query_master
SET query_text =
  'SELECT d.content_id, d.user_id, d.title, d.content, d.date, d.emotion,
          d.day_tag1, d.day_tag2, d.day_tag3, d.content_status, d.role_nm,
          d.selected_times, d.daily_slots, d.reg_dt, d.is_private
     FROM content d
    WHERE d.content_id = CAST(:contentId AS BIGINT)
      AND d.del_yn = ''N'''
WHERE sql_key = 'GET_CONTENT_DETAIL';

-- 3. UPDATE_CONTENT_DETAIL: 수정 저장 시 is_private 반영 (CONTENT_MODIFY)
UPDATE query_master
SET query_text =
  'UPDATE content
      SET title          = :title,
          content        = :content,
          emotion        = CAST(:emotion AS INTEGER),
          selected_times = CAST(:selected_times AS jsonb),
          daily_slots    = CAST(:daily_slots AS jsonb),
          day_tag1       = :day_tag1,
          day_tag2       = :day_tag2,
          day_tag3       = :day_tag3,
          is_private     = CAST(:is_private AS BOOLEAN)
    WHERE content_id     = CAST(:content_id AS BIGINT)
      AND user_sqno      = :userSqno'
WHERE sql_key = 'UPDATE_CONTENT_DETAIL';

-- 4. GET_CONTENT_LIST_PAGE: 목록 조회 시 is_private 필드 노출 (리스트 하트 아이콘 표시용)
UPDATE query_master
SET query_text =
  'SELECT d.content_id AS "content_id", d.user_sqno AS "user_sqno",
          CASE WHEN d.is_private = TRUE THEN CONCAT(d.title, '' ❤️'') ELSE d.title END AS "title",
          d.date AS "date", d.emotion AS "emotion",
          d.day_tag1 AS "tag1", d.day_tag2 AS "tag2", d.day_tag3 AS "tag3",
          d.content_status AS "content_status", d.reg_dt AS "reg_dt",
          d.is_private AS "is_private",
          CASE WHEN u.del_yn = ''Y'' THEN CONCAT(''del_'', d.user_id) ELSE d.user_id END AS "user_id"
     FROM content d
     LEFT JOIN users u ON d.user_sqno = u.user_sqno
    WHERE d.del_yn = ''N''
      AND (d.is_private = FALSE OR d.user_sqno = CAST(:userSqno AS BIGINT))
      AND (NULLIF(CAST(:filterId AS VARCHAR), '''') IS NULL OR d.user_id = CAST(:filterId AS VARCHAR))
    ORDER BY d.reg_dt DESC
    LIMIT  CAST(COALESCE(NULLIF(CAST(:pageSize AS VARCHAR), ''''), ''5'') AS INTEGER)
    OFFSET CAST(COALESCE(NULLIF(CAST(:offset  AS VARCHAR), ''''), ''0'') AS INTEGER)'
WHERE sql_key = 'GET_CONTENT_LIST_PAGE';

