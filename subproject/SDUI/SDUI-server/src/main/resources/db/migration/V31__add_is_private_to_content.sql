-- V31: content 테이블 "나만 보기" (is_private) 기능 추가
-- 체크 시 작성자와 어드민(ROLE_ADMIN)만 콘텐츠를 볼 수 있음
-- 본인의 내 콘텐츠 목록(GET_MEMBER_CONTENT_LIST)에서는 비공개 글도 표시됨

-- (1) content 테이블에 is_private 컬럼 추가
ALTER TABLE content ADD COLUMN IF NOT EXISTS is_private BOOLEAN NOT NULL DEFAULT FALSE;

-- (2) GET_CONTENT_LIST_PAGE: 공개 목록에서 비공개 콘텐츠 완전 제외 (타인에게 노출 안 됨)
UPDATE query_master
SET query_text =
  'SELECT d.content_id AS "content_id", d.user_sqno AS "user_sqno", d.title AS "title",
          d.date AS "date", d.emotion AS "emotion",
          d.day_tag1 AS "tag1", d.day_tag2 AS "tag2", d.day_tag3 AS "tag3",
          d.content_status AS "content_status", d.reg_dt AS "reg_dt",
          CASE WHEN u.del_yn = ''Y'' THEN CONCAT(''del_'', d.user_id) ELSE d.user_id END AS "user_id"
     FROM content d
     LEFT JOIN users u ON d.user_sqno = u.user_sqno
    WHERE d.del_yn = ''N''
      AND d.is_private = FALSE
      AND (NULLIF(CAST(:filterId AS VARCHAR), '''') IS NULL OR d.user_id = CAST(:filterId AS VARCHAR))
    ORDER BY d.reg_dt DESC
    LIMIT  CAST(COALESCE(NULLIF(CAST(:pageSize AS VARCHAR), ''''), ''5'') AS INTEGER)
    OFFSET CAST(COALESCE(NULLIF(CAST(:offset  AS VARCHAR), ''''), ''0'') AS INTEGER)'
WHERE sql_key = 'GET_CONTENT_LIST_PAGE';

-- (3) COUNT_CONTENT_LIST: 공개 콘텐츠 개수만 카운트
UPDATE query_master
SET query_text =
  'SELECT COUNT(*) AS total_count
     FROM content d
    WHERE d.del_yn = ''N''
      AND d.is_private = FALSE
      AND (:userId IS NULL OR :userId = '''' OR d.user_id = :userId)'
WHERE sql_key = 'COUNT_CONTENT_LIST';

-- (4) CONTENT_WRITE 화면에 "나만 보기" 체크박스 UI 메타데이터 추가 (save_btn 직전, sort_order=65)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'is_private', '나만 보기', 'CHECKBOX', 65,
  false, false, NULL, NULL,
  'private-toggle', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true', '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'is_private'
);
