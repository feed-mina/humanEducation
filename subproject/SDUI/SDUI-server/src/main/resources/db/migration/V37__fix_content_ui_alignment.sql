-- V32: CONTENT_DETAIL, CONTENT_MODIFY 나만 보기 기능 추가 및 버튼 중앙 정렬, 상하 텍스트 정렬

-- 1. GET_CONTENT_LIST_PAGE 업데이트 (나만 보기 글일 경우 하트 아이콘 추가, 작성자는 비공개 글 볼 수 있도록 허용)
UPDATE query_master
SET query_text =
  'SELECT d.content_id AS "content_id", d.user_sqno AS "user_sqno", 
          CASE WHEN d.is_private = TRUE THEN CONCAT(d.title, '' ❤️'') ELSE d.title END AS "title",
          d.date AS "date", d.emotion AS "emotion",
          d.day_tag1 AS "tag1", d.day_tag2 AS "tag2", d.day_tag3 AS "tag3",
          d.content_status AS "content_status", d.reg_dt AS "reg_dt",
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

-- 2. GET_MEMBER_CONTENT_LIST 업데이트 (본인 콘텐츠 목록 나만 보기 하트 추가)
UPDATE query_master
SET query_text =
  'SELECT content_id AS "content_id", 
          CASE WHEN is_private = TRUE THEN CONCAT(title, '' ❤️'') ELSE title END AS "title", 
          content AS "content",
          reg_dt AS "reg_dt", img_url AS "img_url", user_id AS "user_id"
     FROM content
    WHERE user_sqno = CAST(:userSqno AS BIGINT)
      AND del_yn = ''N''
    ORDER BY reg_dt DESC
    LIMIT CAST(COALESCE(NULLIF(CAST(:pageSize AS VARCHAR), ''''), ''5'') AS INTEGER) OFFSET CAST(COALESCE(NULLIF(CAST(:offset AS VARCHAR), ''''), ''0'') AS INTEGER)'
WHERE sql_key = 'GET_MEMBER_CONTENT_LIST';

-- 3. CONTENT_DETAIL "나만 보기" 메타데이터 (읽기 전용)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'is_private', '나만 보기', 'CHECKBOX', 65,
  false, true, NULL, NULL,
  'private-toggle', NULL, NULL, NULL,
  NULL, NULL, NULL, 'is_private',
  NULL, NULL, NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true', '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'is_private'
);

-- 4. CONTENT_MODIFY "나만 보기" 메타데이터 (비읽기 방지)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'is_private', '나만 보기', 'CHECKBOX', 65,
  false, false, NULL, NULL,
  'private-toggle', NULL, NULL, NULL,
  NULL, NULL, NULL, 'is_private',
  NULL, NULL, NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true', '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'is_private'
);

-- 5. CONTENT_WRITE 텍스트 태그 정렬 수정
DELETE FROM ui_metadata WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'GROUP_TAGS_ONELINE';

UPDATE ui_metadata SET group_direction = 'COLUMN' WHERE screen_id = 'CONTENT_WRITE' AND component_id IN ('GROUP_TAG_ROW1', 'GROUP_TAG_ROW2', 'GROUP_TAG_ROW3');

UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW1' WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'day_tag1';
UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW2' WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'day_tag2';
UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW3' WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'day_tag3';

-- 6. CONTENT_DETAIL 텍스트 태그 그룹 및 텍스트 추가, 상하 정렬
INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, group_direction, parent_group_id, is_visible)
SELECT 'CONTENT_DETAIL', 'GROUP_TAG_ROW1', '태그1행', 'GROUP', 70, 'GROUP_TAG_ROW1', 'COLUMN', 'DAYTAG_SUB_GROUP', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'GROUP_TAG_ROW1');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, group_direction, parent_group_id, is_visible)
SELECT 'CONTENT_DETAIL', 'GROUP_TAG_ROW2', '태그2행', 'GROUP', 80, 'GROUP_TAG_ROW2', 'COLUMN', 'DAYTAG_SUB_GROUP', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'GROUP_TAG_ROW2');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, group_direction, parent_group_id, is_visible)
SELECT 'CONTENT_DETAIL', 'GROUP_TAG_ROW3', '태그3행', 'GROUP', 90, 'GROUP_TAG_ROW3', 'COLUMN', 'DAYTAG_SUB_GROUP', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'GROUP_TAG_ROW3');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, parent_group_id, is_visible)
SELECT 'CONTENT_DETAIL', 'title_dayTag1', '하루태그1', 'TEXT', 71, 'content-label', 'GROUP_TAG_ROW1', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'title_dayTag1');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, parent_group_id, is_visible)
SELECT 'CONTENT_DETAIL', 'title_dayTag2', '하루태그2', 'TEXT', 81, 'content-label', 'GROUP_TAG_ROW2', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'title_dayTag2');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, parent_group_id, is_visible)
SELECT 'CONTENT_DETAIL', 'title_dayTag3', '하루태그3', 'TEXT', 91, 'content-label', 'GROUP_TAG_ROW3', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'title_dayTag3');

UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW1' WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'dayTag1';
UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW2' WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'dayTag2';
UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW3' WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'dayTag3';

UPDATE ui_metadata SET group_direction = 'ROW' WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'DAYTAG_SUB_GROUP';

-- 7. CONTENT_MODIFY 텍스트 태그 그룹 및 텍스트 추가, 상하 정렬
INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, group_direction, parent_group_id, is_visible)
SELECT 'CONTENT_MODIFY', 'GROUP_TAG_ROW1', '태그1행', 'GROUP', 70, 'GROUP_TAG_ROW1', 'COLUMN', 'DAYTAG_SUB_GROUP', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'GROUP_TAG_ROW1');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, group_direction, parent_group_id, is_visible)
SELECT 'CONTENT_MODIFY', 'GROUP_TAG_ROW2', '태그2행', 'GROUP', 80, 'GROUP_TAG_ROW2', 'COLUMN', 'DAYTAG_SUB_GROUP', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'GROUP_TAG_ROW2');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, group_direction, parent_group_id, is_visible)
SELECT 'CONTENT_MODIFY', 'GROUP_TAG_ROW3', '태그3행', 'GROUP', 90, 'GROUP_TAG_ROW3', 'COLUMN', 'DAYTAG_SUB_GROUP', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'GROUP_TAG_ROW3');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, parent_group_id, is_visible)
SELECT 'CONTENT_MODIFY', 'title_dayTag1', '하루태그1', 'TEXT', 71, 'content-label', 'GROUP_TAG_ROW1', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'title_dayTag1');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, parent_group_id, is_visible)
SELECT 'CONTENT_MODIFY', 'title_dayTag2', '하루태그2', 'TEXT', 81, 'content-label', 'GROUP_TAG_ROW2', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'title_dayTag2');

INSERT INTO ui_metadata (screen_id, component_id, label_text, component_type, sort_order, css_class, parent_group_id, is_visible)
SELECT 'CONTENT_MODIFY', 'title_dayTag3', '하루태그3', 'TEXT', 91, 'content-label', 'GROUP_TAG_ROW3', 'true'
WHERE NOT EXISTS (SELECT 1 FROM ui_metadata WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'title_dayTag3');

UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW1' WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'day_tag1';
UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW2' WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'day_tag2';
UPDATE ui_metadata SET parent_group_id = 'GROUP_TAG_ROW3' WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'day_tag3';

UPDATE ui_metadata SET group_direction = 'ROW' WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'DAYTAG_SUB_GROUP';

-- 8. 버튼 그룹 중앙 정렬을 위한 align-center 클래스 추가
UPDATE ui_metadata SET css_class = COALESCE(css_class, '') || ' align-center' 
WHERE screen_id IN ('CONTENT_WRITE', 'CONTENT_DETAIL', 'CONTENT_MODIFY') 
  AND component_id IN ('save_btn', 'go_modify_btn', 'go_list_btn') 
  AND css_class NOT LIKE '%align-center%';
