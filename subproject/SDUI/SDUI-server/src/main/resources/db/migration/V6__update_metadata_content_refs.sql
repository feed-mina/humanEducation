-- V6: ui_metadata 및 query_master에서 diary → content 변경
-- 모든 메타데이터에서 diary 참조를 content로 변경합니다.

-- ui_metadata: screen_id 변경
UPDATE ui_metadata SET screen_id = 'CONTENT_LIST' WHERE screen_id = 'DIARY_LIST';
UPDATE ui_metadata SET screen_id = 'CONTENT_WRITE' WHERE screen_id = 'DIARY_WRITE';
UPDATE ui_metadata SET screen_id = 'CONTENT_DETAIL' WHERE screen_id = 'DIARY_DETAIL';

-- ui_metadata: component_id 변경 (DIARY_ → CONTENT_)
UPDATE ui_metadata SET component_id = REPLACE(component_id, 'DIARY_', 'CONTENT_') WHERE component_id LIKE 'DIARY_%';

-- ui_metadata: action_type 변경 (DIARY_ → CONTENT_)
UPDATE ui_metadata SET action_type = REPLACE(action_type, 'DIARY_', 'CONTENT_') WHERE action_type LIKE 'DIARY_%';

-- ui_metadata: action_url 변경 (/api/diary/ → /api/content/)
UPDATE ui_metadata SET action_url = REPLACE(action_url, '/api/diary/', '/api/content/') WHERE action_url LIKE '%/api/diary/%';

-- ui_metadata: data_api_url 변경 (/api/diary/ → /api/content/)
UPDATE ui_metadata SET data_api_url = REPLACE(data_api_url, '/api/diary/', '/api/content/') WHERE data_api_url LIKE '%/api/diary/%';

-- ui_metadata: data_sql_key 변경 (GET_DIARY_ → GET_CONTENT_)
UPDATE ui_metadata SET data_sql_key = REPLACE(data_sql_key, 'GET_DIARY_', 'GET_CONTENT_') WHERE data_sql_key LIKE 'GET_DIARY_%';

-- ui_metadata: ref_data_id 변경 (diary → content)
UPDATE ui_metadata SET ref_data_id = REPLACE(ref_data_id, 'diary', 'content') WHERE ref_data_id LIKE '%diary%';

-- query_master: sql_key 변경 (GET_DIARY_ → GET_CONTENT_)
UPDATE query_master SET sql_key = REPLACE(sql_key, 'GET_DIARY_', 'GET_CONTENT_') WHERE sql_key LIKE 'GET_DIARY_%';

-- query_master: query_text 변경 (diary → content, diary_id → content_id)
UPDATE query_master SET query_text = REPLACE(REPLACE(query_text, 'diary', 'content'), 'diary_id', 'content_id') WHERE query_text LIKE '%diary%';

-- query_master: description 변경 (일기 → 콘텐츠)
UPDATE query_master SET description = REPLACE(description, '일기', '콘텐츠') WHERE description LIKE '%일기%';

-- 변경 사항 검증
DO $$
DECLARE
    ui_updated INTEGER;
    query_updated INTEGER;
BEGIN
    SELECT COUNT(*) INTO ui_updated FROM ui_metadata WHERE screen_id LIKE 'CONTENT%';
    SELECT COUNT(*) INTO query_updated FROM query_master WHERE sql_key LIKE 'GET_CONTENT%';

    RAISE NOTICE 'V6 완료 - ui_metadata CONTENT 스크린: %, query_master CONTENT 쿼리: %', ui_updated, query_updated;
END $$;
