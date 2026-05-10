-- V20: ui_metadata diary → content 잔재 정리
-- V6가 놓친 패턴 처리:
--   - action_url /view/DIARY_* → /view/CONTENT_*
--   - component_id 소문자 diary* → content* (diaryTitle 등)
--   - css_class 소문자 diary* → content* (diaryInput, diary-label 등)
--   - data_api_url GET_DIARY_* → GET_CONTENT_*
--   - group_id / parent_group_id DIARYWRITE_* → CONTENTWRITE_*

-- 1. action_url: /view/DIARY_* → /view/CONTENT_*
UPDATE ui_metadata
SET action_url = REPLACE(action_url, '/view/DIARY_', '/view/CONTENT_')
WHERE action_url LIKE '%/view/DIARY_%';

-- 2. data_api_url: /api/execute/GET_DIARY_* → /api/execute/GET_CONTENT_*
UPDATE ui_metadata
SET data_api_url = REPLACE(data_api_url, '/api/execute/GET_DIARY_', '/api/execute/GET_CONTENT_')
WHERE data_api_url LIKE '%/api/execute/GET_DIARY_%';

-- 3. component_id: diary* → content* (diaryTitle → contentTitle 등)
UPDATE ui_metadata
SET component_id = REPLACE(component_id, 'diary', 'content')
WHERE component_id LIKE '%diary%';

-- 4. css_class: 소문자 diary → content
--    diaryInput → contentInput
--    diaryTextarea → contentTextarea
--    diary-label → content-label
--    diaryDate → contentDate
--    bento-card-diary → bento-card-content (V8에서 삽입된 MAIN_PAGE 벤토카드)
UPDATE ui_metadata
SET css_class = REPLACE(
                  REPLACE(
                    REPLACE(
                      REPLACE(
                        REPLACE(css_class, 'diaryInput', 'contentInput'),
                      'diaryTextarea', 'contentTextarea'),
                    'diary-label', 'content-label'),
                  'diaryDate', 'contentDate'),
                'bento-card-diary', 'bento-card-content')
WHERE css_class SIMILAR TO '%(diaryInput|diaryTextarea|diary-label|diaryDate|bento-card-diary)%';

-- 5. group_id: DIARYWRITE_* → CONTENTWRITE_*
UPDATE ui_metadata
SET group_id = REPLACE(group_id, 'DIARYWRITE_', 'CONTENTWRITE_')
WHERE group_id LIKE 'DIARYWRITE_%';

-- 6. parent_group_id: DIARYWRITE_* → CONTENTWRITE_*
UPDATE ui_metadata
SET parent_group_id = REPLACE(parent_group_id, 'DIARYWRITE_', 'CONTENTWRITE_')
WHERE parent_group_id LIKE 'DIARYWRITE_%';

-- 검증 리포트
DO $$
DECLARE
    remaining INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining
    FROM ui_metadata
    WHERE component_id LIKE '%diary%'
       OR css_class LIKE '%diary%'
       OR (action_url LIKE '%/view/DIARY_%')
       OR (data_api_url LIKE '%GET_DIARY_%')
       OR group_id LIKE 'DIARYWRITE_%'
       OR parent_group_id LIKE 'DIARYWRITE_%';

    RAISE NOTICE 'V20 완료 - 남은 diary 잔재: %건 (0이면 정상)', remaining;
END $$;
