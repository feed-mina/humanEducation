-- V9: MAIN_PAGE USER 카드 라벨 변경 (2026-03-07)
-- "콘텐츠 쓰러가기" → "콘텐츠 작성하기"
-- "콘텐츠 보기"    → "콘텐츠 리스트 확인하기"

UPDATE ui_metadata SET label_text = '콘텐츠 작성하기'     WHERE component_id = 'main_bento_diary_title';
UPDATE ui_metadata SET label_text = '콘텐츠 리스트 확인하기' WHERE component_id = 'main_bento_view_title';
UPDATE ui_metadata SET label_text = '📖 콘텐츠 목록'      WHERE component_id = 'main_bento_view_btn';

DO $$ BEGIN RAISE NOTICE 'V9 완료 - MAIN_PAGE USER 카드 라벨 변경'; END $$;
