-- V24: MAIN_PAGE 중복 콘텐츠 카드 정리
-- 원인: main_bento_content_* (ui_id 3~8)의 parent_group_id가 main_bento_diary_*를 참조하고 있고,
--       ui_id 143~148 (main_bento_diary_*) 행이 추가되면서 같은 카드가 2번 렌더링됨
-- 수정: content 행들의 parent 참조를 content_* ID로 교정 후 diary 행 삭제

-- 1. content 자식들의 parent 참조 수정
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

-- 2. 중복된 main_bento_diary_* 행 삭제
DELETE FROM ui_metadata
 WHERE screen_id = 'MAIN_PAGE'
   AND component_id IN (
       'main_bento_diary_grp',
       'main_bento_diary_body',
       'main_bento_diary_icon',
       'main_bento_diary_title',
       'main_bento_diary_desc',
       'main_bento_diary_btn'
   );
