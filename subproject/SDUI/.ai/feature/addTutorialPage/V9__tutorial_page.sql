-- V9: 튜토리얼 페이지 (Playground & Split View) 추가
-- 1번(Split View) + 2번(Playground) 기능을 통합한 TUTORIAL_PLAYGROUND 컴포넌트 배치
BEGIN;
-- 튜토리얼 페이지 정의
-- TUTORIAL_PLAYGROUND 컴포넌트가 화면 전체를 담당하며 내부에서 좌우 분할 렌더링 수행
INSERT INTO ui_metadata (
        screen_id,
        component_id,
        component_type,
        label_text,
        sort_order,
        css_class,
        group_direction
    )
VALUES (
        'TUTORIAL_PAGE',
        'TUTORIAL_MAIN_WIDGET',
        'TUTORIAL_PLAYGROUND',
        'SDUI 튜토리얼',
        1,
        'w-full h-full',
        'COLUMN'
    );
COMMIT;