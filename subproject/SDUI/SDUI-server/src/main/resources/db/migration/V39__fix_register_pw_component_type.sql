-- REGISTER_PAGE의 reg_pw 컴포넌트 타입을 INPUT → PASSWORD로 변경
-- PasswordField는 토글 버튼(👀/🕶️)이 내장되어 있어 로그인 페이지와 동일한 UX 제공
UPDATE ui_metadata
SET component_type = 'PASSWORD'
WHERE screen_id = 'REGISTER_PAGE'
  AND component_id = 'reg_pw';
