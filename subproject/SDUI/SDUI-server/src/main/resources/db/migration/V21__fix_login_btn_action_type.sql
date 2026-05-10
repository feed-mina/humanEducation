-- V21: LOGIN_PAGE 로그인 버튼 action_type 수정 (SUBMIT → LOGIN_SUBMIT)
-- login_btn의 action_type이 'SUBMIT'으로 잘못 설정되어 businessActions의 필수값 검증이
-- 실행되어 "Email은(는) 필수입니다" 오류가 발생하던 문제 수정

UPDATE ui_metadata
SET action_type = 'LOGIN_SUBMIT'
WHERE screen_id = 'LOGIN_PAGE'
  AND component_id = 'login_btn'
  AND action_type = 'SUBMIT';

DO $$ BEGIN RAISE NOTICE 'V21: login_btn action_type SUBMIT → LOGIN_SUBMIT 수정 완료'; END $$;
