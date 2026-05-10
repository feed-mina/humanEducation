-- ============================================================
-- 로컬 테스트용 샘플 데이터 (AWS에는 실행 금지)
-- 비밀번호: Test1234!  (SHA256 해시)
-- 중복 실행 안전: NOT EXISTS 체크 사용
-- ============================================================

-- 관리자 계정
INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'admin_test', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_ADMIN', 'admin@test.com', '010-0000-0001', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'admin@test.com');

-- 일반사용자 6명
INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'user_test1', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_USER', 'user1@test.com', '010-0000-0002', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'user1@test.com');

INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'user_test2', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_USER', 'user2@test.com', '010-0000-0003', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'user2@test.com');

INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'user_test3', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_USER', 'user3@test.com', '010-0000-0004', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'user3@test.com');

INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'user_test4', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_USER', 'user4@test.com', '010-0000-0005', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'user4@test.com');

INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'user_test5', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_USER', 'user5@test.com', '010-0000-0006', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'user5@test.com');

INSERT INTO users (user_id, password, hashed_password, role, email, phone, del_yn, verify_yn, created_at, updated_at)
SELECT 'user_test6', 'Test1234!', '0fadf52a4580cfebb99e61162139af3d3a6403c1d36b83e4962b721d1c8cbd0b',
       'ROLE_USER', 'user6@test.com', '010-0000-0007', 'N', 'Y', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'user6@test.com');

-- 삽입 확인
SELECT user_sqno, user_id, email, role, verify_yn FROM users ORDER BY user_sqno;



SELECT json_build_object(
    'ui_metadata', (SELECT count(*) FROM ui_metadata),
    'query_master', (SELECT count(*) FROM query_master),
    'users', (SELECT count(*) FROM users),
    'content', (SELECT count(*) FROM content),
	'system_logs', (SELECT count(*) FROM system_logs),
	'user_memberships', (SELECT count(*) FROM user_memberships),
    'memberships', (SELECT count(*) FROM memberships),
	'interview_resume', (SELECT count(*) FROM interview_resume)
) AS table_counts;

-- {"ui_metadata" : 187, "query_master" : 18, "users" : 0, "content" : 0, "system_logs" : 5, "user_memberships" : 0, "memberships" : 2, "interview_resume" : 0}
