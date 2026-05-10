-- V32: SERIAL(int4) → BIGINT 타입 수정
-- V28/V29/V30에서 SERIAL로 생성된 id 컬럼이 JPA 엔티티의 Long(bigint)과 불일치
-- Hibernate schema-validation 실패로 서버 기동 불가 → 긴급 수정

ALTER TABLE leetcode_problems    ALTER COLUMN id TYPE BIGINT USING id::BIGINT;
ALTER TABLE interview_questions  ALTER COLUMN id TYPE BIGINT USING id::BIGINT;
ALTER TABLE interview_schedule   ALTER COLUMN id TYPE BIGINT USING id::BIGINT;
