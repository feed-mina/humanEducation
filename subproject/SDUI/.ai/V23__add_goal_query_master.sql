-- V23: query_master에 목표시간 관련 SQL 키 추가
-- GoalTimeQueryService가 사용하는 3개 키: GET_USER_GOAL_TIME, GET_USER_GOAL_LIST, UPDATE_GOAL_RESULT
-- 해당 키들은 AWS DB에만 존재하고 로컬 마이그레이션에 누락된 상태였음

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_USER_GOAL_TIME',
       'SELECT target_time AS "targetTime"
          FROM goal_settings
         WHERE user_sqno = :userSqno
           AND status IS NULL
           AND target_time >= CURRENT_DATE
         ORDER BY target_time ASC
         LIMIT 1',
       'SINGLE',
       '메인 화면용: 가장 가까운 미래 목표 1건 조회',
       'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_USER_GOAL_TIME');

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'GET_USER_GOAL_LIST',
       'SELECT target_time AS "targetTime"
          FROM goal_settings
         WHERE user_sqno = :userSqno
           AND status IS NULL
           AND target_time >= CURRENT_DATE
         ORDER BY target_time ASC
         LIMIT 3 OFFSET 1',
       'MULTI',
       '팝업 리스트용: 메인 목표 이후의 다음 3건 조회',
       'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'GET_USER_GOAL_LIST');

INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'UPDATE_GOAL_RESULT',
       'UPDATE goal_settings
           SET status = :status, recorded_time = :recordedTime
         WHERE id = (
             SELECT id
               FROM goal_settings
              WHERE user_sqno = :userSqno
                AND status IS NULL
                AND target_time >= CURRENT_DATE
              ORDER BY target_time ASC
              LIMIT 1
         )',
       'COMMAND',
       '가장 가까운 미완료 목표의 도착 결과(성공/실패) 업데이트',
       'N', 0, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'UPDATE_GOAL_RESULT');
