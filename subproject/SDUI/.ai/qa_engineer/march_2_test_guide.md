# 3가지 문제 수정 후 테스트 가이드 (2026-03-02)

## 사전 준비

### 1. SQL 스크립트 실행
`fix_sdui_issues.sql` 실행하여 문제 2, 3 해결
//[메모] 실행결과 다음과 같은 json이 나옴
{
"check_type": "REGISTER_PAGE",
"component_id": "reg_submit",
"value": "false"
},
{
"check_type": "REGISTER_PAGE",
"component_id": "reg_addr_btn",
"value": "false"
},
{
"check_type": "CONTENT_LIST",
"component_id": "content_total_count",
"value": "COUNT_CONTENT_LIST"
},
{
"check_type": "CONTENT_LIST",
"component_id": "content_list_source",
"value": "GET_CONTENT_LIST_PAGE"
},
{
"check_type": "query_master",
"component_id": "INSERT_CONTENT",
"value": "신규 콘텐츠 작성 (JSONB 타입 및 day_tag 반영)"
},
{
"check_type": "query_master",
"component_id": "GET_CONTENT_LIST_PAGE",
"value": "콘텐츠 목록 페이징 조회"
},
{
"check_type": "query_master",
"component_id": "GET_CONTENT_DETAIL",
"value": "콘텐츠 상세 정보 조회 (JSON 데이터 포함)"
},
{
"check_type": "query_master",
"component_id": "COUNT_CONTENT_LIST",
"value": "전체 콘텐츠 개수 조회"
},
{
"check_type": "query_master",
"component_id": "UPDATE_CONTENT_DELETE",
"value": "선택한 콘텐츠 일괄 삭제 처리 (본인 확인 포함)"
},
{
"check_type": "query_master",
"component_id": "GET_MEMBER_CONTENT_LIST",
"value": "로그인한 사용자의 콘텐츠 목록 조회"
},
{
"check_type": "query_master",
"component_id": "UPDATE_CONTENT_DETAIL",
"value": "콘텐츠 내용 수정 쿼리"
}
]


### 2. Redis 캐시 클리어
```bash
docker exec -it sdui-redis-1 redis-cli FLUSHDB
```
//[메모]   로컬에서는 postgreSql은 testdb/postgres@PostgreSQL 18 도커는 my-redis 에서 실행

### 3. 백엔드 재시작 
//[메모] 아래와 같은 결과가 나옴

* What went wrong:
  Execution failed for task ':clean'.
> java.io.IOException: Unable to delete directory 'C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build'
Failed to delete some children. This might happen because a process has files open or has its working directory set in the target directory.
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\content\controller
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\content\domain
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\content\dto
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\content\service
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\content
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\Location\controller
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\Location\dto
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\Location\service
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\Location
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\query\controller
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\query\domain
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\query\repository
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\query\service
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\query
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\time\controller
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend\domain\time\domain
- and more ...
New files were found. This might happen because a process is still writing to the target directory.
- C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\SDUI-server\build\classes\java\main\com\domain\demo_backend

* Try:
> Run with --stacktrace option to get the stack trace.
> Run with --info or --debug option to get more log output.
> Run with --scan to get full insights.
> Get more help at https://help.gradle.org.

BUILD FAILED in 1m 11s
1 actionable task: 1 executed
Configuration cache entry reused.


```bash
cd SDUI-server
./gradlew clean build
./gradlew bootRun
```

## 테스트 시나리오

### 테스트 1: 신규 카카오 사용자 플로우
//[메모] Cannot read properties of undefined (reading 'label')
components/fields/AddressSearchGroup.tsx (29:20) @ AddressSearchGroup


27 |         <div className="address-group-container space-y-2 mb-4">
28 |             {/* 1. 라벨 (DB에서 넘어온 props.label 사용) */}
> 29 |             {props.label && <label className="text-sm font-bold block mb-1">{props.label}</label>}
|                    ^
30 |
31 |             {/* 2. 우편번호 & 검색 버튼 영역 */}
32 |             <div className="flex gap-2 mb-2">
> 
> 
**준비:**
```sql //[메모] Query returned successfully in 272 msec.
-- 기존 테스트 계정 삭제 
DELETE FROM users WHERE email = 'test@kakao.com'; 
```

**실행:**
1. http://localhost:3000/view/LOGIN_PAGE 접속
2. "카카오 로그인" 버튼 클릭
3. 카카오 계정 인증

**검증:**
- ✅ `/view/ADDITIONAL_INFO_PAGE`로 자동 리다이렉트
- ✅ 브라우저 쿠키에 `role=ROLE_GUEST` 확인
- ✅ 전화번호, 주소 입력 필드 표시

**추가 정보 입력:**
1. 전화번호: 010-9999-8888
2. 우편번호: 06000 (주소 검색 버튼 사용)
3. 도로명주소: 서울시 강남구 테헤란로 123
4. 상세주소: 4층
5. "제출하기" 클릭

**검증:**
- ✅ 쿠키가 `role=ROLE_USER`로 업데이트
- ✅ `/view/MAIN_PAGE`로 이동
- ✅ DB에서 role 확인:
  ```sql
  SELECT email, role, phone, road_address FROM users WHERE email = 'test@kakao.com';
  -- 결과: role='ROLE_USER', 추가 정보 저장됨
  ```

### 테스트 2: 기존 사용자 로그인

**실행:**
1. 일반 로그인 (userId, password)
2. 로그인 성공

**검증:**
- ✅ 브라우저 쿠키에 role 값 확인 (F12 → Application → Cookies)
- ✅ `/view/MAIN_PAGE`로 정상 이동
- ✅ `/api/auth/me` 응답 확인:
  ```json
  {
    "isLoggedIn": true,
    "email": "user@example.com",
    "role": "ROLE_USER"  // 실제 role 반환
  }
  ```
// [메모]  테스트 2 성공 
{
"role": "ROLE_USER",
"userSqno": 1,
"isLoggedIn": true,
"socialType": "N",
"userId": "pagingmina",
"email": "paging@test.com"
}

### 테스트 3: 콘텐츠 리스트 표시
// [메모 ] 네트워크 탭에 http://localhost:3000/api/execute/GET_CONTENT_LIST_PAGE 에서는 응답값이
{
"sqlKey": "GET_CONTENT_LIST_PAGE",
"data": [
{
"content_id": 4,
"user_sqno": 1,
"title": "콘텐츠 1",
"date": null,
"emotion": 1,
"tag1": "태그1",
"tag2": null,
"tag3": null,
"content_status": null,
"reg_dt": "2026-03-02T05:41:35.705+00:00",
"user_id": "pagingmina"
},
{
"content_id": 3,
"user_sqno": 1,
"title": "콘텐츠 테스트 1",
"date": null,
"emotion": 8,
"tag1": "태그1",
"tag2": null,
"tag3": null,
"content_status": null,
"reg_dt": "2026-03-02T05:40:42.681+00:00",
"user_id": "pagingmina"
},
{
"content_id": 2,
"user_sqno": 1,
"title": "콘텐츠 테스트 1",
"date": null,
"emotion": 8,
"tag1": "태그1",
"tag2": null,
"tag3": null,
"content_status": null,
"reg_dt": "2026-03-02T05:40:05.477+00:00",
"user_id": "pagingmina"
}
],
"status": "success"
}
이렇게 보이는데 화면에는 안뜸


**실행:**
1. 로그인 후 `/view/CONTENT_LIST` 접속

**검증:**
- ✅ 3개의 콘텐츠 아이템 표시
- ✅ 각 아이템에 제목, 내용 미리보기, 날짜 표시
- ✅ "전체 개수: 3" 표시
- ✅ 브라우저 콘솔에 에러 없음

**데이터베이스 검증:**
```sql //[메모] 아래와 같은 결과
[{"component_id":"content_total_count","data_sql_key":"COUNT_CONTENT_LIST"}, {"component_id":"content_list_source","data_sql_key":"GET_CONTENT_LIST_PAGE"}]

-- ui_metadata 확인
SELECT component_id, data_sql_key
FROM ui_metadata
WHERE screen_id = 'CONTENT_LIST' AND component_type = 'DATA_SOURCE';

-- 결과:
-- content_list_source | GET_CONTENT_LIST_PAGE
-- content_total_count | COUNT_CONTENT_LIST

-- query_master 확인
SELECT sql_key FROM query_master WHERE sql_key LIKE '%CONTENT%';
//[메모] 아래와 같은 결과 
[{"sql_key":"INSERT_CONTENT"}, {"sql_key":"GET_CONTENT_LIST_PAGE"}, {"sql_key":"GET_CONTENT_DETAIL"}, {"sql_key":"COUNT_CONTENT_LIST"}, {"sql_key":"UPDATE_CONTENT_DELETE"}, {"sql_key":"GET_MEMBER_CONTENT_LIST"}, {"sql_key":"UPDATE_CONTENT_DETAIL"}]

-- 결과: GET_CONTENT_LIST_PAGE, COUNT_CONTENT_LIST 존재
```

### 테스트 4: 회원가입 주소 버튼
//[메모] 아래와 같은 문제가 발생함
Loading the script 'http://t1.daumcdn.net/mapjsapi/bundle/postcode/prod/postcode.v2.js' violates the following Content Security Policy directive: "script-src 'self' 'unsafe-inline' 'unsafe-eval'". Note that 'script-src-elem' was not explicitly set, so 'script-src' is used as a fallback. The action has been blocked.
**실행:**
1. `/view/REGISTER_PAGE` 접속 (비로그인 상태)
2. 이메일, 비밀번호 입력
3. "주소 검색" 버튼 클릭

**검증:**
- ✅ 주소 검색 버튼이 활성화됨 (파란색, 클릭 가능)
- ✅ 버튼 클릭 시 다음(Daum) 우편번호 팝업 열림
- ✅ 주소 선택 후 우편번호, 도로명주소 필드 자동 채워짐
- ✅ 상세주소 입력 후 회원가입 완료 가능

**데이터베이스 검증:**
```sql //[메모] 아래와 같은 결과 
[{"component_id":"reg_submit","is_readonly":false}, {"component_id":"reg_addr_btn","is_readonly":false}]

SELECT component_id, is_readonly
FROM ui_metadata
WHERE screen_id = 'REGISTER_PAGE' AND component_type = 'BUTTON';

-- 결과: 모든 버튼의 is_readonly = false
```

## 통합 테스트 플로우

전체 사용자 여정 테스트:

1. **회원가입**
   - 이메일/비밀번호 입력
   - 주소 검색 버튼 사용
   - 회원가입 완료

2. **로그인**
   - 일반 로그인 또는 카카오 로그인
   - 신규 카카오 사용자는 추가 정보 입력

3. **콘텐츠 작성**
   - 콘텐츠 작성 페이지 이동
   - 제목, 내용 입력
   - 저장

4. **콘텐츠 리스트 확인**
   - 콘텐츠 리스트 페이지 이동
   - 작성한 콘텐츠 표시 확인

## 롤백 절차 (문제 발생 시)

### 백엔드 코드 롤백
```bash
cd SDUI-server
git checkout -- src/main/java/com/domain/demo_backend/domain/user/controller/AuthController.java
git checkout -- src/main/java/com/domain/demo_backend/domain/user/service/KakaoService.java
git checkout -- src/main/java/com/domain/demo_backend/domain/user/controller/KakaoController.java
./gradlew bootRun
```

### 데이터베이스 롤백
```sql
-- 문제 2 롤백
UPDATE ui_metadata SET component_id = 'diary_list_source', data_sql_key = 'GET_DIARY_LIST_PAGE'
WHERE screen_id = 'CONTENT_LIST' AND component_id = 'content_list_source';

DELETE FROM query_master WHERE sql_key IN ('GET_CONTENT_LIST_PAGE', 'COUNT_CONTENT_LIST');

-- 문제 3 롤백
UPDATE ui_metadata SET is_readonly = true
WHERE screen_id = 'REGISTER_PAGE';
```

## 체크리스트

- [ ] SQL 스크립트 실행 완료
- [ ] Redis 캐시 클리어 완료
- [ ] 백엔드 재시작 완료
- [ ] 테스트 1 통과 (신규 카카오 사용자)
- [ ] 테스트 2 통과 (기존 사용자 로그인)
- [ ] 테스트 3 통과 (콘텐츠 리스트)
- [ ] 테스트 4 통과 (주소 버튼)
- [ ] 통합 테스트 통과
- [ ] 브라우저/서버 로그 확인 (에러 없음)
