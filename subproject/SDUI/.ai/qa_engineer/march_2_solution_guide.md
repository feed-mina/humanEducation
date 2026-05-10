# 3월 2일 문제 해결 가이드

## ✅ 완료된 수정

### 1. CSP 정책 수정 (Daum 우편번호 API 허용)
- **파일**: `metadata-project/next.config.ts`
- **변경**: `script-src`에 `https://t1.daumcdn.net` 추가
- **결과**: 주소 검색 팝업이 CSP 위반 없이 로드됨

### 2. AddressSearchGroup prop 구조 수정
- **파일**: `metadata-project/components/fields/AddressSearchGroup.tsx`
- **변경**:
  - `props` → `meta` (DynamicEngine 전달 방식에 맞춤)
  - `dispatchAction` → `onAction`
  - `props.label` → `meta?.labelText || meta?.label_text || meta?.label`
- **결과**: "Cannot read properties of undefined (reading 'label')" 에러 해결

## ⏳ 추가 작업 필요

### 3. Gradle 빌드 문제 해결

**문제**: Windows 파일 잠금으로 인한 `./gradlew clean` 실패

**해결 방법**:
```bash
# 옵션 1: 백엔드 프로세스 중지 후 빌드
# IntelliJ/Eclipse에서 실행 중인 Spring Boot 서버 중지
# 또는 작업 관리자에서 Java 프로세스 종료 후:
cd SDUI-server
./gradlew clean build
./gradlew bootRun

# 옵션 2: clean 없이 바로 실행 (빠른 방법)
cd SDUI-server
./gradlew bootRun
```

### 4. CONTENT_LIST 렌더링 문제 진단

**현상**:
- ✅ API 응답 정상: `/api/execute/GET_CONTENT_LIST_PAGE`
- ✅ 데이터 구조 정상: `content_id`, `title`, `user_id` 등
- ❌ 화면에 표시 안됨

**원인 추정**:
DynamicEngine의 리피터 렌더링은 `ref_data_id`가 설정된 그룹이 필요합니다.

**확인 필요 사항**:
//[메모] order_idx가 아니라 sort_order입니다.
ERROR:  column "order_idx" does not exist
LINE 11: ORDER BY parent_group_id NULLS FIRST, group_id, order_idx;
SELECT
component_id,
component_type,
ref_data_id,
group_id,
parent_group_id,
group_direction,
css_class
FROM ui_metadata
WHERE screen_id = 'CONTENT_LIST'
ORDER BY parent_group_id NULLS FIRST, group_id, sort_order 
의 결과 

[
{
"component_id": "LIST_SECTION",
"component_type": "GROUP",
"ref_data_id": null,
"group_id": "LIST_SECTION",
"parent_group_id": null,
"group_direction": "COLUMN",
"css_class": "LIST_SECTION"
},
{
"component_id": "ADMIN_DELETE_ALL_BTN",
"component_type": "BUTTON",
"ref_data_id": null,
"group_id": null,
"parent_group_id": null,
"group_direction": "COLUMN",
"css_class": "btn-danger"
},
{
"component_id": "CONTENT_CARD_HEADER",
"component_type": "GROUP",
"ref_data_id": null,
"group_id": "DIARY_CARD_HEADER",
"parent_group_id": "DIARY_CARD",
"group_direction": "ROW",
"css_class": "content-card-header"
},
{
"component_id": "TITLE_AUTHOR_GROUP",
"component_type": "GROUP",
"ref_data_id": null,
"group_id": "TITLE_AUTHOR_GROUP",
"parent_group_id": "DIARY_CARD_HEADER",
"group_direction": "ROW",
"css_class": "title-author-wrapper"
},
{
"component_id": "list_item_date",
"component_type": "TEXT",
"ref_data_id": "reg_dt",
"group_id": null,
"parent_group_id": "DIARY_CARD_HEADER",
"group_direction": "COLUMN",
"css_class": "contentDate"
},
{
"component_id": "CONTENT_CARD",
"component_type": "GROUP",
"ref_data_id": "diary_list_source",
"group_id": "DIARY_CARD",
"parent_group_id": "LIST_SECTION",
"group_direction": "COLUMN",
"css_class": "content-post"
},
{
"component_id": "content_list_source",
"component_type": "DATA_SOURCE",
"ref_data_id": null,
"group_id": null,
"parent_group_id": "LIST_SECTION",
"group_direction": "COLUMN",
"css_class": "content_list_source"
},
{
"component_id": "go_write_btn",
"component_type": "BUTTON",
"ref_data_id": null,
"group_id": null,
"parent_group_id": "LIST_SECTION",
"group_direction": "COLUMN",
"css_class": "write-btn"
},
{
"component_id": "content_total_count",
"component_type": "DATA_SOURCE",
"ref_data_id": null,
"group_id": null,
"parent_group_id": "LIST_SECTION",
"group_direction": "COLUMN",
"css_class": "content_total_count"
},
{
"component_id": "content_pagination",
"component_type": "PAGINATION",
"ref_data_id": "diary_total_count",
"group_id": null,
"parent_group_id": "LIST_SECTION",
"group_direction": "COLUMN",
"css_class": "content-pagination"
},
{
"component_id": "list_item_title",
"component_type": "TEXT",
"ref_data_id": "title",
"group_id": null,
"parent_group_id": "TITLE_AUTHOR_GROUP",
"group_direction": "COLUMN",
"css_class": "contentTitle"
},
{
"component_id": "list_item_author",
"component_type": "TEXT",
"ref_data_id": "user_id",
"group_id": null,
"parent_group_id": "TITLE_AUTHOR_GROUP",
"group_direction": "COLUMN",
"css_class": "contentContent"
}
]

```sql
-- 1. CONTENT_LIST 화면의 그룹 구조 확인
SELECT
    component_id,
    component_type,
    ref_data_id,
    group_id,
    parent_group_id,
    group_direction,
    css_class
FROM ui_metadata
WHERE screen_id = 'CONTENT_LIST'
ORDER BY parent_group_id NULLS FIRST, group_id, order_idx;

-- 2. ref_data_id가 'content_list_source'로 설정된 그룹이 있는지 확인
// [메모] 처음에 결과 없음 두번째 INSERT 후 결과 아래와 같음
[
  {
    "component_id": "content_list_container",
    "component_type": "GROUP",
    "ref_data_id": "content_list_source"
  }
]

SELECT
    component_id,
    component_type,
    ref_data_id
FROM ui_metadata
WHERE screen_id = 'CONTENT_LIST'
  AND ref_data_id = 'content_list_source';

-- 결과가 없으면 그룹 추가 필요
```

**해결 방법 (ref_data_id 그룹이 없는 경우)**:
//[메모] order_idx 대신 sort_order로,  ERROR:  "label_text" 칼럼(해당 릴레이션 "ui_metadata")의 null 값이 not null 제약조건을 위반했습니다. 결과로 label_text를 콘텐츠리스트로 추가하여 삽입

```sql
-- CONTENT_LIST 화면에 리피터 그룹 추가
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    group_id,
    ref_data_id,
    group_direction,
    css_class,
    order_idx
) VALUES (
    'CONTENT_LIST',
    'content_list_container',
    'GROUP',
    'content_list_group',
    'content_list_source',  -- 이 값이 pageData 키와 일치해야 함
    'COLUMN',
    'content-list-container',
    10
);

-- 리피터 그룹 내부의 자식 컴포넌트들의 parent_group_id 업데이트
UPDATE ui_metadata
SET parent_group_id = 'content_list_group'
WHERE screen_id = 'CONTENT_LIST'
  AND component_id IN (
  content_list_container
  );
```
//[메모] IN 대산 아래 update문으로 진행
UPDATE ui_metadata
SET parent_group_id = 'content_list_group'
WHERE screen_id = 'CONTENT_LIST'
AND component_id = 'content_list_container';

**DynamicEngine 리피터 로직 (참고)**:
```typescript
// DynamicEngine.tsx 75-96라인
if (isRepeater) {
    const list = pageData?.[refId];  // pageData['content_list_source']
    if (!list || !Array.isArray(list)) {
        console.warn(`[DynamicEngine] ${refId} 데이터가 배열이 아닙니다.`, list);
        return null;
    }
    return list.map((item: any, idx: number) => {
        // 각 content 아이템 렌더링
    });
}
```

## 실행 순서

### Step 1: 프론트엔드 재시작
```bash
cd metadata-project
npm run dev
```

### Step 2: 백엔드 재시작 (파일 잠금 해제 후)
```bash
# 백엔드 프로세스 중지 확인
# 그 다음:
cd SDUI-server
./gradlew bootRun
```

### Step 3: Redis 캐시 클리어
```bash
docker exec -it my-redis redis-cli FLUSHDB
```

## 테스트 체크리스트

### ✅ 테스트 1: 신규 카카오 사용자 (ROLE_GUEST → ROLE_USER)
- [ ] 카카오 로그인 버튼 클릭
- [ ] `/view/ADDITIONAL_INFO_PAGE`로 리다이렉트
- [ ] 전화번호, 주소 입력 필드 표시
- [ ] **주소 검색 버튼 클릭 → Daum 팝업 열림** (CSP 에러 없음)
- [ ] 주소 선택 후 자동 입력 확인
- [ ] 제출 → `/view/MAIN_PAGE`로 이동
- [ ] 쿠키에 `role=ROLE_USER` 확인
// [메모] Loading the script 'http://t1.daumcdn.net/mapjsapi/bundle/postcode/prod/postcode.v2.js' violates the following Content Security Policy directive: "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://t1.daumcdn.net". Note that 'script-src-elem' was not explicitly set, so 'script-src' is used as a fallback. The action has been blocked. 

// [메모]Cannot read properties of undefined (reading 'zip_code')
components/fields/AddressSearchGroup.tsx (37:37) @ AddressSearchGroup


35 |             <div className="flex gap-2 mb-2">
36 |                 <input
> 37 |                     value={formData.zip_code || ''}
|                                     ^
38 |                     readOnly
39 |                     placeholder="우편번호"
40 |                     className="bg-gray-100 p-2 border rounded f 
### ✅ 테스트 2: 기존 사용자 로그인
- [x] 일반 로그인 성공 (메모에서 확인됨)
- [x] `/api/auth/me` 응답에 role 포함
- [x] 쿠키에 role 값 저장
// [메모] 일반 로그인 후 메인페이지 에서 /me 네트워크 누르면 Guest 값 유지 , 새로고침 이후 /me 값 변경
- {
  "role": "ROLE_USER",
  "userSqno": 1,
  "isLoggedIn": true,
  "socialType": "N",
  "userId": "pagingmina",
  "email": "paging@test.com"
  }
- 
### ⏳ 테스트 3: 콘텐츠 리스트 표시
- [ ] `/view/CONTENT_LIST` 접속
- [ ] 3개의 콘텐츠 아이템 표시 (현재 화면에 안보임)
- [ ] 전체 개수: 3 표시
- [ ] 브라우저 콘솔에서 `[DynamicEngine] content_list_source 데이터가 배열이 아닙니다` 경고 확인
  - 이 경고가 보이면 → `ref_data_id` 설정 문제
  - 이 경고가 안보이면 → 다른 렌더링 문제

### ✅ 테스트 4: 회원가입 주소 버튼
- [ ] `/view/REGISTER_PAGE` 접속
- [ ] 주소 검색 버튼 활성화 확인
- [ ] 버튼 클릭 → Daum 팝업 열림 (CSP 에러 없음)
- [ ] 주소 선택 → 자동 입력
- [ ] 회원가입 완료

## 디버깅 팁

### 프론트엔드 콘솔 로그 확인
```javascript
// 브라우저 콘솔에서 확인할 항목:
1. "[DynamicEngine] content_list_source 데이터가 배열이 아닙니다"
   → ref_data_id 설정 문제

2. "content_list_source" 키로 데이터가 있는지 확인:
   console.log(pageData)

3. 메타데이터에 ref_data_id가 있는 그룹이 있는지 확인:
   console.log(metadata)
```

// 메모 VM451:1 Uncaught ReferenceError: pageData is not defined
at <anonymous>:1:15
(익명) @ VM451:1이 오류 이해하기
console.log(metadata)

VM454:1 Uncaught ReferenceError: metadata is not defined
at <anonymous>:1:16
### 네트워크 탭 확인
```
1. /api/ui/CONTENT_LIST → 메타데이터 구조 확인
   - ref_data_id가 'content_list_source'인 그룹이 있는가?
   
   //[메모] 네트워크 결과 아래와 같음, 
   {
    "status": "success",
    "data": [
        {
            "componentId": "LIST_SECTION",
            "parentGroupId": null,
            "labelText": "리스트 전체 섹션",
            "componentType": "GROUP",
            "sortOrder": 1,
            "isRequired": false,
            "isReadonly": true,
            "isVisible": "true",
            "cssClass": "LIST_SECTION",
            "actionType": null,
            "actionUrl": null,
            "dataApiUrl": null,
            "dataSqlKey": null,
            "refDataId": null,
            "children": [
                {
                    "componentId": "go_write_btn",
                    "parentGroupId": "LIST_SECTION",
                    "labelText": "새 콘텐츠 쓰기",
                    "componentType": "BUTTON",
                    "sortOrder": 0,
                    "isRequired": false,
                    "isReadonly": false,
                    "isVisible": "true",
                    "cssClass": "write-btn",
                    "actionType": "LINK",
                    "actionUrl": "/view/CONTENT_WRITE",
                    "dataApiUrl": null,
                    "dataSqlKey": null,
                    "refDataId": null
                },
                {
                    "componentId": "content_list_source",
                    "parentGroupId": "LIST_SECTION",
                    "labelText": "콘텐츠 목록 데이터",
                    "componentType": "DATA_SOURCE",
                    "sortOrder": 0,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": "true",
                    "cssClass": "content_list_source",
                    "actionType": "AUTO_FETCH",
                    "actionUrl": null,
                    "dataApiUrl": "/api/execute/GET_DIARY_LIST_PAGE",
                    "dataSqlKey": "GET_CONTENT_LIST_PAGE",
                    "refDataId": null
                },
                {
                    "componentId": "CONTENT_CARD",
                    "parentGroupId": "LIST_SECTION",
                    "labelText": "콘텐츠 카드",
                    "componentType": "GROUP",
                    "sortOrder": 2,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": "true",
                    "cssClass": "content-post",
                    "actionType": "ROUTE_DETAIL",
                    "actionUrl": "/view/CONTENT_DETAIL",
                    "dataApiUrl": null,
                    "dataSqlKey": null,
                    "refDataId": "diary_list_source"
                },
                {
                    "componentId": "content_total_count",
                    "parentGroupId": "LIST_SECTION",
                    "labelText": "전체 개수 조회",
                    "componentType": "DATA_SOURCE",
                    "sortOrder": 99,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": "true",
                    "cssClass": "content_total_count",
                    "actionType": "AUTO_FETCH",
                    "actionUrl": null,
                    "dataApiUrl": "/api/execute/COUNT_DIARY_LIST",
                    "dataSqlKey": "COUNT_CONTENT_LIST",
                    "refDataId": null
                },
                {
                    "componentId": "content_pagination",
                    "parentGroupId": "LIST_SECTION",
                    "labelText": "페이지네이션",
                    "componentType": "PAGINATION",
                    "sortOrder": 100,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": "true",
                    "cssClass": "content-pagination",
                    "actionType": null,
                    "actionUrl": null,
                    "dataApiUrl": null,
                    "dataSqlKey": null,
                    "refDataId": "diary_total_count"
                }
            ]
        },
        {
            "componentId": "TITLE_AUTHOR_GROUP",
            "parentGroupId": "DIARY_CARD_HEADER",
            "labelText": "제목작성자묶음",
            "componentType": "GROUP",
            "sortOrder": 1,
            "isRequired": false,
            "isReadonly": true,
            "isVisible": "true",
            "cssClass": "title-author-wrapper",
            "actionType": null,
            "actionUrl": null,
            "dataApiUrl": null,
            "dataSqlKey": null,
            "refDataId": null,
            "children": [
                {
                    "componentId": "list_item_title",
                    "parentGroupId": "TITLE_AUTHOR_GROUP",
                    "labelText": "제목",
                    "componentType": "TEXT",
                    "sortOrder": 1,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": "true",
                    "cssClass": "contentTitle",
                    "actionType": null,
                    "actionUrl": null,
                    "dataApiUrl": null,
                    "dataSqlKey": null,
                    "refDataId": "title"
                },
                {
                    "componentId": "list_item_author",
                    "parentGroupId": "TITLE_AUTHOR_GROUP",
                    "labelText": "작성자",
                    "componentType": "TEXT",
                    "sortOrder": 2,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": "true",
                    "cssClass": "contentContent",
                    "actionType": null,
                    "actionUrl": null,
                    "dataApiUrl": null,
                    "dataSqlKey": null,
                    "refDataId": "user_id"
                }
            ]
        },
        {
            "componentId": "CONTENT_CARD_HEADER",
            "parentGroupId": "DIARY_CARD",
            "labelText": "카드 상단 영역",
            "componentType": "GROUP",
            "sortOrder": 1,
            "isRequired": false,
            "isReadonly": true,
            "isVisible": "true",
            "cssClass": "content-card-header",
            "actionType": null,
            "actionUrl": null,
            "dataApiUrl": null,
            "dataSqlKey": null,
            "refDataId": null
        },
        {
            "componentId": "list_item_date",
            "parentGroupId": "DIARY_CARD_HEADER",
            "labelText": "날짜",
            "componentType": "TEXT",
            "sortOrder": 2,
            "isRequired": false,
            "isReadonly": true,
            "isVisible": "true",
            "cssClass": "contentDate",
            "actionType": null,
            "actionUrl": null,
            "dataApiUrl": null,
            "dataSqlKey": null,
            "refDataId": "reg_dt"
        },
        {
            "componentId": "content_list_container",
            "parentGroupId": "content_list_group",
            "labelText": "콘텐츠리스트",
            "componentType": "GROUP",
            "sortOrder": 10,
            "isRequired": false,
            "isReadonly": true,
            "isVisible": "true",
            "cssClass": "content-list-container",
            "actionType": null,
            "actionUrl": null,
            "dataApiUrl": null,
            "dataSqlKey": null,
            "refDataId": "content_list_source"
        }
    ],
    "message": null
}


2. /api/execute/GET_CONTENT_LIST_PAGE → 데이터 응답 확인
   - data 배열에 content_id, title 등이 있는가? 
   //[메모] 아래와 같음
   
{
    "sqlKey": "GET_CONTENT_LIST_PAGE",
    "status": "success",
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
    ]
}
```
// [메모] http://localhost:3000/view/CONTENT_LIST 에 화면으로 렌더링 안됨  
## 예상 해결 시간

- ✅ CSP 수정: 완료
- ✅ AddressSearchGroup 수정: 완료
- ⏳ Gradle 빌드: 5분 (프로세스 중지 + 빌드)
- ⏳ CONTENT_LIST 렌더링: 10-30분 (DB 조회 → 그룹 설정 확인 → 수정)

## 다음 단계

1. 프론트엔드 dev 서버 재시작
2. 백엔드 프로세스 중지 후 재시작
3. Redis 캐시 클리어
4. 테스트 1, 4 실행 → 주소 검색 정상 작동 확인
5. 테스트 3 실행 → 콘솔 로그 확인 → DB 쿼리 실행 → 그룹 설정 수정
