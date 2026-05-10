# Architect — Research

> 이 파일은 아키텍처 분석 결과를 기록한다. 모든 설계 결정의 근거가 된다.

---

## 현재 시스템 구조 분석 (2026-02-28 기준)

### 전체 기술 스택

| 레이어 | 기술 | 버전 |
|--------|------|------|
| Frontend | Next.js | 16.1.3 |
| UI Library | React | 19.2.3 |
| Language (FE) | TypeScript | 5.7.3 |
| Styling | Tailwind CSS | 4.0.0 |
| Data Fetching | TanStack React Query | 5.66.0 |
| HTTP Client | Axios | 1.7.9 |
| Backend | Spring Boot | 3.1.4 |
| Language (BE) | Java | 17 |
| ORM | JPA/Hibernate + MyBatis | - |
| Database | PostgreSQL | 15 |
| Cache | Redis | latest |
| Auth | JWT (jjwt 0.11.5) | - |
| WebSocket | STOMP | - |
| Container | Docker Compose | - |

---

### SDUI 핵심 데이터 플로우

```
URL /view/{screenId}/{refId?}
  → MetadataProvider
      → React Query: GET /api/ui/{screenId}
      → Query Key: {rolePrefix}_{screenId} (RBAC)
      → Stale Time: 5분
  → CommonPage
      → usePageMetadata
          → AUTO_FETCH 컴포넌트 감지
          → POST /api/execute/{sqlKey} (pageData)
      → usePageHook (액션 라우터)
  → DynamicEngine
      → useDynamicEngine (데이터 바인딩)
      → componentMap[component_type] → 렌더링
```

### ui_metadata 테이블 필드 전체 목록

```
ui_id               (PK, auto-increment)
screen_id           (화면 식별자: LOGIN_PAGE, DIARY_LIST 등)
component_id        (컴포넌트 고유 ID)
component_type      (React 컴포넌트 타입: INPUT, TEXT, BUTTON 등)
label_text          (표시 텍스트)
sort_order          (렌더링 순서)
is_required         (필수 입력 여부)
is_readonly         (읽기 전용 여부)
placeholder         (입력 힌트 텍스트)
css_class           (적용할 CSS 클래스명)
action_type         (클릭 시 실행 액션: LOGIN_SUBMIT, ROUTE 등)
ref_data_id         (pageData 키 연결, Repeater 식별자)
group_id            (자신의 그룹 ID)
parent_group_id     (부모 그룹 ID — 트리 구조 핵심)
group_direction     (ROW: 가로 | COLUMN: 세로)
is_visible          (가시성 조건)
inline_style        (인라인 CSS)
data_sql_key        (query_master 연결 키)
default_value       (기본값)
submit_group_id     (폼 제출 그룹)
submit_group_order  (제출 순서)
submit_group_separator (제출 구분자)
```

### 트리 빌딩 알고리즘 분석 (UiService.java)

```
시간 복잡도: O(n) — LinkedHashMap 활용
공간 복잡도: O(n)

단계:
1. screen_id로 모든 레코드 조회 (sort_order ASC)
2. LinkedHashMap<componentId, UiResponseDto> 생성
3. 순회: parent_group_id 있으면 부모의 children에 추가
4. 루트 노드 = parent_group_id가 null 또는 Map에 없는 것
5. 고아 노드(부모 없음) → 루트로 승격

리스크:
- 중복 componentId → 감지 로직 있음 (경고 로그)
- 순환 참조 → 현재 감지 로직 없음 (잠재 무한루프)
```

### Redis 캐시 키 구조

| 키 패턴 | 값 | TTL | 무효화 시점 |
|---------|----|-----|------------|
| `SQL:{sqlKey}` | query_master.query_text | 영구 | query_master 변경 시 수동 삭제 |
| `ui:metadata:{screenId}` | List&lt;UiMetadata&gt; JSON | **1시간** | 수동 삭제 (`UiMetadataService`) |
| `{userId}` (Refresh Token) | RefreshToken 객체 | 7일 | logout, 토큰 재발급 |

### React Query 캐시 키 구조

| 키 패턴 | Stale Time | 무효화 시점 |
|---------|------------|------------|
| `[metadata, {rolePrefix}_{screenId}]` | 5분 | SUBMIT 액션 후 invalidate |

### componentMap 현황 (18개 타입) — 2026-03-06 확인

```
MODAL, INPUT, TEXT, PASSWORD, BUTTON, SNS_BUTTON, LINK_BUTTON, IMAGE,
EMAIL_SELECT, EMOTION_SELECT, SELECT, TEXTAREA,
TIME_RECORD_WIDGET, DATETIME_PICKER, TIME_SELECT,
TIME_SLOT_RECORD, ADDRESS_SEARCH_GROUP, GROUP
```

> LINK_BUTTON 추가(ButtonField 재사용), GROUP(GroupComponent)으로 18개

### screenMap 현황 (9개 경로, 8개 screen_id) — 2026-03-06 확인

```
"/"                → MAIN_PAGE
"/MAIN_PAGE"       → MAIN_PAGE  (추가됨)
"/LOGIN_PAGE"      → LOGIN_PAGE
"/SET_TIME_PAGE"   → SET_TIME_PAGE
"/TUTORIAL_PAGE"   → TUTORIAL_PAGE
"/CONTENT_LIST"    → CONTENT_LIST   (DIARY_LIST → 변경)
"/CONTENT_WRITE"   → CONTENT_WRITE  (DIARY_WRITE → 변경)
"/CONTENT_DETAIL"  → CONTENT_DETAIL (DIARY_DETAIL → 변경)
"/CONTENT_MODIFY"  → CONTENT_MODIFY (DIARY_MODIFY → 변경)
// "/MY_PAGE"      → MY_PAGE (주석 처리 — 미구현)
// "/DASHBOARD_PAGE" → DASHBOARD_PAGE (주석 처리 — 미구현)
```

### 보호 화면 목록 (인증 필요) — 2026-03-06 확인

```
MY_PAGE, CONTENT_LIST, CONTENT_WRITE, CONTENT_DETAIL, CONTENT_MODIFY
```

### 인증 플로우 분석

```
회원가입: POST /api/auth/register → 이메일 발송 → VERIFY_CODE_PAGE
인증:     POST /api/auth/verify-code → verifyYn='Y'
로그인:   POST /api/auth/login → AccessToken(1h) + RefreshToken(7d)
카카오:   GET  /api/kakao → OAuth → 자동 회원가입/로그인
갱신:     POST /api/auth/refresh → 새 AccessToken
로그아웃: POST /api/auth/logout → Redis RefreshToken 삭제
```

### 현재 확인된 아키텍처 리스크

1. **순환 참조 감지 없음**: `UiService` 트리 빌딩 시 `parent_group_id` 순환 참조 발생 시 무한 루프 가능성
2. **Redis 캐시 수동 관리**: `SQL:{sqlKey}` 캐시는 자동 만료 없음. 쿼리 변경 시 수동 삭제 필요
3. **Metadata 이중 필드**: `Metadata` 타입에 `componentId`와 `component_id` 모두 존재 (snake_case, camelCase 혼용)
4. **단일 DynamicEngine**: 모든 화면이 동일한 엔진을 사용. 특정 화면 전용 로직이 엔진에 추가될 위험

---

## 분석 히스토리

| 날짜 | 분석 내용 | 결론 |
|------|-----------|------|
| 2026-02-28 | 전체 코드베이스 초기 분석 | 위 내용 도출 |
| 2026-03-06 | componentMap/screenMap/Redis 재확인 | LINK_BUTTON 추가(18개), DIARY→CONTENT 전환 반영, UI Redis 캐시(`ui:metadata:{screenId}`, 1h) 확인 |
| 2026-03-11 | 링글 과제 → SDUI AI 튜터+면접관 전환 분석 | .ai2 작업 → 아래 섹션 병합 완료 |

---

## AI 튜터 + 면접관 아키텍처 분석 (2026-03-11, .ai2 병합)

> 원본: `.ai2/architect/research.md`

### 요구사항 비교

| 기능 | 원안 (링글) | SDUI 확장 |
|------|------------|-----------|
| 백엔드 | Rails 7 | Spring Boot (SDUI-server) |
| 인증 | X-User-Id 헤더 | 기존 JWT 그대로 |
| 어드민 UI | 별도 React 페이지 | SDUI ui_metadata |
| AI 응답 표시 | TTS 음성 | 채팅창 텍스트 스트리밍 |
| 언어 | 영어 전용 | 영어 + 한국어 |
| 신규 기능 | 없음 | AI 면접관 (이력서 → 질문 → 음성 답변) |

### 기술 선택

| 항목 | 선택 | 이유 |
|------|------|------|
| STT | OpenAI `whisper-1` | 영어/한국어 모두 지원 |
| LLM | `gpt-4o` + Streaming | 대화 + 이력서 분석 |
| TTS | 미사용 (텍스트만 표시) | 사용자 요청 |
| HTTP 클라이언트 | Spring `RestClient` (내장) | 외부 SDK 의존성 없음 |
| SSE | Spring `SseEmitter` | 기존 MVC 유지 (WebFlux 불필요) |

### SDUI 커스텀 컴포넌트 등록 패턴

```typescript
// componentMap.tsx 추가
AI_CHAT: AIChatComponent,
AI_CHAT_V2: AIChatComponentV2,
AI_INTERVIEW: AIInterviewComponent,
```

### 리스크

| 리스크 | 대응 |
|--------|------|
| OpenAI 레이턴시 | SSE Streaming으로 체감 지연 감소 |
| AudioContext 최대 6개 | useRef 단일 인스턴스 패턴 |
| SseEmitter 타임아웃 | 30초 설정 + 클라이언트 retry |
| Flyway label_text NOT NULL | GROUP 행에 `label_text, ''` 명시 |