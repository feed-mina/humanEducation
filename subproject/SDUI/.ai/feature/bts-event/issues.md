# BTS 이벤트 페이지 + SDUI 프로젝트 이슈 분석

> 작성일: 2026-03-21
> 최종 수정: 2026-03-21 (2차 분석 - 추가 버그 발견 및 수정 완료)
> 브랜치 현황: `feature/bts-event-complete` (로컬), `main` (EC2 배포 기준)
> 분석 범위: bts-event 페이지 전체 기능 + SDUI 프로젝트 영향 범위

---

## 핵심 요약

| 구분 | 이슈 수 | 심각도 |
|------|---------|--------|
| BTS 이벤트 페이지 | 10개 | Critical 4 / Warning 3 / 추가버그 3 |
| SDUI 프로젝트 | 2개 | Critical 1 / Warning 1 |

**공통 근본 원인**: `feature/bts-event-complete` 브랜치의 핵심 변경사항(SecurityConfig, AiChatController, V35 마이그레이션)이 `main` 브랜치에 미머지된 채로 EC2 배포됨.

---

## A. BTS 이벤트 페이지 이슈

### A-1. [CRITICAL] AI 채팅 완전 불가
- **증상**: 채팅 입력 후 전송 시 오류 / 응답 없음
- **원인**: `POST /api/ai/guest/chat` 엔드포인트가 `main` 브랜치의 `AiChatController`에 없음
  - feature 브랜치: `@PostMapping("/guest/chat")` + `ChatService.createGuestReply()` 존재
  - main 브랜치: `AiChatController`에 `/guest/chat` 없음 → 404 반환
- **추가 원인**: SecurityConfig(main)에 `/api/ai/guest/**` permitAll 없음 → 401 반환 (404보다 먼저 차단)
- **파일**: `SDUI-server/.../ai/controller/AiChatController.java`, `SecurityConfig.java`
- **수정 방향**: SecurityConfig + AiChatController의 guest chat 엔드포인트를 main에 반영

---

### A-2. [CRITICAL] 이미지 업로드 "로그인 필요" 오류
- **증상**: FanBoard 글쓰기 시 이미지 첨부 → "Image upload failed. Please check your connection or login status."
- **원인 1**: main 브랜치 SecurityConfig에 `/api/ai/interview/resume/upload` permitAll 없음 → 401
- **원인 2**: main 브랜치 `AiInterviewController.uploadResume()`에서 `userDetails.getUserSqno()` 호출 시 null 체크 없음 → 설령 permitAll이 있어도 NPE 발생
  - feature 브랜치: `(userDetails != null) ? userDetails.getUserSqno() : 0L` 처리됨
  - main 브랜치: `userDetails.getUserSqno()` 직접 호출 → NPE
- **파일**: `SecurityConfig.java`, `AiInterviewController.java`
- **수정 방향**: permitAll 추가 + null 체크 null-safe 처리 main 반영

---

### A-3. [CRITICAL] 음성 입력(STT) 불가
- **증상**: 마이크 버튼 녹음 후 전송 실패
- **원인 1**: main 브랜치 SecurityConfig에 `/api/ai/stt` permitAll 없음 → 401
- **원인 2**: main 브랜치 `AiSttController`에서 `userDetails.getUserSqno()` null 체크 없음 → NPE
  - feature 브랜치: `userDetails != null ? ... : "GUEST"` 처리됨
  - main 브랜치: `userDetails.getUserSqno()` 직접 호출 → NPE
- **파일**: `SecurityConfig.java`, `AiSttController.java`
- **수정 방향**: permitAll 추가 + null-safe 처리 main 반영

---

### A-4. [CRITICAL] 팬 게시판 FanBoard 데이터 조회/작성 불가
- **증상**: 게시판 목록 로딩 실패, 게시글 작성 실패
- **원인**: V35 마이그레이션이 main 브랜치에 없음 → EC2 DB에 fan_board 테이블 없음
  - `GET_FANBOARD_LIST`, `GET_FANBOARD_DETAIL`, `INSERT_FANBOARD` 쿼리도 query_master에 없음
  - `/api/execute/GET_FANBOARD_LIST` → query_master에서 sql_key 조회 실패 → 에러
- **추가 이슈**: `UPDATE_CONTENT_DETAIL` 쿼리도 V35에 없음 → 게시글 수정 기능 항상 실패
- **파일**: `V35__setup_fanboard_system.sql` (feature 브랜치에만 존재)
- **수정 방향**: V35 마이그레이션을 main에 반영 후 EC2 재배포 (단, 아래 SDUI 이슈 A4 확인 필수)

---

### A-5. [WARNING] CCTV 버튼 연결 불가 ✅ 수정 완료
- **증상**: CCTV 버튼 클릭 시 "사이트에 연결할 수 없습니다"
- **원인**: `InfoPanel.tsx`에서 `https://cctv.seoul.go.kr` 사용 → SSL 미지원 사이트
- **수정**: `http://cctv.seoul.go.kr` 으로 변경 완료 (`InfoPanel.tsx:84`)

---

### A-6. [WARNING] Vercel 환경변수 NEXT_PUBLIC_API_BASE 미설정 가능성
- **증상**: 모든 API 호출 실패 (채팅, 게시판 포함)
- **원인**: `bts-event/next.config.ts`에서 `NEXT_PUBLIC_API_BASE || "http://localhost:8080"` 사용
  - Vercel 대시보드에 이 변수가 설정 안 된 경우, 모든 API 요청이 localhost:8080으로 라우팅 → 실패
- **수정**: Vercel 환경변수 확인 완료 — `NEXT_PUBLIC_API_BASE = https://yerin.duckdns.org` ✅

---

### A-7. [WARNING] GuestChat.tsx 미커밋 변경사항 ✅ 추가 버그 수정 완료
- **증상**: 마이크 녹음 후 자동 전송 기능이 Vercel에 미반영
- **원인**: `handleVoiceToText()` 에서 `setInput(data.data.text)` → `handleSend(data.data.text)` 자동 전송으로 변경됨 (로컬에서 수정됨)
- **추가 발견 버그**: `handleSend` 내부의 `if (!textToSend.trim() || loading) return` 조건이 문제
  - `handleVoiceToText`가 STT 후 `loading=true` 상태에서 `handleSend`를 호출하면 loading 체크에서 즉시 return → 자동 전송 무시됨
  - `handleVoiceToText`가 `handleSend`를 await 없이 호출하면 finally가 먼저 실행되어 loading 관리 충돌
- **수정 완료**:
  - `handleSend`: `loading && !manualText` 조건으로 변경 (manualText 있으면 loading 무시)
  - `handleVoiceToText`: `await handleSend(data.data.text)` 로 변경

---

### A-8. [CRITICAL] FanBoard 게시글 목록 미표시 ✅ 수정 완료
- **증상**: V35 적용 후에도 게시판 목록이 비어있음
- **원인**: `CommonQueryController`가 `{ status: "success", data: [...] }` 형태로 응답하는데, `FanBoard.tsx`에서 `data.code === "SUCCESS"` 만 체크 → 조건이 항상 false
  - `fetchPosts()`: `data.code === "SUCCESS" && Array.isArray(data.data)` → 목록 미표시
  - `openDetail()`: `data.code === "SUCCESS"` → 상세 화면 미진입
- **수정 완료**: `(data.code === "SUCCESS" || data.status === "success")` 로 변경 (두 형식 모두 처리)

---

### A-9. [CRITICAL] FanBoard 상세 조회 빈 화면 ✅ 수정 완료
- **증상**: 게시글 클릭 시 상세 화면으로 이동 안 됨
- **원인 1**: A-8과 동일한 `data.code` 체크 문제
- **원인 2**: `GET_FANBOARD_DETAIL`의 `return_type = 'MAP'` 오류 → `CommonQueryController`에서 MAP 타입 처리 없음 → 리스트 형태 반환 → `setSelectedPost([{...}])` (배열이 객체 대신 저장됨)
- **수정 완료**:
  - `FanBoard.tsx`: `data.status === "success"` 체크 추가
  - `V35__setup_fanboard_system.sql`: `GET_FANBOARD_DETAIL` return_type `'MAP'` → `'SINGLE'` (list.get(0) 반환)

---

### A-10. [CRITICAL] 업로드 이미지 표시 불가 ✅ 수정 완료
- **증상**: 게시글에 첨부한 이미지가 게시판에서 표시 안 됨
- **원인**: `GET /api/ai/interview/resume/view` 엔드포인트가 SecurityConfig에서 허용 목록 없음
  - POST `/api/ai/**` → authenticated 처리
  - GET `/api/ai/interview/resume/view` → `anyRequest().denyAll()` 에 걸려 403 반환
  - `<img src="/api/ai/interview/resume/view?fileKey=...">` 로드 실패
- **수정 완료**: `SecurityConfig.java`에 `/api/ai/interview/resume/view` permitAll 추가

---

## B. SDUI 프로젝트 이슈 (이벤트 배포 영향)

### B-1. [CRITICAL] V35 마이그레이션이 GET_CONTENT_LIST_PAGE 파괴 → ✅ V35 내에서 수정 완료
- **증상 (V35 미수정 시)**: SDUI 콘텐츠 목록에서 filterId가 day_tag1 필터로 바뀌어 사용자별 필터 불가
- **원인 분석**:
  - V31 GET_CONTENT_LIST_PAGE: `filterId = user_id` 기반 필터, 올바른 상태
  - V35 UPDATE: `filterId = day_tag1` 기반으로 덮어씀 → SDUI 콘텐츠 목록 파괴
  - FanBoard는 `GET_FANBOARD_LIST`를 사용하므로 V35에서 GET_CONTENT_LIST_PAGE 변경 불필요
- **수정**: V35에서 GET_CONTENT_LIST_PAGE UPDATE 구문 완전 제거

---

### B-2. [WARNING] SDUI CORS — bts-gwanghwamun.vercel.app 미등록 ✅ 수정 완료
- **증상**: bts-event 페이지에서 클라이언트 사이드 직접 호출 시 CORS 오류 가능성
- **원인**: `SecurityConfig.corsConfigurationSource()`의 allowedOrigins에 `https://bts-gwanghwamun.vercel.app` 없음
- **수정 완료**: `SecurityConfig.java:129`에 추가됨

---

## 수정 완료 목록 (2026-03-21)

| # | 파일 | 수정 내용 |
|---|------|----------|
| 1 | `bts-event/components/InfoPanel.tsx:84` | CCTV URL https→http |
| 2 | `SecurityConfig.java` | CORS: bts-gwanghwamun.vercel.app 추가 |
| 3 | `SecurityConfig.java` | permitAll: /api/ai/interview/resume/view 추가 (이미지 표시) |
| 4 | `V35__setup_fanboard_system.sql` | LIKE content INCLUDING ALL → LIKE content INCLUDING DEFAULTS (PK/인덱스 이름 충돌 방지 → 마이그레이션 실패 원인) |
| 5 | `V35__setup_fanboard_system.sql` | GET_CONTENT_LIST_PAGE UPDATE 제거 (SDUI 콘텐츠 목록 파괴 방지) |
| 6 | `V35__setup_fanboard_system.sql` | UPDATE_FANBOARD 쿼리 추가 (fan_board 테이블 수정용) |
| 7 | `V35__setup_fanboard_system.sql` | GET_FANBOARD_DETAIL return_type: MAP → SINGLE (단일 객체 반환) |
| 8 | `bts-event/lib/api.ts` | updateBoardPost: UPDATE_CONTENT_DETAIL → UPDATE_FANBOARD |
| 9 | `bts-event/components/Chat/GuestChat.tsx` | handleSend loading 체크 수정 (manualText 시 loading 무시) |
| 10 | `bts-event/components/Chat/GuestChat.tsx` | handleVoiceToText: await handleSend 변경 |
| 11 | `bts-event/components/Board/FanBoard.tsx` | fetchPosts: data.status === "success" 체크 추가 |
| 12 | `bts-event/components/Board/FanBoard.tsx` | openDetail: data.status === "success" 체크 추가 |

**feature 브랜치에 이미 적용된 항목 (수정 불필요):**
- SecurityConfig: /api/ai/stt, /api/ai/guest/**, /api/ai/interview/resume/upload → permitAll
- AiChatController: POST /api/ai/guest/chat 엔드포인트
- AiSttController: null-safe userDetails 처리
- AiInterviewController: null-safe userDetails 처리

---

## EC2 재배포 필요 사항

1. `feature/bts-event-complete` 브랜치로 Docker 이미지 빌드
2. EC2에서 새 컨테이너 배포 (기존 `sdui-backend-lab` 대체)
3. Flyway V35 자동 적용 (fan_board 테이블 + 쿼리 4개 생성)
4. Vercel bts-gwanghwamun 환경변수 확인: `NEXT_PUBLIC_API_BASE=https://yerin.duckdns.org` ✅

---

## 배포 현황 (확인 완료 2026-03-21)

| 항목 | 상태 |
|------|------|
| EC2 브랜치 | **main** (feature/bts-event-complete 미반영) |
| Flyway 적용 버전 | **V34까지 적용** (V35 미실행, 이전 실패는 PostgreSQL 트랜잭셔널 DDL로 롤백됨) |
| Vercel NEXT_PUBLIC_API_BASE | **https://yerin.duckdns.org** ✅ 정상 |

**결론**:
- 코드 수정 12건 완료 → feature 브랜치 커밋 후 EC2 재배포 필요
- V35 수정됨 → fan_board 테이블 정상 생성, SDUI GET_CONTENT_LIST_PAGE 안전
