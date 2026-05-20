# 커뮤니티 + 챗봇 통합 테스트 결과

> 실행일: 2026-05-20
> 브랜치: `refactor/krider_backup`

---

## 전체 요약

| Layer | Framework | Tests | Status | 실행 시간 |
|-------|-----------|-------|--------|-----------|
| Spring Boot (Backend) | JUnit 5 + Mockito | 19 | **ALL PASSED** | 2m 7s |
| Next.js (Frontend) | Jest + jest.mock | 9 | **ALL PASSED** | 64.9s |
| FastAPI (AI Server) | pytest + TestClient | 9 | **ALL PASSED** | 4.04s |
| **합계** | | **37** | **ALL PASSED** | |

---

## 1. Spring Boot — Unit Tests (JUnit 5 + Mockito)

### 실행 명령어

```bash
cd D:/kride-project/subproject/SDUI/SDUI-server
./gradlew test --tests 'com.domain.demo_backend.domain.community.*' --tests 'com.domain.demo_backend.domain.kridechat.*'
```

### 1.1 CommunityPostServiceTest (7 tests)

게시글 CRUD 서비스의 핵심 비즈니스 로직을 검증한다.

| # | Test Case | 검증 의미 |
|---|-----------|----------|
| 1 | 게시글 작성 — 이미지 없이 성공 | `PostCreateRequest` → `CommunityPost` 엔티티 변환 + DB 저장이 정상 동작하는지, 이미지 없을 때 `PostImageRepository.save()`가 호출되지 않는지 |
| 2 | 게시글 작성 — 존재하지 않는 사용자 | `userSqno`가 DB에 없을 때 `IllegalArgumentException("사용자를 찾을 수 없습니다.")` 발생 — 인증 토큰의 사용자가 실제 존재하는지 검증 |
| 3 | 게시글 목록 조회 — 페이징 | `Pageable` 기반 페이징이 정상 동작하고, `delYn='N'`인 게시글만 반환하는지 |
| 4 | 게시글 상세 조회 — 성공 | `@EntityGraph`로 author+images를 JOIN FETCH하여 N+1 없이 조회하는지 |
| 5 | 게시글 상세 조회 — 존재하지 않는 게시글 | 없는 `postId` 조회 시 `IllegalArgumentException("게시글을 찾을 수 없습니다.")` — 404 응답의 근거 |
| 6 | 게시글 수정 — 작성자 본인만 가능 | `authorSqno != requestUserSqno` 일 때 `IllegalArgumentException("수정 권한이 없습니다.")` — 타인 게시글 수정 차단 |
| 7 | 게시글 수정 — 작성자 본인 성공 | 본인이 수정 시 title/content가 정상 반영되는지 |
| 8 | 게시글 삭제 — soft delete 처리 | 삭제 시 `delYn`이 `'N'` → `'Y'`로 변경되는지 (물리 삭제가 아닌 논리 삭제) |
| 9 | 게시글 삭제 — 작성자 아닌 경우 예외 | 타인이 삭제 시도 시 `IllegalArgumentException("삭제 권한이 없습니다.")` — 권한 검증 |

### 1.2 PostLikeServiceTest (3 tests)

좋아요 토글 로직의 정합성을 검증한다.

| # | Test Case | 검증 의미 |
|---|-----------|----------|
| 1 | 좋아요 토글 — 좋아요 추가 | 기존 좋아요가 없을 때 `PostLike` 엔티티 생성 + `likeCount` 증가(5→6) |
| 2 | 좋아요 토글 — 좋아요 취소 | 기존 좋아요가 있을 때 `PostLike` 삭제 + `likeCount` 감소(5→4) |
| 3 | 좋아요 상태 조회 | `existsBy...` 쿼리로 현재 사용자의 좋아요 여부 + `countBy...`로 총 좋아요 수 반환 |

### 1.3 UserFollowServiceTest (4 tests)

팔로우 토글 + 자기 팔로우 방지 로직을 검증한다.

| # | Test Case | 검증 의미 |
|---|-----------|----------|
| 1 | 팔로우 토글 — 팔로우 추가 | 기존 팔로우 없을 때 `UserFollow` 생성 + `following=true` + 팔로워 수 반환 |
| 2 | 팔로우 토글 — 언팔로우 | 기존 팔로우 있을 때 `UserFollow` 삭제 + `following=false` |
| 3 | 자기 자신 팔로우 — 예외 | `followerSqno == followeeSqno` 일 때 `IllegalArgumentException("자기 자신을 팔로우할 수 없습니다.")` — 비즈니스 규칙 |
| 4 | 팔로우 상태 조회 | `following` 여부 + `followerCount` + `followingCount` 3개 값 정확히 반환 |

### 1.4 KrideChatServiceTest (5 tests)

챗봇 의도 분류 + FastAPI 프록시 + 장애 대응 로직을 검증한다.

| # | Test Case | 검증 의미 |
|---|-----------|----------|
| 1 | 의도 분류 — '추천' 키워드로 recommend 의도 인식 | `"맛집 추천해줘"` 메시지에서 `추천` 키워드 감지 → `intent=recommend` → `FastApiChatClient.recommendAi()` 호출 |
| 2 | 의도 분류 — '일정' 키워드로 itinerary 의도 인식 | `"2박3일 일정 짜줘"` 메시지에서 `일정` 키워드 감지 → `intent=itinerary` → `FastApiChatClient.generateItinerary()` 호출 |
| 3 | 의도 분류 — 일반 질문은 qa 의도 | 추천/일정 키워드가 없는 메시지 → `intent=qa` → PDF Q&A 경로 |
| 4 | 명시적 intent 지정 시 해당 intent 사용 | `request.setIntent("recommend")` 명시 → 키워드 분석 건너뛰고 바로 recommend 처리 |
| 5 | FastAPI 연결 실패 시 fallback 응답 | `Mono.error(RuntimeException)` → 예외 catch 후 `"실패"` 포함 fallback 메시지 반환 (서비스 중단 방지) |

---

## 2. Next.js — Unit Tests (Jest + jest.mock)

### 실행 명령어

```bash
cd D:/kride-project/subproject/SDUI/metadata-project
npx jest tests/services/communityService.test.ts
```

### communityService.test.ts (9 tests)

프론트엔드 API 서비스 레이어의 HTTP 호출 정합성을 검증한다. `axios`를 jest.mock으로 대체하여 실제 서버 없이 테스트.

| # | Test Case | 검증 의미 |
|---|-----------|----------|
| 1 | 페이지 파라미터로 게시글 목록을 조회해야 함 | `GET /api/v1/community/posts?page=0&size=10` 호출 + `ApiResponse<Page>` 구조 파싱 |
| 2 | 게시글 상세 정보를 조회해야 함 | `GET /api/v1/community/posts/{postId}` 호출 + `ApiResponse<PostResponse>` 구조 파싱 |
| 3 | FormData로 게시글을 생성해야 함 | `POST /api/v1/community/posts` + `Content-Type: multipart/form-data` 헤더 + `FormData` body 전송 |
| 4 | 게시글을 삭제해야 함 | `DELETE /api/v1/community/posts/{postId}` 호출 + `status: 'success'` 응답 확인 |
| 5 | 좋아요를 토글해야 함 | `POST /api/v1/community/posts/{postId}/likes` 호출 + `{ liked, likeCount }` 응답 파싱 |
| 6 | 좋아요 상태를 조회해야 함 | `GET /api/v1/community/posts/{postId}/likes/status` 호출 + `liked` boolean 반환 |
| 7 | 게시글을 신고해야 함 | `POST /api/v1/community/posts/{postId}/reports` + `{ reasonCode, detailText }` body 전송 |
| 8 | 팔로우를 토글해야 함 | `POST /api/v1/community/users/{userSqno}/follow` 호출 + `{ following, followerCount }` 응답 |
| 9 | 팔로우 상태를 조회해야 함 | `GET /api/v1/community/users/{userSqno}/follow/status` 호출 + `{ following, followerCount, followingCount }` |

**Configuration 검증 포인트:**
- 모든 API 경로가 `/api/v1/community/` prefix를 사용하는지 (Spring Boot SecurityConfig과 일치)
- `createPost`가 `multipart/form-data`로 전송하는지 (이미지 업로드 지원)
- `ApiResponse<T>` 래퍼(`{ status, data }`)에서 `data`만 추출하는지

---

## 3. FastAPI — Integration Tests (pytest + TestClient)

### 실행 명령어

```bash
cd D:/kride-project
pytest tests/test_community_chatbot_integration.py -v
```

### Configuration: 외부 의존성 Stub

테스트 실행 전 아래 모듈을 stub/mock으로 대체하여 외부 서비스 없이 테스트 가능하게 구성:

| Stub 대상 | 이유 |
|-----------|------|
| `neo4j`, `chromadb`, `groq`, `supabase` | 외부 DB/API 클라이언트 — 로컬에 미설치 가능 |
| `sentence_transformers` | 임베딩 모델 (800MB+) — CI 환경에 미설치 |
| `lightgbm`, `sklearn` | ML 모델 역직렬화 시 필요 — stub으로 로드 차단 |
| `src.api.ensemble_client` | `pickle.load(ensemble_ranker.pkl)` 모델 로드 → hang 방지 |
| `src.ml.feature_engineering` | numpy 기반 feature 추출 — 모듈 자체를 MagicMock 대체 |

### test_community_chatbot_integration.py (9 tests)

FastAPI 엔드포인트의 요청/응답 구조와 에러 핸들링을 검증한다.

| # | Test Case | 검증 의미 |
|---|-----------|----------|
| 1 | AI 추천 503 when no AI | `HAS_AI=False` (Neo4j/ChromaDB 미연결) → 503 Service Unavailable 반환 — graceful degradation |
| 2 | AI 추천 returns pois and text | `HAS_AI=True` → `{ pois, recommendation_text, count }` 응답 구조 + count >= 1 |
| 3 | AI 추천 empty request | 빈 artists/regions/purposes → 에러 없이 `count=0` 반환 — null-safe 처리 |
| 4 | AI 추천 deduplication | Neo4j와 ChromaDB에서 동일 `poi_id` 반환 시 중복 제거 — `len(poi_ids) == len(set(poi_ids))` |
| 5 | Itinerary 503 when no AI | `HAS_AI=False` → 503 반환 — AI 서비스 미가동 시 명확한 에러 |
| 6 | Itinerary returns structure | 일정 생성 결과에 `itinerary` 또는 `source_pois` 키 존재 — 프론트엔드 계약 준수 |
| 7 | Artists returns list | `GET /api/artists` → `{ "artists": [...] }` 구조로 반환 — 프론트엔드 계약 |
| 8 | Regions returns list | `GET /api/regions` → `{ "regions": [...] }` 구조 + 2개 이상 반환 |
| 9 | Health OK | `GET /api/health` → `{ "status": "ok" }` — 서버 가동 상태 확인 |

---

## 테스트 파일 위치

| Layer | 파일 경로 |
|-------|----------|
| Spring Boot | `SDUI-server/src/test/java/com/domain/demo_backend/domain/community/service/CommunityPostServiceTest.java` |
| Spring Boot | `SDUI-server/src/test/java/com/domain/demo_backend/domain/community/service/PostLikeServiceTest.java` |
| Spring Boot | `SDUI-server/src/test/java/com/domain/demo_backend/domain/community/service/UserFollowServiceTest.java` |
| Spring Boot | `SDUI-server/src/test/java/com/domain/demo_backend/domain/kridechat/KrideChatServiceTest.java` |
| Next.js | `metadata-project/tests/services/communityService.test.ts` |
| FastAPI | `tests/test_community_chatbot_integration.py` |

---

## pytest 수정 이력

### 문제: 테스트 실행 시 무한 대기 (hang)

**원인 체인:**
```
pytest collect → import test file
  → import fastapi_server.py
    → import ensemble_client.py (line 55)
      → import numpy
      → import feature_engineering.py
      → pickle.load("models/ensemble_ranker.pkl")
        → lightgbm/sklearn 모델 역직렬화 → hang
```

**해결:** `sys.modules`에 가짜 모듈을 사전 등록하여 실제 import/pickle.load 차단

| 변경 전 (hang) | 변경 후 (4초 통과) |
|----------------|-------------------|
| stub 5개: neo4j, chromadb, groq, supabase, sentence_transformers | stub 8개: + lightgbm, sklearn, sklearn.model_selection |
| ensemble_client가 실제 import됨 → pickle.load 발생 | `src.api.ensemble_client` 모듈 자체를 MagicMock 대체 |
| feature_engineering이 실제 import됨 | `src.ml.feature_engineering` 모듈을 MagicMock 대체 |
| `duration: 2` (int) → 422 Unprocessable Entity | `duration: "1박2일"` (string) — Pydantic 스키마 일치 |
| `assert isinstance(body, list) or "data" in body` → False | `assert "artists" in body` — 실제 응답 구조 매칭 |
