# Phase 1 패치 가이드

Phase 1 멤버십 도메인을 위해 **기존 파일**에 적용해야 할 변경사항입니다.
새 파일들(`domain/membership/` 패키지 전체)은 `.ai2/backend_engineer/impl/phase1/` 에 준비되어 있습니다.

---

## 1. SecurityConfig.java 수정

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/config/SecurityConfig.java`

`anyRequest().denyAll()` 바로 위에 아래 규칙들을 추가합니다.

```java
// 기존 (line 88~92)
.requestMatchers("/api/auth/editPassword", ...).authenticated()
.requestMatchers("/api/content/**").authenticated()
// ADMIN ONLY
.requestMatchers("/api/admin/**").hasRole("ADMIN")

// ✅ 추가: 멤버십 API
.requestMatchers(HttpMethod.GET,  "/api/v1/memberships").permitAll()         // 플랜 목록은 공개
.requestMatchers(HttpMethod.POST, "/api/v1/memberships").hasRole("ADMIN")    // 생성은 어드민
.requestMatchers(HttpMethod.DELETE, "/api/v1/memberships/**").hasRole("ADMIN") // 삭제는 어드민
.requestMatchers("/api/v1/user-memberships/**").authenticated()               // 사용자 멤버십은 인증 필수

// ✅ 추가 (Phase 2): AI API
.requestMatchers(HttpMethod.POST, "/api/ai/**").authenticated()

// DEFAULT — 명시되지 않은 모든 요청 차단
.anyRequest().denyAll()
```

> **주의**: `PATCH`/`PUT`이 CORS allowedMethods에 없으므로 `hasRole("ADMIN")` 각 메서드별 선언 필요.

---

## 2. SQL 마이그레이션 파일 배치

```
SDUI-server/src/main/resources/db/migration/
├── V22__create_memberships.sql        ← impl/phase1/V22__create_memberships.sql
└── V23__create_user_memberships.sql   ← impl/phase1/V23__create_user_memberships.sql
```

서버 재기동 시 Flyway가 자동으로 V22, V23 마이그레이션을 실행합니다.

---

## 3. 새 파일 배치 위치 요약

```
SDUI-server/src/main/java/com/domain/demo_backend/
└── domain/membership/
    ├── domain/
    │   ├── Membership.java            ← impl/phase1/domain/Membership.java
    │   ├── MembershipRepository.java  ← impl/phase1/domain/MembershipRepository.java
    │   ├── UserMembership.java        ← impl/phase1/domain/UserMembership.java
    │   └── UserMembershipRepository.java ← impl/phase1/domain/UserMembershipRepository.java
    ├── dto/
    │   ├── MembershipRequest.java     ← impl/phase1/dto/MembershipRequest.java
    │   ├── MembershipResponse.java    ← impl/phase1/dto/MembershipResponse.java
    │   ├── UserMembershipRequest.java ← impl/phase1/dto/UserMembershipRequest.java
    │   └── UserMembershipResponse.java← impl/phase1/dto/UserMembershipResponse.java
    ├── service/
    │   ├── MembershipService.java     ← impl/phase1/service/MembershipService.java
    │   └── UserMembershipService.java ← impl/phase1/service/UserMembershipService.java
    └── controller/
        ├── MembershipController.java  ← impl/phase1/controller/MembershipController.java
        └── UserMembershipController.java ← impl/phase1/controller/UserMembershipController.java
```

---

## 4. API 엔드포인트 요약

| 메서드 | 경로 | 인증 | 설명 |
|--------|------|------|------|
| `GET`    | `/api/v1/memberships`              | 공개    | 멤버십 플랜 목록 조회 |
| `POST`   | `/api/v1/memberships`              | ADMIN   | 멤버십 플랜 생성 |
| `DELETE` | `/api/v1/memberships/{id}`         | ADMIN   | 멤버십 플랜 삭제 |
| `GET`    | `/api/v1/user-memberships/current` | 인증    | 현재 사용자 활성 멤버십 조회 |
| `POST`   | `/api/v1/user-memberships`         | 인증    | 사용자 멤버십 부여 |

---

## 5. Phase 4 멤버십 게이팅 (향후)

`UserMembershipService`에 `canConverse(Long userId)`, `canLearn()`, `canAnalyze()` 이미 구현.

AI 컨트롤러에 추가 시:
```java
// AiChatController.java 예시
Long userId = userDetails.getUserSqno();
if (!userMembershipService.canConverse(userId)) {
    emitter.send(SseEmitter.event().data("{\"error\":\"멤버십이 필요합니다.\"}"));
    emitter.complete();
    return emitter;
}
```
