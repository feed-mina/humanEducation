# Flyway 마이그레이션 개발 로그 (통합본)

> 원본: `3월16일/Migration_Troubleshooting.md`, `3월17일/03월17일_MIGRATION_SQUASH.md`
> 마지막 수정: 2026-03-19

---

## 마이그레이션 버전 전체 이력

| 버전 | 파일명 | 내용 | 상태 |
|------|--------|------|------|
| V1~V7 | (초기) | 기본 테이블 스키마 | ✅ 완료 |
| V8 | `V8__main_page_bento_grid.sql` | MAIN_PAGE 벤토 그리드 전환 | ✅ 완료 |
| V9 | `V9__user_card_label.sql` | USER 카드 라벨 변경 | ✅ 완료 |
| V10 | `V10__bento_clickable.sql` | 벤토 카드 전체 클릭 가능 | ✅ 완료 |
| V11~V21 | (다수) | 관리자 대시보드, AI 채팅/인터뷰 페이지 | ✅ Squash 전 완료 |
| **V22** | `V22__seed_ui_metadata.sql` | 시퀀스 리셋 포함 전체 UI 시드 | ✅ 완료 |
| **V23** | `V23__create_tables_and_queries.sql` | 테이블 + 쿼리 (squash) | ✅ 완료 |
| **V24** | `V24__ai_chat_pages.sql` | AI 채팅 페이지 (squash) | ✅ 완료 |
| **V25** | `V25__interview_and_japanese.sql` | 면접 + 일본어 채팅 (squash) | ✅ 완료 |
| **V26** | `V26__interview_resume_and_tutorial.sql` | interview_resume 테이블 + 튜토리얼→AI 전환 | ✅ 완료 |
| **V27** | `V27__add_kakao_tokens_and_notif_flags.sql` | 카카오 토큰 + 알림 플래그 | ✅ 브랜치 적용, 로컬/Docker 미적용 |

---

## 2026-03-16 — 마이그레이션 트러블슈팅

### 이슈: V34 system_prompt_template 컬럼 누락

**증상**: Spring Boot 시작 시 `column "system_prompt_template" does not exist` 오류
**원인**: AI 채팅 시스템 프롬프트 외부화를 위한 컬럼을 참조 코드는 있으나 마이그레이션 누락
**수정**: `V34__add_system_prompt_column.sql` 신규 추가
```sql
ALTER TABLE query_master
  ADD COLUMN IF NOT EXISTS system_prompt_template TEXT;
```

### 이슈: V34/V35 번호 충돌

**증상**: AI_JAPANESE_CHAT_PAGE 마이그레이션이 이미 V34를 사용 중
**수정**: 일본어 채팅 마이그레이션을 V35로 번호 이동
```
V34__add_system_prompt_column.sql   (신규 추가)
V35__ai_japanese_chat_page.sql      (V34 → V35 재번호)
```

---

## 2026-03-17 — V11~V40 Squash (AWS 배포 전)

### Squash 배경

AWS EC2 배포 전 마이그레이션 파일이 18개(V11~V40+)로 증가 → 관리 복잡도 증가
V11~V21을 V22~V26으로 통합 squash 실행.

### Squash 결과 (V11~V40 → V22~V26)

| 새 버전 | 통합 내용 | 원본 버전 |
|---------|-----------|-----------|
| V22 | `ui_metadata` 전체 시드 + 시퀀스 리셋 | V11~V21 + 시퀀스 fix |
| V23 | 신규 테이블 (`memberships`, `user_memberships` 등) + query_master | V28~V30 |
| V24 | AI 채팅 페이지 (`AI_ENGLISH_CHAT_PAGE`, `AI_KOREAN_CHAT_PAGE`) | V31~V33 |
| V25 | AI 면접 + 일본어 채팅 + 시스템 프롬프트 컬럼 | V34~V36 |
| V26 | `interview_resume` 테이블 + 튜토리얼→AI 영어 채팅 전환 | V37~V40 |

### V22 시퀀스 리셋 (중요)

Docker DB가 V10까지만 적용된 상태에서 V22 실행 시 PK 충돌 발생:
- **원인**: V1~V10이 명시적 `ui_id` 없이 INSERT → 시퀀스가 이미 존재하는 ID 생성
- **수정**: `V22__seed_ui_metadata.sql` ALTER TABLE 직후에 시퀀스 리셋 추가

```sql
SELECT setval(
  pg_get_serial_sequence('ui_metadata', 'ui_id'),
  COALESCE((SELECT MAX(ui_id) FROM ui_metadata), 0)
);
```

### Flyway 체크섬 불일치 복구 방법

```bash
# 증상: FlywayValidateException: Migration checksum mismatch for migration version N
# 원인: EC2 DB에 이미 적용된 파일을 로컬에서 수정

# 복구 (checksum을 로컬에서 재계산한 값으로 업데이트)
docker exec sdui-db psql -U mina -d SDUI_TD -c \
  "UPDATE flyway_schema_history SET checksum = <resolved_value> WHERE version = 'N';"
```

**재발 방지**: 커밋된 Flyway 파일 절대 수정 금지 → 변경 시 새 버전(V28...)으로 추가

---

## 마이그레이션 재실행 방법

```sql
-- 특정 버전 재실행 (실패한 경우)
DELETE FROM flyway_schema_history WHERE version = 'N';
-- Spring Boot 재시작 → Flyway가 해당 버전 재실행
```

```sql
-- 실패한 마이그레이션 확인
SELECT version, description, success, installed_on
FROM flyway_schema_history
ORDER BY installed_rank DESC
LIMIT 10;
```

---

## 주의사항

### label_text NOT NULL 제약

- 로컬/Docker 모두 `label_text` 컬럼 `NOT NULL`
- GROUP, TIME_RECORD_WIDGET 등 label_text 없는 행에도 반드시 빈 문자열 명시:
```sql
INSERT INTO ui_metadata (..., label_text, ...) VALUES (..., '', ...);
```

### Docker DB 초기화 시

```bash
# 기존 컨테이너 제거 후 재시작
docker-compose down -v   # 볼륨까지 삭제
docker-compose up -d
# → Spring Boot bootRun 시 V1부터 전체 마이그레이션 자동 실행
```
