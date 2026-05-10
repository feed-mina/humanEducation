# DB 마이그레이션 Squash 작업 기록
날짜: 2026-03-17

## 배경
- main 브랜치: V1~V21 (AWS SDUI_TD에 적용 완료)
- feature/addAIChore: V1~V40 (V22~V40은 AWS 미적용)
- merge 전 V22~V40을 V22~V26으로 squash하여 정리

## Squash 결과

| 이전 | 새 파일 | 통합 내용 |
|------|---------|---------|
| V22 (유지) | V22__seed_ui_metadata.sql | 기존 그대로 (3228줄, 핵심 화면 시드) |
| V23~V25, V28, V29 | **V23__setup_tables_and_queries.sql** | memberships/user_memberships 테이블 + 핵심 query_master + MAIN_PAGE 수정 |
| V26~V32 | **V24__add_ai_chat_pages.sql** | AI_ENGLISH_CHAT_PAGE(V2 최종) + AI_KOREAN_CHAT_PAGE |
| V33~V39 | **V25__add_ai_interview_and_japanese_chat.sql** | system_prompt_template 컬럼 + AI_INTERVIEW_PAGE + AI_JAPANESE_CHAT_PAGE |
| V40 | **V26__create_interview_resume_table.sql** | interview_resume 테이블 |

**삭제된 파일**: V23~V40 원본 18개

## 주요 squash 포인트

### AI_ENGLISH_CHAT_PAGE (V24)
- 구 V26(V1 등록) → V32(V2로 교체) 과정을 생략
- 처음부터 V2 최종 상태로 직접 등록
- component_id: `ai_en2_root`, `ai_en2_config`, `ai_en2_chat`
- data_sql_key: `AI_ENGLISH_CHAT_CONFIG_V2`
- `AI_ENGLISH_CHAT_CONFIG` (V1 orphaned key)는 생략

### AI_JAPANESE_CHAT_PAGE (V25)
- 구 V35(등록) + V36(is_readonly=false) + V39(프롬프트 강화) 합침
- 처음부터 강화 프롬프트 + is_readonly=false로 등록

### AI_INTERVIEW_PAGE (V25)
- 구 V33(English) + V37(한국어 전환) + V38(label_text 한국어) 합침
- 처음부터 한국어 최종 설정으로 등록

## 로컬 DB 초기화
```sql
-- PG18 (포트 5432) 기준으로 실행
DROP DATABASE IF EXISTS "SDUI_TD";
CREATE DATABASE "SDUI_TD";
-- → Spring Boot 재시작 시 Flyway V1~V26 순서대로 실행됨
```
실행 완료: `psql -U postgres -p 5432`

## 다음 단계

### 1. Git 작업 
```bash
cd SDUI-server/src/main/resources/db/migration

# 삭제된 파일 반영
git rm V23__add_goal_query_master.sql
git rm V24__fix_main_page_duplicate_diary_card.sql
git rm V25__seed_query_master_core.sql
git rm V26__ai_chat_pages.sql
git rm V27__fix_ai_chat_readonly.sql
git rm V28__create_memberships.sql
git rm V29__create_user_memberships.sql
git rm V30__ai_chat_page2.sql
git rm V31__fix_ai_chat_v2_readonly.sql
git rm V32__upgrade_ai_chat_v2.sql
git rm V33__add_ai_interview_page.sql
git rm V34__add_system_prompt_column.sql
git rm V35__add_ai_japanese_chat_page.sql
git rm V36__enable_japanese_chat.sql
git rm V37__update_interview_config_to_korean.sql
git rm V38__update_interview_label_to_korean.sql
git rm V39__strengthen_japanese_chat_prompt.sql
git rm V40__create_interview_resume_table.sql

# 새 파일 추가
git add V23__setup_tables_and_queries.sql
git add V24__add_ai_chat_pages.sql
git add V25__add_ai_interview_and_japanese_chat.sql
git add V26__create_interview_resume_table.sql

git commit -m "refactor: DB 마이그레이션 squash (V23~V40 → V23~V26)"
git push --force-with-lease origin feature/addAIChore
```

### 2. 로컬 검증
- VS Code Task: `Backend: bootRun (local)` 실행
- Flyway V1~V26 정상 실행 확인
- AI_INTERVIEW_PAGE, AI_JAPANESE_CHAT_PAGE, AI_ENGLISH_CHAT_PAGE 동작 확인

### 3. main merge
- feature/addAIChore → main PR/merge
- GitHub Actions가 AWS SDUI_TD에 V22~V26 자동 적용

## AWS 배포 후 DB 상태
- AWS SDUI_TD: V1~V26 (기존 V1~V21 + 신규 V22~V26)
- AWS SDUI_LAB: 별도 (lab/claude-dev 브랜치, 유지)
