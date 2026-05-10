# .ai 문서 마스터 인덱스

> 최종 수정: 2026-03-20 (콘텐츠 나만 보기 기능 추가)
> 이 파일은 `.ai` 폴더 내 모든 문서의 목적과 위치를 안내합니다.

---

## 역할별 핵심 문서

### Architect
| 파일 | 내용 |
|------|------|
| [research.md](architect/research.md) | 시스템 구조 분석, AI 파이프라인 기술 선택, SDUI 확장 설계 (2026-02-28 ~ 03-11) |
| [plan.md](architect/plan.md) | 전체 구현 계획 (Phase별 로드맵) |
| [plan_admin_permission_page.md](architect/plan_admin_permission_page.md) | 어드민 권한 페이지 설계 |
| [diary_to_content_migration_plan.md](architect/diary_to_content_migration_plan.md) | 다이어리→콘텐츠 마이그레이션 |
| [mobile_plan_hybrid.md](architect/mobile_plan_hybrid.md) | 모바일 하이브리드 방식 계획 (**정본**) |
| [mobile_plan_parallel.md](architect/mobile_plan_parallel.md) | 모바일 병렬 방식 계획 (**정본**) |

### Backend Engineer
| 파일 | 내용 |
|------|------|
| [research.md](backend_engineer/research.md) | Spring Boot 도메인 분석, AI RestClient 패턴, 멤버십 스키마, 카카오 알림 (2026-02-28 ~ 03-18) |
| [plan.md](backend_engineer/plan.md) | 백엔드 구현 계획 |
| [kakao_notification.md](backend_engineer/kakao_notification.md) | 카카오톡 약속 알림 설계 및 구현 |
| [slack_notification.md](backend_engineer/slack_notification.md) | Slack 웹훅 알림 모듈 — 현재 구현 + 4단계 로드맵 (Phase 0 완료) |
| [plans/](backend_engineer/plans/) | 개별 기능 계획서 (타임아웃 수정, Flyway 검증 등) |
| [reports/](backend_engineer/reports/) | 배포 트러블슈팅, Flyway 체크섬 수정 가이드 |

### Frontend Engineer
| 파일 | 내용 |
|------|------|
| [research.md](frontend_engineer/research.md) | SDUI 엔진 분석, AI 컴포넌트 설계, 벤토 그리드, 모바일 CSS (2026-02-28 ~ 03-18) |
| [plan.md](frontend_engineer/plan.md) | 프론트엔드 구현 계획 |
| [product_main_plan.md](frontend_engineer/product_main_plan.md) | 메인 페이지 리디자인 계획 |
| [error_handling_guide.md](frontend_engineer/error_handling_guide.md) | 프론트엔드 에러 처리 가이드 |
| [reports/2026-03-07_main_page_bento_grid_implementation.md](frontend_engineer/reports/2026-03-07_main_page_bento_grid_implementation.md) | 벤토 그리드 구현 보고서 |

### Designer
| 파일 | 내용 |
|------|------|
| [research.md](designer/research.md) | UI/UX 분석 |
| [plan.md](designer/plan.md) | 디자인 계획 |

### Planner
| 파일 | 내용 |
|------|------|
| [research.md](planner/research.md) | 기획 리서치 |
| [plan.md](planner/plan.md) | 기획 계획 |

### QA Engineer
| 파일 | 내용 |
|------|------|
| [research.md](qa_engineer/research.md) | 테스트 전략 |
| [plan.md](qa_engineer/plan.md) | 테스트 계획 |
| [testReport.md](qa_engineer/testReport.md) | 테스트 결과 보고서 |
| [rbac_manual_test.md](qa_engineer/rbac_manual_test.md) | RBAC 수동 테스트 가이드 |
| [march_2_*.md](qa_engineer/) | 2026-03-02 버그 수정 가이드 모음 |

---

## 기능별 문서

### Feature 폴더
| 폴더/파일 | 내용 |
|-----------|------|
| [feature/bts-event/plan.md](feature/bts-event/plan.md) | BTS 이벤트 페이지 기획 및 아키텍처 |
| [feature/bts-event/issues.md](feature/bts-event/issues.md) | **🔥 현재 이슈 목록 (2026-03-21)** — BTS페이지 7건 + SDUI 2건 |
| [feature/hybridMobile_Web/](feature/hybridMobile_Web/) | 모바일/웹 하이브리드 설계 원안 (정본 2개 파일) |
| [feature/addAdminPage/](feature/addAdminPage/) | 어드민 페이지 추가 작업 기록 |
| [feature/ai_pronunciation_redesign/](feature/ai_pronunciation_redesign/) | AI 발음 기능 리디자인 계획 |
| [feature/migration_flyway_issue/](feature/migration_flyway_issue/) | Flyway 이슈 해결 가이드 모음 |

---

## 운영/유지보수 문서

| 파일 | 내용 |
|------|------|
| [maintenance/local_docker_dev_guide.md](maintenance/local_docker_dev_guide.md) | 로컬 Docker 개발환경 가이드 |
| [maintenance/aws_environment_guide.md](maintenance/aws_environment_guide.md) | AWS EC2 환경 가이드 |
| [maintenance/deployment_checklist.md](maintenance/deployment_checklist.md) | 배포 체크리스트 |
| [maintenance/database_change_guide.md](maintenance/database_change_guide.md) | DB 변경 가이드 |
| [maintenance/troubleshooting_guide.md](maintenance/troubleshooting_guide.md) | 트러블슈팅 가이드 |
| [maintenance/SDUI_Debugging_Guide_0305.md](maintenance/SDUI_Debugging_Guide_0305.md) | SDUI 디버깅 가이드 (2026-03-05) |
| [maintenance/SDUI_HTTPS_Mixed_Content_Debugging.md](maintenance/SDUI_HTTPS_Mixed_Content_Debugging.md) | HTTPS Mixed Content 이슈 |
| [maintenance/kakao_login_error_history.md](maintenance/kakao_login_error_history.md) | 카카오 로그인 오류 이력 |
| [maintenance/vercel_image_troubleshooting.md](maintenance/vercel_image_troubleshooting.md) | Vercel 이미지 트러블슈팅 |

---

## 개발 일지 (assets/log/)

일자별 로그는 통합본으로 정리되었습니다:

| 파일 | 내용 | 기간 |
|------|------|------|
| [assets/log/devlog_AI_CHAT.md](../assets/log/devlog_AI_CHAT.md) | AI 채팅 개발 이력 (버그픽스, V2 런칭, 테스트) | 3/13 ~ 3/17 |
| [assets/log/devlog_AI_INTERVIEW.md](../assets/log/devlog_AI_INTERVIEW.md) | AI 면접관 개발 이력 (V1, 이미지/PDF, S3 이슈) | 3/15 ~ 3/17 |
| [assets/log/devlog_MIGRATION.md](../assets/log/devlog_MIGRATION.md) | Flyway 마이그레이션 이력 (Squash, 트러블슈팅) | 3/16 ~ 3/17 |

> 원본 일자별 폴더: `assets/log/3월13일/` ~ `assets/log/3월17일/`

---

## .ai2 폴더

2026-03-11 AI 기능 설계 세션에서 별도 생성. 내용은 `.ai` 각 role 폴더로 병합 완료.
자세한 내용: [`.ai2/README.md`](../.ai2/README.md)
