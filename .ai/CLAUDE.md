## Global Workflow Rules

Always do:

커밋 전 항상 테스트 실행  
스타일 가이드의 네이밍 컨벤션 항상 준수 (프론트: camelCase 변수/함수, PascalCase 컴포넌트 / 백엔드: camelCase 메서드, PascalCase 클래스)
오류는 항상 모니터링 서비스에 로깅 (백엔드: SLF4J logger / 프론트: 콘솔 대신 에러 추적 서비스)
Ask first:

데이터베이스 스키마 수정 전 (테이블 컬럼 추가/변경/삭제, 인덱스 변경 등)
새 의존성 추가 전  
CI/CD 설정 변경 전 (GitHub Actions, Dockerfile, docker-compose.yml 변경 등)
Never do:

시크릿이나 API 키 절대 커밋 금지 (.env, application-secret.yml, 하드코딩된 비밀번호/토큰 등)
-절대 편집 금지
명시적 승인 없이 실패하는 테스트 제거 금지
사용자 요청 없이 브라우저를 자동으로 실행하지 않음 (Never open the browser automatically)
폴더/파일 생성이 필요하다면 우선 .ai 폴더 안에서 생성 후 plan에 필요한 폴더 위치를 설명
Python 스크립트를 직접 실행하지 않음 — 코드 작성 후 사용자가 직접 실행 (Never run python scripts; the user runs them)
