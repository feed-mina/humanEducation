# FastAPI Contract Test 502 에러 및 Neo4j 로그 트러블슈팅

**작성일**: 2026-03-10
**대상**: 백엔드 API (FastAPI)

---

## 1. 502 Bad Gateway 에러 (`/api/artists`)

### 📌 현상
`pytest src/api/test_contract.py -v -m contract` 실행 시, 아래 두 테스트에서 502 에러가 발생했습니다.
- `test_artists_schema`
- `test_onboarding_movies_flow`

두 테스트 모두 FastAPI의 `GET /api/artists` 엔드포인트를 호출할 때 502(Bad Gateway) 응답을 받았습니다.

### 🔍 원인
`/api/artists` 엔드포인트는 내부적으로 Supabase 데이터베이스와 통신하여 데이터를 가져옵니다. 502 에러는 FastAPI 서버가 Supabase로 요청을 보냈으나, 인증(Key) 실패 또는 연결 오류로 정상적인 데이터를 받아오지 못했을 때 발생합니다.

`.env` 파일에 `SUPABASE_URL`과 `SUPABASE_KEY`를 업데이트했으나, **FastAPI 서버 구동 시 변경된 `.env` 내용이 반영되지 않았거나**, 입력된 키의 형태(따옴표, 공백 등) 문제일 수 있습니다.

### 💡 해결 방법
1. **`.env` 파일 형식 확인**
   - `.env` 파일에 값이 다음과 같이 공백이나 불필요한 따옴표 없이 입력되어 있는지 확인합니다.
   ```env
   SUPABASE_URL=https://zyixwzkbszsgiltatkog.supabase.co
   SUPABASE_KEY=sb_secret_5DkG10rNgJ6Hewaj1CsL0g_5mSAFAit
   ```
2. **FastAPI 서버 완전히 재시작하기 (필수)**
   - 터미널에서 구동 중인 `uvicorn src.api.fastapi_server:app --port 8000` 프로세스를 `Ctrl + C`를 눌러 완전히 종료합니다.
   - 다시 서버를 구동시켜 변경된 환경변수를 읽어오도록 합니다.
3. **네트워크 접근 확인**
   - 해당 키가 올바른 권한을 가진 키인지 확인하고, 계속 502가 발생할 경우 Supabase 대시보드 로그를 확인해야 합니다.

---

## 2. Neo4j DB 경고 로그 (Warning)

### 📌 현상
FastAPI 터미널 로그에 다음과 같은 `GqlStatusObject` 경고가 다수 출력되었습니다.
- `warn: property key does not exist. The property image_url does not exist...`
- `warn: relationship type does not exist. The relationship type IN_REGION does not exist...`

### 🔍 원인
FastAPI가 Neo4j 데이터베이스에 `image_url` 필드나 `IN_REGION` 관계선을 요구하는 쿼리를 보냈으나, **현재 데이터베이스에는 아직 그런 데이터가 적재되어 있지 않아** 나타나는 주의(Warning) 수준의 로그입니다. 

### 💡 해결 방법
**당장 조치할 필요는 없습니다.** Neo4j는 스키마가 없는(schema-less) 특성이 있어 데이터가 없으면 단순히 빈 데이터(Null)를 반환하고 정상 처리(200 OK)합니다. 추후 노드 마이그레이션 스크립트를 통해 전체 POI 노드와 Region 데이터를 Neo4j에 `Import(적재)`하고 나면 이 경고는 자동으로 모두 사라집니다.