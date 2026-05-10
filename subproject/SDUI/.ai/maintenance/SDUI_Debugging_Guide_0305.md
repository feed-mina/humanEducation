# SDUI 하얀 화면 에러 디버깅 문서 (White Screen Troubleshooting)

Vercel에서 배포된 프론트엔드 화면이 잠시 렌더링 된 후 하얗게 변하는 현상의 정확한 원인을 파악하고 해결하기 위한 가이드입니다.
문제를 **TodoList(방향성) -> Research(진단) -> Solution(해결)** 의 3단계로 나누어 접근합니다.

---

## 📋 1. TodoList (디버깅 방향성)

어디에서 병목이나 에러가 발생했는지 구간별로 나누어서 추적합니다.

- [X] Create step-by-step diagnostic plan
- [X] Check AWS EC2 connectivity (User confirmed: `ERR_CONNECTION_TIMED_OUT` and Instance Status Check Failed)
- [X] Recover the AWS EC2 instance (Reboot/Stop & Start)
- [X] Verify Docker container status after recovery

---

## 🔍 2. Research (진단 방법 및 수행 항목)

각 구간별로 어떤 명령어나 확인 절차가 필요한지 정의합니다.

- **[진단 완료]** AWS EC2 인프라 장애 (인스턴스 상태 검사 1/2 통과 실패)
- Vercel 프론트엔드가 하얗게 변한 이유는 백엔드 서버 자체가 완전히 죽어서(네트워크 단절) 응답을 주지 못했기 때문입니다.

### 구간 A: Vercel 프론트엔드 진단

1. **브라우저 개발자 도구 확인**:
    - Vercel URL 접속 후 키보드 `F12`를 눌러 **콘솔(Console)** 창에 빨간색 에러(CORS, Mixed Content 등)가 뜨는지 확인합니다.
    - **네트워크(Network)** 탭에서 `/api/ui/...` 통신이 `Failed` 되었거나 HTTP 상태코드 `(canceled)`인지 확인합니다.

### 구간 B: AWS EC2 (인프라) 진단

1. **서버 핑(Ping) 테스트**: AWS 서버 컴퓨터(`43.201.237.68`)가 켜져 있는지 확인합니다.
2. **Postman 또는 터미널 cURL 테스트**: 브라우저를 거치지 않고 직접 API를 호출했을 때 응답이 오는지 확인합니다.
    - 명세 예시: `http://43.201.237.68/api/ui/...`
3. **AWS 보안 그룹(Security Group) 확인**: 80번 포트(HTTP)나 8080번 포트가 Vercel 쪽(또는 외부망 0.0.0.0/0)에 열려 있는지 확인합니다.

### 구간 C: Docker 백엔드 진단 (AWS 접속 필요)

1. **터미널(SSH) 접속**: `.pem` 키를 이용해 AWS EC2 리눅스 환경으로 접속합니다.
2. **도커 컨테이너 생존 확인**:
    - 명령어: `docker ps`
    - `sdui-backend` (Spring Boot), `sdui-db` (PostgreSQL), `sdui-redis`가 모두 포트를 점유하고 `Up` 상태인지 확인합니다.
3. **스프링 부트 에러 로그 확인**:
    - 명령어: `docker logs sdui-backend --tail 50`
    - Vercel의 데이터를 페칭할 때 DB 커넥션 예외 등 백엔드 런타임 에러가 발생했는지 확인합니다.

---

## 💡 3. Solution (예상 해결 방안)

진단(Research) 결과에 따라 아래의 솔루션 중 하나를 적용합니다.

### 진단 결과 1: Vercel의 브라우저 Mixed Content 차단 (가장 유력)

* **원인**: Vercel은 `https`(암호화 O)인데, AWS 백엔드는 `http`(암호화 X)라서 브라우저가 통신을 막음.
* **해결책**: 가비아 등에서 저렴한 도메인(`.shop` 등)을 구입하여 AWS EC2에 연결하고, Nginx를 설치해 **Let's Encrypt 무료 인증서(https)**를 적용해 준 뒤 프론트엔드 URL을 교체한다.

### 진단 결과 2: AWS 보안그룹 또는 Docker 포트 차단

* **원인**: AWS 방화벽이 80 또는 8080 포트를 막고 있거나, 서버 인스턴스가 재부팅되면서 도커 컨테이너가 죽어 있음.
* **해결책**: AWS 콘솔에서 인바운드 보안 규칙을 열어주거나, SSH로 접속해 `docker-compose up -d`로 백엔드를 다시 깨운다.

### 진단 결과 3: 백엔드 DB/서버 런타임 에러

* **원인**: 백엔드는 API 요청을 받았으나, 쿼리 수행 중 예외 발생 (Postgres 연결 실패 등).
* **해결책**: `docker logs` 오류 메시지를 기반으로 Spring Boot 코드를 수정하여 다시 빌드 & 배포한다.