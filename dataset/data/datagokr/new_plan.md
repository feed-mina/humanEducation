# K-Ride 2.0 — 한국 관광 AI 플랫폼 마스터 플랜

> **최종 업데이트**: 2026-04-28 (LLM/임베딩/Reranker/멀티모달 모델 선택 전략 + Chunking/Retrieval/Reranking 파이프라인 계획 추가)
> **프로젝트명**: K-Ride 2.0 (Korea Ride & Discover)
> **슬로건**: "한국을 발견하다 — AI가 만드는 나만의 여행 코스"

---

## 프로젝트 기획
```
* 프로젝트 계획은 바꾸려고 합니다. 

기획의도는 한국인/외국인 기반으로 새로운 관광 서비스 제공하는 서비스를 제공하려고 합니다. 
대상 : 한국인, K한류/Kpop을 좋아하는 외국인
목적 : (1)한국 여행을 하면서 맛집정보 (유튜브 또간집, 블루리본, 방송에서 나온 맛집 등) 와 (2-1)자신이 좋아하는 아이돌/가수/배우 등이 뮤직비디오, 영화, 드라마 촬영지나 SNS에 올린 장소나 팬덤이 방문하고 싶은 장소 정보 (2-2) 요즘 떠오르고 있는 케데헌/한국 문화를 알고싶어하는 사람들을 위한 국립중앙박물관 (서울, 경주, 기타 등등) 과 유명 관광지 (2-3) 자연을 좋아하는 사람이라면 자연치화적이면 아마 전국에서 제공하는 지역마다 스탬프를 찍을 수 있는 둘레길이나 관광경로 정보 (4) 당일치기, 1박2일, 2박3일, 3박 4일 정도로 추천해주는 관광지 경로로 이동할때 들어가는 예측 소비,  (5) 이동수단은 걷기, 자전거, 오토바이/자동차, 기차여행 등이 있습니다. 

(6)
공공데이터나 사용가능한 api / 크롤링도 한가지 방법이 될수있음 을 통해 학습데이터를 찾고 로컬에서 ml,dl,llm 등 모델링 
(7) ollma를 통한 모델 
(8) groq, huggingface, roboflow, kaggle 에서도 필요한 데이터(원천 또는 api )를 받아서 사용할수 있음 
(9) rag, graphrag와 mcp까지 확대 예정 
(10) 저비용을 위한 원천학습  모델링 생성> 오픈소스 모델링 사용 > 배포를 위해 최대한 경량화  (허깅페이스, 로보플로우, 캐글, groq등 외부에 올릴 수 있다면 올려서 사용)
(11) 구글 드라이브에서 gpu 사용이나 클라우드 배포까지 

현재 학습하고 모델링 되어있는 부분에서 전국으로 범위를 확대하고 지도 기반의 새로운 비즈니스 를 만들어보려고 합니다. 이동수단에 필요한 학습 부분은 시간이 오래 걸릴꺼 같아 1차 MVP 이후에 2차나 3차 MVP 에 붙이고 싶습니다 

기획의도에 맞춰서 현재 상태와 plan, research를 전부 읽고 새로운 new_plan, new_research 문서를 만들어주세요. 


```

## 1. 기획 의도

### 대상
- **한국인**: 국내 여행·맛집·둘레길·문화 관광 수요
- **외국인**: K-Pop/K-Drama/K-Culture 팬덤, 한국 여행 계획자

### 서비스 핵심 기능 (5개 카테고리)

| # | 카테고리 | 내용 | 데이터 소스 |
|---|---------|------|-----------|
| 1 | 🍽️ **맛집 정보** | 유튜브 또간집·블루리본·방송 맛집 | 소상공인 상가 ZIP(주력), TourAPI, 크롤링, 공공데이터 |
| 2-1 | 🎤 **K-Culture 성지순례** | MV/드라마/영화 촬영지, SNS 핫플, 팬덤 방문지 | 크롤링, 커뮤니티, 위키 |
| 2-2 | 🏛️ **문화·관광지** | 국립중앙박물관(서울·경주 등), 유명 관광지 | TourAPI(15,905건 보유), 공공데이터 |
| 2-3 | 🌿 **자연·둘레길** | 전국 둘레길, 스탬프 코스, 자연 관광 경로 | 공공데이터(두루누비), 산림청 |
| 3 | 💰 **일정별 소비 예측** | 당일~3박4일 코스별 예상 비용 | AI Hub 여행로그(보유), TabNet v2 |

### 이동수단
- **1차 MVP**: 걷기 + 자전거 (기존 자산 활용)
- **2차 MVP**: 대중교통 (지하철/버스 API)
- **3차 MVP**: 자동차/기차 (네이버 길찾기 API, 코레일 API)

---

## 2. 기존 자산 재활용 현황

### ✅ 전국 데이터 완료 — 즉시 활용 가능

| 자산 | 파일 | 활용 방법 | 시각화 |
|------|------|----------|:---:|
| 전국 관광 POI 15,905건 | `tour_poi.csv` | poi 테이블 | ✅ DB 로드 완료 |
| **WeatherLSTM** | `weather_lstm.pt` | 날씨 예보 Acc=82.16% (전국 67개 관측소) | ✅ 완료 |
| 전국 자전거도로 | `road_clean_nationwide.csv` | trail 테이블 (20,262행, 16개 시도) | ✅ DB 로드 완료 |
| **안전 RF 모델** | `safety_regressor.pkl` / `safety_classifier.pkl` | R²=0.9919 / F1=0.9990 (전국 도로+서울 사고) | ✅ 학습 완료 |
| 날씨 데이터 | `weather_asos_daily_nationwide.csv` | weather_daily 테이블 (73,426행) | ✅ DB 로드 완료 |
| 시설 데이터 | `facility_clean.csv` | poi 테이블 (3,368행) | ✅ DB 로드 완료 |
| 카카오 로컬 API 스크립트 | `step3_food_collect.py` | 맛집/카페 POI 수집 | — |
| 네이버 DataLab 스크립트 | `step4_naver_trend.py` | SNS 인기도 수집 | — |
| FastAPI 서버 코드 | `fastapi_server.py` (27KB) | API 서버 뼈대 | — |

### ⚠️ 수집/재학습 필요 — 전국 데이터 확보 후 활용

| 자산 | 현재 상태 | 필요 작업 | 시각화 |
|------|----------|----------|:---:|
| **안전 RF 모델** | 전국 도로 + **전국 사고 381 시군구** | 전국 사고 수집 완료 → 재학습 완료 | ✅ 재학습 완료 (89.1% 매핑, 전국) |
| Co-occurrence POI | ⚠️ **수도권 한정** (Recall@5=0.1372) — AI Hub 21,384행 기반 | 전국 AI Hub 신청 후 실데이터 재학습 (평균값 대체 금지) | ⏳ 재학습 후 |
| 소비 TabNet v2 | ⚠️ **수도권 한정** (R²=0.1277) — AI Hub 2,508행 기반 | 전국 AI Hub 신청 후 실데이터 재학습 (평균값 대체 금지) | ⏳ 재학습 후 |
| POI 매력도 TabNet | ⚠️ **수도권 한정** (R²=0.0662) — AI Hub 기반 | 전국 AI Hub 신청 후 실데이터 재학습 (평균값 대체 금지) | ⏳ 재학습 후 |
| 경로 그래프 | LFS 포인터 (빈 파일) | 전국 확대 재빌드 | ❌ |

### ❌ 불필요 / 보류

| 자산 | 이유 |
|------|------|
| Visit GRU 시퀀스 | 폐기 — LLM 코스 생성으로 대체 |
| Phase 7 셀카 패션 추천 | 프로젝트 방향 불일치 |
| 자전거 전용 프로파일 (BMI 등) | 2차 MVP로 이동 |

---

## 3. MVP 로드맵

### 1차 MVP — "지도 기반 관광 추천" (4주)

```
[Week 1] 데이터 수집 + 전처리 + DB 로드 (✅ 로컬 PG 전환 + 전체 로드 완료)
  ✅ 완료: 날씨, 자전거도로, 사고다발지, 안전 모델 전국 재학습
  ✅ 완료: 전국 맛집/카페 POI (803,096행) + DB 로드
  ✅ 완료: 자전거보관소 (18,277행) + DB 로드
  ✅ 완료: 문화시설 POI (18,252행) + DB 재로드 (3,685 → 18,252행)
  ✅ 완료: Neon(512MB 한계) → 로컬 PostgreSQL 16 (port 5434) 전환 — 총 952,967행
  ✅ 완료: TourAPI 음식점(contentTypeId=39) 전국 수집 (10,535건, 17개 시도) + DB 적재
  ✅ 완료: K-Culture 촬영지 수집 (1,073건 — 관광지 필터 62건 + 여행코스 1,065건) + DB 적재
  ⏳ 남은 작업:
  - 둘레길 데이터 수집 (두루누비 공공데이터)
  - K-Culture 크롤링 보완 (나무위키, 팬위키)

[Week 2] 모델링 + 백엔드
  ✅ 완료: 안전 모델 전국 재학습 (R²=0.9995, F1=0.9987)
  - 소비 예측 모델 전국 데이터 보강
  - Ollama 로컬 LLM 연동 (여행 코스 생성)
  - FastAPI 엔드포인트 재설계

[Week 3] 프론트엔드 + 지도
  - React + 네이버 지도 (or Leaflet) 연동
  - 카테고리별 필터 UI (맛집/K-Culture/관광지/둘레길)
  - 일정별 코스 생성 UI (당일~3박4일)
  - 다국어 지원 (한/영)

[Week 4] 통합 + 배포
  - RAG 파이프라인 구축 (POI 정보 + LLM)
  - Docker Compose 구성
  - Vercel(React) + Railway(FastAPI) 배포
  - 모델 경량화 + HuggingFace 업로드
```

### 2차 MVP — "이동수단 확장 + 개인화" (이후)
- 대중교통 경로 (TAGO API)
- 자동차/기차 경로 (네이버 길찾기, 코레일)
- 사용자 프로파일 기반 개인화
- GraphRAG 확대

### 3차 MVP — "소셜 + 실시간" (이후)
- 사용자 리뷰/평점 시스템
- 실시간 혼잡도 연동
- MCP (Model Context Protocol) 통합
- 다국어 챗봇 (일본어/중국어)

---

## 4. 시스템 아키텍처

```
[사용자] ← HTTPS →
[React Frontend — Vercel]
  - 네이버 지도 / Leaflet 지도 시각화
  - 카테고리별 POI 필터 (맛집/K-Culture/관광/둘레길)
  - 일정 생성기 (당일~3박4일)
  - AI 챗봇 (여행 비서)
  - 다국어 (한/영)
       ↓ REST API
[FastAPI ML Server — Railway]
  - /poi/search         ← POI 검색 (카테고리+지역+반경)
  - /poi/recommend      ← Co-occurrence 추천
  - /course/generate    ← 일정별 코스 생성 (LLM 오케스트레이터)
  - /predict/consume    ← 소비 예측 (TabNet v2)
  - /weather            ← 날씨 예보 (LSTM + KMA)
  - /chat               ← RAG 챗봇 (Ollama + LangChain)
       ↓
[Ollama LLM — 로컬 or Groq API]
  - 코스 생성 오케스트레이션
  - 자연어 질의 → POI 검색 변환
  - RAG: POI DB + 벡터 임베딩
       ↓
[PostgreSQL + PostGIS — Neon]
  - poi (전국 POI 통합 DB)
  - course_template (일정별 코스 템플릿)
  - road_segment (자전거도로/둘레길)
  - user_profile (2차 MVP)
```

---

## 5. 데이터 수집 계획

### 5-1. 맛집 데이터 (카테고리 1)

| 소스 | 방법 | 예상 건수 | 상태 |
|------|------|----------|------|
| **소상공인 상가 ZIP (주력)** | `preprocess_soho_food.py` — 업종 필터 + 위경도 직접 사용 | **803,096건** | ✅ 전처리 완료, DB 로드 완료 |
| **모범음식점정보** (65,418행) | 소상공인 POI에 `is_exemplary=True` 플래그 매칭 | 플래그 (17,905건 매칭) | ✅ 완료 |
| **식품_관광식당** (519행) | 소상공인 POI에 `is_tourist_certified=True` 플래그 매칭 | 플래그 (107건 매칭) | ✅ 완료 |
| TourAPI 음식 카테고리 | `collect_tourapi_food.py` — contentTypeId=39 전국 17개 시도 | **10,535건** | ✅ 수집 완료 (2026-04-27), DB 적재 완료 |
| **또간집** (나무위키) | `collect_premium_food.py` — wiki-table 파싱 + 지오코딩 | ~500건 | 🔜 보류 (지오코딩 API 권한 해결 후 재개) |
| **블루리본** (bluer.co.kr) | `collect_premium_food.py` — 검색 페이지 스크래핑 + pagination | ~3,000건 | 🔜 보류 (SPA 렌더링 → Selenium 필요) |
| **레드리본** (bluer.co.kr/about/redribbon) | `collect_premium_food.py` — 정적 리스트 파싱 | ~수백건 | 🔜 보류 |
| 방송 맛집 (백종원 등) | 위키/나무위키 파싱 | ~1,000건 | [ ] |

> 카카오 로컬 API (FD6/CE7): 앱 권한 문제 → 소상공인 상가 ZIP으로 대체. 카카오 API는 해결되면 보완용으로 재활용 가능.

### 5-1-1. 프리미엄 맛집 수집 전략 (`collect_premium_food.py`) — 🔜 보류 중

> **현황 (2026-04-27)**: 스크립트 구현 완료, 수집 보류
> - 또간집(나무위키): ~143건 파싱 확인, 지오코딩 성공률 ~7% (Vworld+JUSO+Nominatim 조합)
> - 블루리본/레드리본: SPA(React) 렌더링으로 0건 — Selenium 필요
> - **보류 사유**: Kakao Local API 403 (앱 활성화 필요) + NCP Maps 401 (구독 신청 필요)
> - **재개 조건**: Kakao 또는 NCP API 활성화 → 성공률 70%+ 확인 후 DB 병합 진행

| 소스 | 수집 방식 | 지오코딩 | 특이사항 |
|------|-----------|---------|---------|
| **또간집 (나무위키)** | `wiki-table` 파싱 → 식당명 + 지역 추출 | 8단계 폴백 체인 | 파싱 OK, 지오코딩 성공률 낮음 |
| **블루리본** | REST API → `__NEXT_DATA__` → HTML 셀렉터 | 동일 | SPA 0건 → Selenium 필요 |
| **레드리본** | `/about/redribbon` 정적 리스트 | 동일 | SPA 0건 → Selenium 필요 |
| **지오코딩 체인** | Kakao→NCP→Naver→Vworld place→JUSO→Vworld road→Vworld parcel→Nominatim | 8단계 | Kakao/NCP 활성화 시 성공률 대폭 향상 예상 |

**DB 병합 전략**: `premium_food_geocoded.csv` 생성 후 `is_premium=True` 컬럼 추가 → `load_data.py --table food_poi` 로 기존 803K 건에 병합

### 5-2. K-Culture 성지순례 (카테고리 2-1)

| 소스 | 방법 | 예상 건수 |
|------|------|----------|
| 한국관광공사 촬영지 DB | `collect_kculture_poi.py` — 키워드 검색 + 관광지(12) K-Culture 필터 + 여행코스(25) | **1,073건** | ✅ 수집 완료 (2026-04-27), DB 적재 완료 |
| K-Drama 촬영지 위키 | 크롤링 (드라마별 촬영지 목록) | ~1,000건 | [ ] |
| K-Pop MV 촬영지 | 팬위키/나무위키 파싱 | ~300건 | [ ] |
| 팬카페/위버스 인기 장소 | 수동 큐레이션 + 크롤링 | ~200건 | [ ] |

### 5-3. 문화·관광지 (카테고리 2-2)

| 소스 | 방법 | 상태 |
|------|------|------|
| 한국관광공사 TourAPI | 전국 15,905건 | ✅ 보유 (`tour_poi.csv`) |
| **문화_박물관및미술관.csv** (1,265행) | Vworld / JUSO API 지오코딩 → `preprocess_culture_poi.py` | ✅ 완료 (1,066행) |
| **문화_한옥체험업.csv** (3,075행) | 동일, 영업중 필터 | ✅ 완료 (2,216행) |
| **문화_종합테마파크업/기타** (6,946행) | 동일, 영업중만 | ✅ 완료 (2,155행) |
| **문화_종합휴양업.csv** (41행) | 동일 | ✅ 완료 (10행) |
| **문화_국내여행업.csv** (20,126행) | 동일, 영업중 필터 | ✅ 완료 (3,208행) |
| **문화_종합여행업.csv** (19,265행) | 동일, 영업중 필터 | ✅ 완료 (9,532행) |
| **문화_시내순환관광업.csv** (114행) | 동일 | ✅ 완료 (65행) |
| **합계** | `culture_poi_nationwide.csv` | ✅ 완료 (**18,252행**) — DB 재로드 완료 |
| 국립중앙박물관 API | 공공데이터 | [ ] |
| 문화재청 국가문화유산 | 공공데이터 | [ ] |

### 5-4. 자연·둘레길·자전거 인프라 (카테고리 2-3 + 시설)

| 소스 | 방법 | 예상 건수 | 상태 |
|------|------|----------|------|
| 두루누비 (durunubi.kr) | 공공데이터 API / 크롤링 | ~800코스 | [ ] |
| 산림청 숲길 | 공공데이터 | ~500코스 | [ ] |
| 한국관광공사 도보코스 | TourAPI | 보유 데이터에서 필터 | ✅ |
| 전국 자전거도로 | data.go.kr (20,262행 보유) | 20,262행 | ✅ DB 로드 완료 |
| **자전거보관소정보.csv** (18,422행) | `preprocess_bike_facility.py` — WGS84 좌표 직접 사용 | 18,277행 | ✅ 전처리 + DB 로드 완료 |
| **bicycleDataset12_24.csv** (5,049행) | 자전거 사고/이용 데이터 2012~2024 — 전처리 스크립트 미작성 | 5,049행 | ⏳ 전처리 필요 (안전 모델 보완 또는 trail 데이터 활용 검토) |

### 5-5. 소비·여행 데이터 (카테고리 3)

| 소스 | 파일 | 행수/규모 | 상태 |
|------|------|----------|------|
| **국민여행조사 2024 (주력)** | `2024년 국민여행조사 국내여행 RAWDATA.xlsx` | **51,754명 응답, 소비값 유효 26,342행** | ✅ 보유 — 전처리 스크립트 작성 필요 |
| **네이버 데이터랩 방문자** | `202504-202603_데이터랩_다운로드2/` | 전국 시군구별 방문자 수 (202504~202603) | ✅ 보유 — 사이드 피처 활용 예정 |
| **네이버 데이터랩 검색** | `202504-202603_데이터랩_다운로드3/` | 카테고리별 검색건수 (음식 42.6% 1위) | ✅ 보유 — POI 가중치 활용 예정 |
| AI Hub 수도권 여행로그 2023 | `data/ai-hub/국내 여행로그 수도권_2023/` | 21,384행 방문지정보 | ⚠️ 수도권 한정 — POI 추천·매력도 모델 용도 |
| AI Hub 전국 여행로그 | — | — | [ ] 신청 필요 — POI 추천·매력도 전국화 시 필요 |

#### 국민여행조사 전처리 전략 (핵심 과제)

```
[문제] D_TRA1_COST 결측률 49.1% (26,342건 유효 / 51,754건 전체)
[해결] 소비값 있는 행만 필터링 → 결측 없는 완전한 26,342행 학습 세트 활용

[전처리 단계]
  1. 필터: D_TRA1_COST > 0 & notna() → 26,342행
  2. 타겟: D_TRA1_ONE_COST (1인당 비용) → log1p 변환
  3. 피처:
     - MON_EXP_1~5 (소득 대리, 100% 완전) → income_tier 3단계
     - D_TRA1_S_Day (박수), D_TRA1_NUM (동반자), D_TRA1_CASE (여행유형)
     - D_TRA1_1_SPOT (방문지 코드) → sido 매핑
     - SA1_1~5 (인구통계 코드, 100% 완전) → label encoding
  4. 이상치: 1%~99% 분위 필터
  5. 사이드 피처: 데이터랩 시도별 방문자 비율 조인 (선택)

[출력] data/raw_ml/national_travel_consume.csv (~24,000행)
[스크립트] src/preprocessing/preprocess_national_travel.py (신규 작성 필요)
[모델] ConsumeTabNet v3 — 전국 26,342행 기반 (v2 수도권 2,508행 대비 10.5×)
```

> ⚠️ 전국 단위 실데이터 원칙: AI Hub 전국 데이터 없이 수도권 데이터로 전국 모델을 만들거나 평균값으로 결측 지역을 채우는 것은 금지.
> ✅ 국민여행조사 데이터는 전국 조사 → ConsumeTabNet v3에 사용 가능 (전국 모델 인정)

---

## 6. 모델링 계획

### 6-1. 기존 모델 현황 및 계획 (2026-04-26)

| 모델 | 현재 성능 | 데이터 범위 | 상태 | 다음 단계 |
|------|-----------|------------|:---:|----------|
| **날씨 LSTM** | Acc=**82.16%**, F1=**0.7213** | 전국 67개 관측소 | ✅ 완료 | 시각화 완료 (`report/figures/`) |
| **안전 RF 회귀** | R²=**0.9995** (leakage⚠️) | 전국 도로 20,262행 + 전국 사고 381 시군구 | ✅ 완료 | 시각화 완료 (`report/figures/`) |
| **안전 RF 분류** | F1=**0.9987** (leakage⚠️) | 전국 도로 20,262행 + 전국 사고 381 시군구 | ✅ 완료 | 동일 |
| **소비 TabNet v2** | R²=**0.1277** | ⚠️ 수도권만 (2,508행) | ✅ 완료 | v3로 대체됨 |
| **소비 TabNet v3** | MAE=**₩42,764원**, R²=**0.5939** | ✅ 전국 (25,893행) | ✅ 완료 (2026-04-27) | 시각화 완료 (`report_step3_consume_tabnet.py`) |
| **POI 매력도 TabNet** | MAE=**0.6558**, R²=**0.0662** | ⚠️ 수도권만 (AI Hub 14,364행) | ✅ 모델 생성 완료 | 시각화 대기 (`report_step2_poi_tabnet.py`) |
| **Co-occurrence POI v2** | Recall@5=**0.1342**, @10=**0.1751** (베이스라인 대비 3.6×) | ⚠️ 수도권만 (2,536 trips) | ✅ 모델 생성 완료 | 시각화 대기 (`visualize_poi_recommender.py`) |

### 6-2. 신규 모델 — 역할별 후보 및 선택 기준

#### LLM (코스 생성 + 챗봇)

| 모델 | 크기 | 한국어 | 실행 방식 | 추천 용도 |
|------|------|--------|----------|-----------|
| `EEVE-Korean-10.8B` | 6.5GB | ★★★★★ | Ollama 로컬 | **한국어 챗봇 1순위** |
| `qwen2.5:14b` | 8.2GB | ★★★★★ | Ollama 로컬 | **한/중/영 멀티링궐** |
| `gemma2:9b` | 5.5GB | ★★★★ | Ollama 로컬 | 코스 생성 |
| `llama3.1:8b` | 4.7GB | ★★★ | Ollama 로컬 | 영어 질의 |
| `llama-3.3-70b` | — | ★★★★ | Groq API | 배포 환경 fallback |

#### Embedding 모델 (RAG 핵심)

| 모델 | 크기 | 특징 | 우선순위 |
|------|------|------|---------|
| `BAAI/bge-m3` | 570MB | 다국어, MTEB 최상위권 | **1순위** |
| `intfloat/multilingual-e5-small` | 118MB | 빠름, 경량 | 2순위 |
| `jhgan/ko-sroberta-multitask` | 110MB | 한국어 특화 경량 | 한국어 only 시 |

> 평가 방법: [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) Korean 탭 → Retrieval 점수 기준

#### Reranker (검색 후 정렬)

| 모델 | 특징 | 우선순위 |
|------|------|---------|
| `BAAI/bge-reranker-v2-m3` | 다국어, Cross-Encoder | **1순위** |
| `Cohere Reranker API` | API 방식, 무료 tier | 배포 fallback |

#### 멀티모달 (이미지 + 텍스트) — 프로젝트 핵심

| 모델 | 용도 | 크기 |
|------|------|------|
| `openai/clip-vit-large-patch14` | 사진 → 유사 POI 검색 | 890MB |
| `llava:13b` | 이미지 보고 자연어 답변 | 8GB (Ollama) |
| `qwen2.5-vl:7b` | 한국어 + 이미지 이해 | 4.5GB (Ollama) |
| `BLIP-2` | 이미지 캡션 자동 생성 | 3~12GB |

> **멀티모달 핵심 시나리오**
> - 사용자 사진 업로드 → CLIP 유사도 → POI 추천
> - "이 사진 어디야?" → LLaVA/Qwen2.5-VL → POI DB 매칭

#### 기타 신규 모델

| 모델 | 목적 | 기술 | 우선순위 |
|------|------|------|---------|
| **POI 카테고리 분류** | 맛집/K-Culture/관광/자연 분류 | BERT or 규칙 기반 | 1순위 |
| **리뷰 감성분석** | POI 품질 점수 | KR-FinBERT (zero-shot) | 2순위 |
| **코스 소비 예측** | 일정별 총 예상 비용 | TabNet v3 확장 | 1순위 |

---

### 6-3. 모델 성능 평가 방법

#### 임베딩 모델 비교 — Recall@K, NDCG@K

```python
# 평가용 정답 셋 구성 (query → ground_truth poi_id 매핑 20~50개)
eval_set = [
    {"query": "BTS 뮤비 촬영지 근처 맛집", "relevant": [poi_id_1, poi_id_2]},
    {"query": "서울 당일치기 K-Culture 코스", "relevant": [poi_id_3, poi_id_4]},
]
# 3개 모델(bge-m3 / e5-small / ko-sroberta) Recall@5 비교 → MLflow 기록
```

#### LLM 코스 생성 품질 평가 기준

```
동일 프롬프트 → 모델별 비교
  1. 코스 논리성  — 이동 거리 합리적인가 (PostGIS로 검증)
  2. 한국어 품질  — 1~5점 수동 평가
  3. 응답 속도    — 토큰/초 (ollama 로그 측정)
  4. 할루시네이션 — 실제 DB에 없는 POI 생성 횟수
  5. 맥락 유지    — 멀티턴에서 이전 맥락 반영 여부
```

#### MLflow 실험 추적 (기존 MLflow 활용)

```python
import mlflow
with mlflow.start_run(run_name="embedding_bge-m3"):
    mlflow.log_param("model", "BAAI/bge-m3")
    mlflow.log_param("chunk_strategy", "name+description+tags")
    mlflow.log_metric("recall_at_5", 0.42)
    mlflow.log_metric("ndcg_at_10", 0.38)
    mlflow.log_metric("latency_ms", 45)
```

---

### 6-4. Chunking → Retrieval → Reranking 파이프라인

POI 데이터는 텍스트가 짧으므로 **필드별 분리 청킹** 전략 사용.

```
[청킹]
  청크 1 (검색용):    name + category + sido/sigungu
  청크 2 (의미 검색): name + description + tags
  청크 3 (멀티모달):  name + image_url 참조
        ↓
[하이브리드 Retrieval]
  Dense  70%: BAAI/bge-m3 → ChromaDB 벡터 검색 Top-50
  Sparse 30%: BM25 키워드 정확 매칭
  Fusion    : RRF (Reciprocal Rank Fusion) 가중 합산
        ↓
[Reranking]
  BAAI/bge-reranker-v2-m3 Cross-Encoder → Top-5~10
        ↓
[지리 + 안전 후처리]
  PostGIS ST_DWithin 반경 필터
  safety_regressor.pkl 안전점수 가중
        ↓
[LLM 컨텍스트 주입 → 코스 생성 or 챗봇 답변]
```

---

### 6-5. LLM 전략 (저비용 원칙)

```
[우선순위 1] Ollama 로컬 실행 (무료)
  - EEVE-Korean-10.8B (한국어 챗봇)
  - qwen2.5:14b (멀티링궐)
  - RAG: ChromaDB + BAAI/bge-m3 임베딩

[우선순위 2] Groq API (무료 tier)
  - 배포 환경에서 로컬 GPU 없을 때
  - llama-3.3-70b (무료 30 req/min)

[우선순위 3] HuggingFace Inference API
  - 경량 모델 배포 (Spaces)
  - 임베딩 모델 호스팅

[우선순위 4] OpenAI API
  - Groq 장애 시 fallback
  - gpt-4o-mini (저비용)
```

### 6-6. RAG → GraphRAG → MCP 확대 로드맵

```
[1차 MVP] RAG
  - ChromaDB 벡터 스토어
  - 하이브리드 검색 (Dense + BM25) + Reranker
  - BAAI/bge-m3 임베딩 + bge-reranker-v2-m3
  - LLM이 검색 결과 기반 코스 생성
  - 멀티모달: CLIP 이미지 검색 (사진 → POI)

[2차 MVP] GraphRAG
  - POI 간 관계 그래프 (NEAR / FILMING_AT / VISITED_SAME)
  - NetworkX (프로토타입) → Neo4j (운영)
  - "BTS 촬영지 근처 맛집" 같은 관계 질의
  - LLaVA/Qwen2.5-VL 이미지 이해 추가

[3차 MVP] MCP (Model Context Protocol)
  - 외부 도구 연결 (날씨 API, 지도 API, 교통 API)
  - LLM이 도구를 직접 호출하여 실시간 정보 활용
```

#### 모델 탐색 실행 순서

```bash
# Step 1: Ollama LLM 비교
ollama pull EEVE-Korean-10.8B
ollama pull qwen2.5:14b
ollama pull gemma2:9b

# Step 2: 임베딩 + 리랭커
pip install sentence-transformers
# BAAI/bge-m3, multilingual-e5-small, ko-sroberta → Recall@K 비교
# BAAI/bge-reranker-v2-m3 → CrossEncoder

# Step 3: 멀티모달
ollama pull llava:13b
ollama pull qwen2.5-vl:7b

# Step 4: 배포용 Groq API
# console.groq.com → 무료 tier → llama-3.3-70b
```

---

## 7. 프론트엔드 설계

### 화면 흐름 (사용자 여정)

```
[1단계] 넷플릭스 스타일 인트로 — 풀스크린 영상/이미지 배경
  ↓
[2단계] 온보딩 Step 1: 좋아하는 아이돌/유명인 선택 (국내 TOP 50)
  ↓
[3단계] 온보딩 Step 2: 여행 목적 선택 (추억/가성비/힐링/맛집)
  ↓
[4단계] 온보딩 Step 3: 가고 싶은 여행 장소 선택 (9개 권역)
  ↓
[5단계] 취향 분석 완료 애니메이션
  ↓
[6단계] 메인 화면 (서비스 소개 + 차별성 + 맞춤 추천 + 지도)
```

### 7-1. 인트로 화면 (넷플릭스 스타일)

```
┌─────────────────────────────────────────────────────┐
│  풀스크린 영상/이미지 배경 (한국 관광 하이라이트)       │
│                                                     │
│              🇰🇷  K-Ride 2.0                         │
│         "한국을 발견하다 — AI가 만드는 나만의 여행"     │
│                                                     │
│              [시작하기 ▶]                             │
│                                                     │
│  ─── 또는 로그인 ───                                  │
└─────────────────────────────────────────────────────┘
```

### 7-2. 온보딩 Step 1 — 좋아하는 아이돌/유명인 선택

넷플릭스에서 영화를 골라 취향을 파악하듯, **국내 TOP 50 엔터테인먼트(그룹+솔로)**를 카드로 보여주고 다중 선택.

| 구분 | 포함 대상 (예시) |
|------|----------------|
| K-Pop 그룹 | BTS, BLACKPINK, Stray Kids, SEVENTEEN, aespa, NewJeans, (G)I-DLE, TWICE, EXO, TXT, ENHYPEN, LE SSERAFIM, ATEEZ, NCT, ITZY, IVE, NMIXX, Red Velvet, TREASURE, MONSTA X, GOT7, SHINee, 2NE1, BIGBANG |
| K-Pop 솔로 | IU(아이유), 태양, 지코, DEAN, 화사, 선미, 청하, BIBI, 로제, 제니 |
| 배우 | 이민호, 송혜교, 박서준, 전지현, 김수현, 변우석, 한소희, 송강 |
| 콘텐츠 | 오징어게임, 더글로리, 우영우, 눈물의여왕, 도깨비 |

> **UI**: 포스터/프로필 카드 그리드, 터치하면 체크마크(✓), 최소 3개 이상 선택 권장

### 7-3. 온보딩 Step 2 — 여행 목적(Trip Reason) 선택

| 이유 | 아이콘 | 설명 |
|------|:---:|------|
| 추억 | 📸 | 특별한 기억을 만들고 싶어요 |
| 가성비 | 💰 | 적은 비용으로 알차게! |
| 힐링 | 🧘 | 일상에서 벗어나 쉬고 싶어요 |
| 맛집 | 🍽️ | 맛있는 음식이 최우선! |

> **UI**: 넷플릭스 장르 선택 스타일 대형 카드, 다중 선택 가능

### 7-4. 온보딩 Step 3 — 가고 싶은 여행 장소 선택

| 지역 | 포함 영역 |
|------|----------|
| 서울 | 서울특별시 |
| 경기 | 경기도 |
| 인천 | 인천광역시 |
| 강원 | 강원특별자치도 |
| 경북 | 경상북도 |
| 경남 | 경상남도 |
| 전북 | 전라북도 (전북특별자치도) |
| 전남 | 전라남도 |
| 제주 | 제주특별자치도 |

> **UI**: 한국 지도 실루엣 위에 지역별 클릭 또는 카드 그리드 선택, 다중 선택 가능

### 7-5. 메인 화면 (온보딩 후 진입 — 대시보드)

```
┌─────────────────────────────────────────────────────┐
│  🇰🇷 K-Ride 2.0   [한국어|English]   [👤 프로필]      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ★ 히어로 섹션 — 서비스 소개                          │
│  "AI가 만드는 나만의 한국 여행 코스"                    │
│  [지도로 탐색하기]  [AI 코스 생성]                     │
│                                                     │
├─────────────────────────────────────────────────────┤
│  ✨ 우리 서비스만의 차별성 (4개 카드)                   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐               │
│  │K-Pop │ │AI코스│ │소비  │ │다국어│               │
│  │성지  │ │생성  │ │예측  │ │지원  │               │
│  └──────┘ └──────┘ └──────┘ └──────┘               │
│                                                     │
├─────────────────────────────────────────────────────┤
│  📍 당신을 위한 맞춤 추천 (온보딩 취향 반영)            │
│  [맛집 카드] [K-Culture 카드] [힐링 코스 카드] [...]   │
│                                                     │
├─────────────────────────────────────────────────────┤
│  🗺️ 지도 탐색 (네이버 or 카카오 지도)                 │
│  ┌────────────┬──────────────────────────┐          │
│  │ 필터 패널   │        지도 영역          │          │
│  │ - 카테고리  │  POI 마커 + 경로 오버레이 │          │
│  │ - 지역      │                          │          │
│  │ - 일정      │                          │          │
│  │ - AI 추천   │                          │          │
│  └────────────┴──────────────────────────┘          │
│                                                     │
├─────────────────────────────────────────────────────┤
│  📋 인기 코스  |  💰 예상 비용  |  🌤️ 날씨 예보       │
└─────────────────────────────────────────────────────┘
│  💬 AI 여행 비서 (플로팅 챗봇 버튼)                    │
```

### 7-6. 지도 연동 계획 (네이버 or 카카오)

| 항목 | 네이버 지도 | 카카오 지도 |
|------|-----------|-----------|
| JavaScript API | `naver.maps` | `kakao.maps` |
| 무료 쿼터 | 월 3만건 | 월 30만건 |
| 국내 품질 | ★★★★★ | ★★★★☆ |
| 외국인 친화 | ★★★☆☆ | ★★☆☆☆ |

**지도에 붙일 기존 여행지 추천 검색 로직:**
1. 카테고리 필터 → DB `poi.category` (food/kculture/tourism/nature)
2. 지역 필터 → DB `poi.sido/sigungu`
3. 반경 검색 → PostGIS `ST_DWithin` 쿼리
4. 안전 점수 오버레이 → `safety_regressor.pkl`
5. 날씨 정보 → WeatherLSTM + KMA API
6. 소비 예측 → ConsumeTabNet v3

### AI 챗봇 패널

```
💬 "BTS 뮤비 촬영지 돌면서 맛집도 가고 싶어요. 1박2일로!"

🤖 추천 코스:
  Day 1: 용산 하이브 → 성수동 MV 촬영지 → 을지로 맛집 → 홍대
  Day 2: 경복궁 → 북촌 한옥마을 → 광장시장

  💰 예상 비용: ₩185,000 (숙박 포함)
  🌤️ 내일 날씨: 맑음 22°C
```

---

## 8. DB 스키마 (PostgreSQL + PostGIS)

```sql
-- 통합 POI 테이블
CREATE TABLE poi (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    name_en         VARCHAR(200),          -- 영문명 (다국어)
    category        VARCHAR(30) NOT NULL,  -- food/kculture/tourism/nature/trail
    sub_category    VARCHAR(50),           -- MV촬영지/드라마촬영지/블루리본 등
    sido            VARCHAR(20),
    sigungu         VARCHAR(30),
    address         VARCHAR(200),
    geom            GEOMETRY(POINT, 4326),
    description     TEXT,
    description_en  TEXT,
    source          VARCHAR(50),           -- kakao/tourapi/crawl/manual
    score           DECIMAL(3,2),          -- 0.00~5.00
    visit_count     INTEGER DEFAULT 0,
    image_url       VARCHAR(500),
    tags            TEXT[],                -- ['BTS','MV','성수동']
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_poi_geom ON poi USING GIST(geom);
CREATE INDEX idx_poi_category ON poi(category);
CREATE INDEX idx_poi_tags ON poi USING GIN(tags);

-- 코스 템플릿
CREATE TABLE course_template (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(200),
    title_en        VARCHAR(200),
    duration_days   SMALLINT,              -- 1=당일, 2=1박2일 ...
    category        VARCHAR(30),
    transport       VARCHAR(20),           -- walk/bike/public/car
    estimated_cost  INTEGER,               -- 원
    poi_ids         INTEGER[],             -- 방문 POI 순서
    route_geom      GEOMETRY(LINESTRING, 4326),
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 둘레길/자전거도로
CREATE TABLE trail (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200),
    trail_type      VARCHAR(20),           -- dullegil/bike_road/forest
    sido            VARCHAR(20),
    length_km       DECIMAL(8,3),
    difficulty      SMALLINT,              -- 1=쉬움, 2=보통, 3=어려움
    has_stamp       BOOLEAN DEFAULT FALSE,
    geom            GEOMETRY(LINESTRING, 4326),
    source          VARCHAR(50)
);
CREATE INDEX idx_trail_geom ON trail USING GIST(geom);
```

---

## 9. 배포 전략 (저비용)

| 서비스 | 기술 | 배포처 | 비용 |
|--------|------|--------|------|
| Frontend | React + Vite | Vercel | 무료 |
| ML API | FastAPI | Railway (500h/월) | 무료 |
| LLM | Ollama | 로컬 or Groq API | 무료 |
| DB (개발) | PostgreSQL 16 + PostGIS | 로컬 (port 5434) | 무료 |
| DB (배포) | PostgreSQL + PostGIS | Neon or Supabase | 무료 |
| 벡터 DB | ChromaDB | FastAPI 내장 | 무료 |
| ML 모델 파일 | pkl/pt/zip | HuggingFace Hub | 무료 |
| GPU 학습 | Colab / Kaggle | Google Drive 연동 | 무료 |
| 도메인 | — | Vercel 기본 도메인 | 무료 |

---

## 12. DB 셋업 가이드 (실행 전 준비)

### DB 전환 이력

- **Neon (2026-04-26)**: 무료 512MB — 기본 데이터 131K행은 OK, food_poi 803K행에서 용량 초과
- **로컬 PostgreSQL 16 (2026-04-27)**: 포트 5434, PostGIS 3.x 설치 — 938,400행 전체 로드 성공
- **배포 시**: Neon URL 주석으로 보관 중 — 인증 POI만 필터해서 Neon에 넣거나, Supabase 등 대안 검토

### 현재 상태 (2026-04-26 17:20 업데이트)

| 항목 | 상태 |
|------|:---:|
| `psycopg2` 패키지 | ✅ 설치 완료 (2.9.12) |
| `openpyxl` 패키지 | ✅ 설치 완료 (3.1.5) |
| `python-dotenv` 패키지 | ✅ 설치됨 |
| `.env` 파일 | ✅ DATABASE_URL 설정 완료 (Neon) |
| Neon 계정 | ✅ 생성 완료 (ap-southeast-1) |
| `init_db.py`, `load_data.py` 코드 | ✅ 완성 |

### 준비 단계

```
[✅] Step 1. 패키지 설치
  pip install psycopg2     ← 완료 (2.9.12)
  pip install openpyxl     ← 완료 (3.1.5, 사고 Excel 읽기용)
  (python-dotenv는 이미 설치됨)

[✅] Step 2. Neon DB 생성 완료
  → ap-southeast-1 리전
  → Connection string → .env에 저장 완료

[✅] Step 3. .env 파일에 DATABASE_URL 추가 완료
  DATABASE_URL=postgresql://neondb_owner:***@ep-fragrant-dream-***.neon.tech/neondb?sslmode=require
  (KMA_API_KEY, ASOS_API_KEY 도 있음)

[✅] Step 4. 스키마 초기화 완료
  python src/db/init_db.py
  → poi, course_template, trail, district_danger, weather_daily 테이블 생성됨

[✅] Step 5–8. DB 로드 전체 완료 (Neon → 로컬 PG16 전환, 총 952,967행)
  → poi(tour_poi): 15,905행
  → poi(facility): 3,368행
  → poi(food_poi): 803,096행 ✅
  → poi(bike_facility): 18,277행 ✅
  → poi(culture_poi): 18,252행 ✅ (재로드 완료 2026-04-27)
  → trail: 20,262행
  → danger: 381행
  → weather: 73,426행

[ ] Step 9. Week 1 남은 데이터 수집
  - K-Culture 촬영지 수집 (TourAPI + 크롤링)
  - 둘레길 데이터 수집 (두루누비)
  - TourAPI 음식 카테고리 (contentTypeId=39)
```

### 생성된 DB 스크립트

| 파일 | 역할 |
|------|------|
| `src/db/init_db.py` | PostGIS 확장 + 전체 테이블 생성 (`--drop` 옵션으로 초기화) |
| `src/db/load_data.py` | 기존 CSV 일괄 로드 (`--table poi/trail/danger/weather`) |

### 전국 데이터 수집 스크립트 (DB 저장 전 선행)

```bash
# 1. 자전거도로 전처리 ✅ 완료 (20,262행, 16개 시도)
python src/preprocessing/preprocess_road_nationwide.py

# 2. 안전 모델 학습 ✅ 완료 (R²=0.9919, F1=0.9990)
python src/ml/build_safety_model.py

# 3. 날씨 시각화 ✅ 완료 (Acc=82.16%, F1=0.7213)
python src/dl/visualize_weather_lstm.py

# 4. 사고 위험도 수집 (전국 확대 필요)
python src/data_collect/collect_nationwide_accident.py

# 5. 편의시설 수집 + DB 저장 (KAKAO_REST_API_KEY 필요)
python src/data_collect/collect_nationwide_facility.py --db

# 6. 날씨 수집 (KMA_API_KEY 필요, 시간 소요)
python src/data_collect/collect_nationwide_weather.py
```

---

### 모델 경량화 전략

```
[로컬 학습] 원천 모델 (full precision)
  ↓ 양자화 / ONNX 변환
[HuggingFace / Roboflow / Kaggle 업로드] 경량 모델
  ↓ 런타임 다운로드
[Railway / Render] FastAPI 서버에서 추론
```

---

## 10. 기술 스택 요약

| 레이어 | 기술 |
|--------|------|
| Frontend | React, Vite, TypeScript, Leaflet or Naver Maps |
| Backend | FastAPI, Python 3.11+ |
| ML/DL | scikit-learn, PyTorch, pytorch-tabnet |
| LLM | Ollama (llama3/gemma2), LangChain, Groq API |
| RAG | ChromaDB, sentence-transformers |
| DB | PostgreSQL + PostGIS (Neon) |
| 배포 | Docker Compose, Vercel, Railway |
| 데이터 | 공공데이터포털(소상공인/자전거보관소/문화시설), TourAPI, Vworld API / JUSO API, AI Hub |
| 버전관리 | Git + Git LFS, HuggingFace Hub |

---

## 11. 환경 변수

```env
# API Keys
KMA_API_KEY=...
KAKAO_REST_API_KEY=...
VWORLD_API_KEY=...          # Vworld 지오코딩 (국토지리정보원)
JUSO_CONFIRM_KEY=...        # JUSO 도로명주소 API (행안부) — 둘 중 성공률 높은 것 사용
TOUR_API_KEY=...

# LLM
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=...
OPENAI_API_KEY=...          # fallback

# DB
DATABASE_URL=postgresql://...@...neon.tech/kride

# Deployment
HF_REPO_ID=your-name/kride-models
VERCEL_URL=https://kride.vercel.app
```
