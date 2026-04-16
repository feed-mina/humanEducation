# K-Ride 리팩토링 & 전국 확대 계획

> 작성일: 2026-04-16
> 목적: ML/DL 성능 향상 + 전국 확대 + SpringBoot / FastAPI / React 배포 아키텍처 설계

---

## 1. 전체 목표 로드맵

```
[현재]                          [단기 목표]                 [장기 목표]
서울 자전거도로 1,647개          → 전국 자전거도로 15,000+    → 전국 자전거 여행 플랫폼
Streamlit 단일 앱              → FastAPI + React (웹앱)    → SpringBoot MSA 배포
수도권 AI Hub 데이터            → 전국 AI Hub + TAAS        → 실시간 데이터 연동
소비 R²=0.1277 (v2)            → 소비 R²≥0.25             → 소비 R²≥0.35+
날씨 Acc=73%                   → 날씨 Acc≥80%             → 실시간 LSTM 예보
Co-occ Recall@5=0.1372 (v2)    → Recall@5≥0.20            → Recall@5≥0.30 (전국)
```

---

## 2. 즉시 실행 — ML 성능 향상 (Phase R-1)

> **기간**: 이번 주 (2026-04-16 ~)
> **선결 조건**: 없음 (이미 보유한 데이터만 사용)

### 2-1. 소비 모델 리빌드 (build_consume_model_v2.py)

**목표**: R²=0.0053 → R²≥0.20

#### 실행 순서

| 단계 | 작업 | 예상 시간 |
|------|------|---------|
| 1 | `tn_traveller_master_*.csv` 파일 존재 확인 + TRAVEL_ID 머지 | 30분 |
| 2 | 이상치 제거 (q01~q99) + 로그 변환 타겟 | 30분 |
| 3 | 타겟 재정의: 4개 소비 테이블 합산 (파일 있으면) | 1시간 |
| 4 | 피처 추가 (income_tier, age_grp, gender, travel_purpose) | 1시간 |
| 5 | TargetEncoding(sgg_code) + 계절 가중치 | 30분 |
| 6 | TabNet 재학습 (3분할: 70/15/15) | 1~2시간 |
| 7 | 결과 비교 (old vs new) | 30분 |

#### 소득 분위 → 3단계 구간 변환 (income_tier)

연속형 소득 분위(1~8)를 소비 패턴이 유사한 3단계로 구간화.

```python
# build_consume_model_v2.py — 소득 구간화 전처리
def map_income_tier(income: int) -> int:
    """
    INCOME (1~8분위) → income_tier (0/1/2)
    0=짠순이 (1~3분위): 저소득, 비용 최소화
    1=보통   (4~6분위): 중간 소득, 평균적 소비
    2=호캉스 (7~8분위): 고소득, 숙박/식사 지출 많음
    """
    if income <= 3:
        return 0   # 짠순이
    elif income <= 6:
        return 1   # 보통
    else:
        return 2   # 호캉스

df["income_tier"] = df["INCOME"].apply(map_income_tier)
# income_tier는 범주형 → OneHotEncoding 또는 CatBoost 범주형 처리
```

| income_tier | 원본 분위 | 레이블 | 예상 소비 수준 |
| ----------- | --------- | ------ | ------------- |
| 0 | 1~3분위 | 짠순이 | 저소비, 활동 중심 |
| 1 | 4~6분위 | 보통 | 평균적 소비 패턴 |
| 2 | 7~8분위 | 호캉스 | 숙박/식음료 비중 높음 |

#### 새 파일

| 파일 | 설명 |
|------|------|
| `build_consume_model_v2.py` | 개선된 소비 예측 모델 스크립트 |
| `models/consume_regressor_v2.zip` | v2 TabNet 가중치 |
| `models/consume_meta_v2.json` | MAE/R²/피처 목록 (3분할 기준) |

### 2-2. Co-occurrence 지리 필터 추가 (build_poi_recommender_v2.py)

**목표**: Recall@5=0.1260 → Recall@5≥0.18

```python
# 추가 기능: 현재 위치 반경 내 POI만 추천 + 카테고리 부스트
def recommend_v2(seeds, lat, lon, radius_km=20, top_n=10, category_boost=True):
    scores = jaccard_aggregate(seeds)
    if lat and lon:
        scores = filter_by_radius(scores, lat, lon, radius_km)
    if category_boost:
        scores = boost_same_category(scores, seeds)
    return top_n_results(scores, top_n)
```

### 2-3. 경기도 레저스포츠 POI 재수집 (step3_tour_collect_v2.py)

| 항목 | 현재 | 목표 |
|------|------|------|
| 레저스포츠 POI | 83개 (경기도 타임아웃) | 500개+ |
| 방법 | 공공데이터포털 (재시도) | 새벽 시간대 + 타임아웃 60초 |
| API | 한국관광공사 TourAPI contentTypeId=28 | 동일 |

---

## 3. 단기 계획 — 전국 데이터 수집 (Phase R-2)

> **기간**: 2주차 (2026-04-23~)
> **선결 조건**: 데이터 다운로드/신청

### 3-1. 데이터 수집 체크리스트

| 데이터 | URL | 방법 | 상태 |
|-------|-----|------|------|
| 전국 자전거도로 (공공데이터포털) | data.go.kr 검색: 자전거도로 현황 | CSV 다운로드 | [ ] 미수집 |
| TAAS 자전거 사고 데이터 | taas.koroad.or.kr | 회원가입 후 다운로드 | [ ] 미수집 |
| AI Hub 전국 여행로그 | aihub.or.kr | 신청 후 1~3일 승인 | [ ] 미신청 |
| 한국관광공사 TourAPI (전국) | data.visitkorea.or.kr | API 키 발급 후 전국 수집 | [x] 완료 — 15,905건 (관광지 8,931 / 레저 3,243 / 숙박 2,080 / 문화 1,651) |
| 전국 ASOS 기상 관측소 | data.kma.go.kr | API 키 발급됨 → 관측소 확대 | [ ] 서울/경기만 |
| 카카오 로컬 API (맛집/POI) | developers.kakao.com | REST API 키 (테스트 앱) | [x] **사용 가능** — 로컬 REST API는 비즈니스 등록 불필요, 테스트 앱으로 FD6/CE7 반경 검색 |
| 네이버 DataLab 검색어 트렌드 | developers.naver.com | Client ID/Secret 발급 완료 (1,000콜/일) | [x] 완료 — krider 앱 등록, sns_mention_norm 산출에 활용 |
| 네이버 지역 검색 API (맛집) | developers.naver.com | API 설정 탭에서 지역 검색 추가 (25,000콜/일) | [ ] 미추가 — 카카오 REST API 우선 사용, 보조 수단으로 추가 가능 |
| 네이버 Cloud Maps | console.ncloud.com | Geocoding / Reverse Geocoding | [x] **완료** — krider 앱 등록 (2026-04-16), 주소↔좌표 변환 활용 |
| 한국관광공사 숙박 API | data.visitkorea.or.kr | TourAPI contentTypeId=32 (숙박) | [x] 완료 — --include_lodging 옵션으로 포함 수집 (전국 2,080건) |

### 3-1-1. 외부 API 연계 계획 (관광 점수 · 맛집 POI)

> **여기어때 / 야놀자**: 공개 API 없음 (파트너 계약 필요) → **한국관광공사 숙박 API**로 대체  
> **또간집 / 망고플레이트 / 블루리본**: 공개 API 없음, 크롤링 약관 위반 위험 → **카카오 로컬 API**로 맛집 정보 수집  
> **카카오 로컬 API**: REST API는 비즈니스 등록 불필요 → **사용 가능** (테스트 앱 REST 키, FD6 음식점/CE7 카페 반경 검색, 300,000콜/일)  
> **네이버 Cloud Maps**: Geocoding/Reverse Geocoding 등록 완료 (2026-04-16) → 주소↔좌표 변환에 활용

#### 네이버 DataLab 검색어 트렌드 — tourism_score sns_mention_norm 산출

> 발급 완료 (2026-04-16): Client ID / Client Secret (네이버 krider 앱)  
> 제약: 1,000콜/일, 요청당 최대 5개 키워드 → 주요 POI 배치 처리 필요

```python
# step4_naver_trend.py — 네이버 DataLab으로 관광지별 검색 트렌드 수집
import os, requests, time

NAVER_CLIENT_ID     = os.environ["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = os.environ["NAVER_CLIENT_SECRET"]

def get_search_trend(keyword_groups: list, start: str, end: str,
                     time_unit: str = "month") -> dict:
    """
    네이버 DataLab 검색어 트렌드
    - keyword_groups: [{"groupName": "한강공원", "keywords": ["한강공원"]}] 최대 5개
    - 반환: {groupName: avg_ratio (0~100)}
    """
    resp = requests.post(
        "https://openapi.naver.com/v1/datalab/search",
        headers={
            "X-Naver-Client-Id":     NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            "Content-Type":          "application/json",
        },
        json={
            "startDate":    start,       # "2024-01-01"
            "endDate":      end,         # "2024-12-31"
            "timeUnit":     time_unit,
            "keywordGroups": keyword_groups,
        },
        timeout=10,
    )
    data = resp.json()
    result = {}
    for item in data.get("results", []):
        ratios = [p["ratio"] for p in item.get("data", []) if p["ratio"] > 0]
        result[item["title"]] = sum(ratios) / len(ratios) if ratios else 0.0
    return result

# 배치 처리: tour_poi.csv에서 상위 N개 POI 트렌드 수집
# 1,000콜/일 ÷ 5키워드 = 5,000개 POI/일 처리 가능
def collect_poi_trends(poi_titles: list, batch_size: int = 5) -> dict:
    all_trends = {}
    for i in range(0, len(poi_titles), batch_size):
        batch = poi_titles[i:i + batch_size]
        groups = [{"groupName": t, "keywords": [t]} for t in batch]
        trends = get_search_trend(groups, "2024-01-01", "2024-12-31")
        all_trends.update(trends)
        time.sleep(0.1)  # API 레이트 리밋 방지
    return all_trends
```

#### 네이버 지역 검색 API — 맛집 POI 수집 (Kakao 대체)

> krider 앱 API 설정 탭 → '지역(검색)' API 추가 신청 후 사용 가능  
> 25,000콜/일 무료

```python
# step3_food_collect.py — 네이버 지역 검색으로 맛집 수집
def collect_food_poi_naver(query: str, display: int = 5) -> list:
    """
    네이버 지역 검색 API (카카오 FD6 대체)
    - query: "한강공원 맛집" 등 복합 키워드
    - 반환 필드: title, category, address, mapx, mapy
    """
    resp = requests.get(
        "https://openapi.naver.com/v1/search/local.json",
        headers={
            "X-Naver-Client-Id":     NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        },
        params={"query": query, "display": display, "sort": "comment"},
        timeout=10,
    )
    items = resp.json().get("items", [])
    # mapx/mapy는 KATECH 좌표 → WGS84 변환 필요 (/ 10,000,000.0)
    for item in items:
        item["lon_wgs84"] = int(item.get("mapx", 0)) / 10_000_000.0
        item["lat_wgs84"] = int(item.get("mapy", 0)) / 10_000_000.0
    return items
```

#### 관광 점수 공식 (업데이트)

```
tourism_score_final =
    0.5 × poi_density_score     (관광지/레저 POI 밀도 — TourAPI 15,905건 기반)
  + 0.2 × attraction_score      (방문자 만족도 TabNet)
  + 0.2 × food_poi_density      (맛집 POI 밀도 — 카카오 로컬 FD6/CE7 반경 검색, step3_food_collect.py)
  + 0.1 × sns_mention_norm      (네이버 DataLab 검색 트렌드 정규화 — API 발급 완료)
```

#### DB 반영: poi 테이블 category 추가

```sql
-- 기존 category: '관광지', '문화', '레저', '편의시설'
-- 추가 category: '맛집', '숙박'

-- 맛집 POI 삽입 예시
INSERT INTO poi (name, category, sido_nm, sgg_nm, geom, visit_count)
VALUES ('성수동 카페거리', '맛집', '서울', '성동구',
        ST_SetSRID(ST_MakePoint(127.056, 37.544), 4326), 0);
```

### 3-2. 전국 전처리 파이프라인 설계

```
step1_facility_clean.py  (현재: 서울)  → step1_v2 (전국)
step2_road_clean.py      (현재: 서울/경기)  → step2_v2 (전국)
step3_tour_collect.py    (현재: 서울/경기)  → step3_v2 (전국 TourAPI)
step4_spatial_join.py    (현재: 서울)  → step4_v2 (전국)
build_safety_model.py    (현재: 1,647행)  → v2 (전국 15,000+ 행)
build_route_graph.py     (현재: 서울 OSM)  → v2 (전국 OSM)
```

### 3-3. 전국 OSM 경로 그래프 빌드 전략

전국을 한번에 빌드하면 RAM 32GB+ 필요. 시도별 분할 방식 채택:

```python
SIDO_CONFIG = {
    "서울": {"bbox": (37.413, 37.715, 126.764, 127.185), "done": True},
    "경기": {"bbox": (37.000, 38.300, 126.500, 127.900), "done": False},
    "부산": {"bbox": (35.000, 35.400, 128.800, 129.300), "done": False},
    "인천": {"bbox": (37.200, 37.900, 126.300, 126.900), "done": False},
    "대구": {"bbox": (35.700, 36.100, 128.400, 128.800), "done": False},
    "대전": {"bbox": (36.100, 36.600, 127.200, 127.600), "done": False},
    "광주": {"bbox": (35.000, 35.300, 126.600, 127.000), "done": False},
    # ... 나머지 광역시도
}

# 시도별 pkl 생성 후 서비스 시 필요한 시도만 로드
# route_graph_seoul.pkl, route_graph_busan.pkl ...
```

---

## 4. SpringBoot + FastAPI + React 배포 아키텍처

### 4-1. 전체 시스템 아키텍처

```
[사용자]
   |
   | HTTPS
   v
[React Frontend — Vercel]
  - 지도 시각화 (네이버 지도 or Leaflet)
  - 경로 추천 UI
  - AI 비서 채팅 (WebSocket)
  - 사용자 프로파일
   |
   | REST API / WebSocket
   v
[SpringBoot Gateway — 메인 백엔드]
  - 인증/인가 (JWT)
  - 사용자 관리
  - 비즈니스 로직 (경로 저장, 즐겨찾기)
  - PostgreSQL + PostGIS
   |
   | HTTP 내부 통신
   v
[FastAPI ML Server — Python 모델 서버]
  - /api/predict/safety   ← RF 모델
  - /api/predict/consume  ← TabNet 모델
  - /api/recommend/route  ← Dijkstra
  - /api/recommend/poi    ← Co-occurrence
  - /api/weather          ← LSTM + KMA API
  - /api/ai-chat          ← LangChain Agent (Phase 8)
   |
   v
[PostgreSQL + PostGIS]
  - road_segment (공간 인덱스)
  - poi (전국 관광지)
  - travel_log
  - user_profile
```

### 4-2. SpringBoot 역할 (Java)

| 기능 | 엔드포인트 | 설명 |
|------|-----------|------|
| 회원가입/로그인 | POST /auth/signup, /auth/login | JWT 발급 |
| 사용자 프로파일 | GET/PUT /user/profile | 나이/성별/자전거종류/사는곳 |
| 경로 저장/조회 | POST/GET /routes | 추천 경로 즐겨찾기 |
| 리뷰 작성 | POST /reviews | POI 리뷰 (KLUE-BERT 감성 분석 연동) |
| 알림 | WebSocket /ws/alert | 이벤트/날씨 알림 |
| ML 서버 프록시 | 내부 HTTP → FastAPI | 인증된 사용자만 ML API 접근 |

#### SpringBoot 의존성 (build.gradle)

```groovy
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-security'
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
    implementation 'io.jsonwebtoken:jjwt-api:0.11.5'
    implementation 'org.postgresql:postgresql'
    // PostGIS
    implementation 'org.hibernate.orm:hibernate-spatial'
    // WebSocket
    implementation 'org.springframework.boot:spring-boot-starter-websocket'
    // RestTemplate (FastAPI 통신)
    implementation 'org.springframework.boot:spring-boot-starter-webflux'
}
```

#### PostgreSQL + PostGIS (SpringBoot 연동)

```yaml
# application.properties
spring.datasource.url=jdbc:postgresql://localhost:5432/kride
spring.datasource.username=kride_user
spring.datasource.password=${DB_PASSWORD}
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect
```

### 4-3. FastAPI ML Server 역할 (Python)

| 엔드포인트 | 모델 | 입력 | 출력 |
|-----------|------|------|------|
| POST /predict/safety | safety_regressor.pkl | {width_m, length_km, sgg_code, ...} | {safety_score, danger_level} |
| POST /predict/consume | consume_regressor_v2.zip | {sgg_code, duration_h, income, ...} | {predicted_amt, confidence} |
| POST /recommend/route | route_graph.pkl + Dijkstra | {start_lat, start_lon, end_lat, end_lon, weights} | {path: [...], distance_km, time_min} |
| POST /recommend/poi | poi_cooccurrence.pkl | {seeds: [...], lat, lon, radius_km} | {recommendations: [...]} |
| GET /weather | weather_lstm.pt + KMA API | {lat, lon, date} | {weather, temp, rain_prob, safety_weight} |
| POST /ai-chat | LangChain Agent (Phase 8) | {message, intent, user_profile} | {response, map_geojson} |
| GET /segments | road_segment DB | {sido, bbox} | {segments: [...]} |

#### FastAPI 서버 구조

```
fastapi_server/
├── main.py                  ← FastAPI 앱 + CORS
├── routers/
│   ├── predict.py           ← /predict/safety, /predict/consume
│   ├── recommend.py         ← /recommend/route, /recommend/poi
│   ├── weather.py           ← /weather
│   ├── segments.py          ← /segments
│   └── ai_chat.py           ← /ai-chat (Phase 8)
├── models/
│   ├── loader.py            ← 모델 파일 로딩 (앱 시작 시 1회)
│   └── schemas.py           ← Pydantic 요청/응답 스키마
├── services/
│   ├── route_service.py     ← Dijkstra 경로 탐색
│   ├── safety_service.py    ← RF 안전 예측
│   ├── consume_service.py   ← TabNet 소비 예측
│   └── poi_service.py       ← Co-occurrence 추천
└── db/
    └── database.py          ← SQLAlchemy + PostGIS
```

#### FastAPI 주요 엔드포인트 코드 구조

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict, recommend, weather, segments

app = FastAPI(title="K-Ride ML Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # React dev
        "https://kride.vercel.app",       # 배포
        "http://localhost:8080",          # SpringBoot
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,   prefix="/predict")
app.include_router(recommend.router, prefix="/recommend")
app.include_router(weather.router)
app.include_router(segments.router)

# 모델 사전 로딩 (서버 시작 시 1회)
@app.on_event("startup")
async def load_models():
    from models.loader import ModelLoader
    app.state.models = ModelLoader.load_all()
```

```python
# models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    safety_weight: float = 0.6
    tourism_weight: float = 0.4
    bike_type: str = "city"         # road/mtb/city/electric/folding

class RouteResponse(BaseModel):
    path: List[dict]                # [{lat, lon}, ...]
    distance_km: float
    estimated_time_min: int
    avg_safety_score: float
    avg_tourism_score: float
    pois_on_route: List[dict]

class ConsumeRequest(BaseModel):
    sgg_code: str
    duration_h: float
    distance_km: float
    companion_cnt: int = 1
    season: int                     # 1~4
    has_lodging: bool = False
    income_tier: Optional[int] = None  # 0=짠순이(1~3분위), 1=보통(4~6분위), 2=호캉스(7~8분위)
    age_grp: Optional[str] = None
    gender: Optional[str] = None

class ConsumeResponse(BaseModel):
    predicted_amt: int
    log_predicted: float
    confidence_note: str            # "MAE ±X원 / R²=0.XX"
    model_version: str
```

### 4-4. React Frontend 구조

```
react-kride/
├── src/
│   ├── components/
│   │   ├── Map/
│   │   │   ├── KrideMap.tsx        ← 네이버 지도 or Leaflet
│   │   │   ├── RouteLayer.tsx      ← 경로 폴리라인
│   │   │   └── PoiMarkers.tsx      ← POI 마커
│   │   ├── Chat/
│   │   │   ├── AiAssistant.tsx     ← AI 비서 채팅 (Phase 8)
│   │   │   └── IntentButtons.tsx   ← 목적 선택 버튼
│   │   ├── Route/
│   │   │   ├── RouteForm.tsx       ← 출발/도착 입력
│   │   │   └── RouteCard.tsx       ← 추천 경로 카드
│   │   ├── Profile/
│   │   │   └── UserProfile.tsx     ← 나이/자전거/사는곳
│   │   └── Weather/
│   │       └── WeatherWidget.tsx   ← 날씨 + 안전 가중치
│   ├── pages/
│   │   ├── Home.tsx                ← 메인 지도
│   │   ├── RouteSearch.tsx         ← 경로 탐색
│   │   ├── PoiRecommend.tsx        ← 관광지 추천
│   │   └── AiAssistant.tsx         ← AI 비서 전용 페이지
│   ├── api/
│   │   ├── mlApi.ts               ← FastAPI 통신
│   │   └── springApi.ts           ← SpringBoot 통신
│   └── store/
│       └── userStore.ts           ← Zustand 상태 관리
└── package.json
```

#### React → SpringBoot → FastAPI 데이터 흐름

```
React (사용자 경로 검색 입력)
  |
  | POST /api/routes/recommend (with JWT)
  v
SpringBoot (인증 확인 + 요청 로깅)
  |
  | POST http://fastapi:8001/recommend/route
  v
FastAPI (route_graph.pkl + Dijkstra)
  |
  | JSON response (path, scores, POIs)
  v
SpringBoot (경로 DB 저장 + 응답 가공)
  |
  | JSON response
  v
React (지도 렌더링 + 카드 표시)
```

---

## 5. 배포 환경 구성

### 5-1. 개발 환경 (Docker Compose)

```yaml
# docker-compose.yml
version: "3.8"

services:
  postgres:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: kride
      POSTGRES_USER: kride_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  fastapi:
    build: ./fastapi_server
    ports:
      - "8001:8001"
    environment:
      DATABASE_URL: postgresql://kride_user:${DB_PASSWORD}@postgres:5432/kride
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      KMA_API_KEY: ${KMA_API_KEY}
    volumes:
      - ./kride-project/models:/app/models  # 모델 파일 마운트
    depends_on:
      - postgres

  springboot:
    build: ./springboot_server
    ports:
      - "8080:8080"
    environment:
      DB_URL: jdbc:postgresql://postgres:5432/kride
      DB_PASSWORD: ${DB_PASSWORD}
      FASTAPI_URL: http://fastapi:8001
      JWT_SECRET: ${JWT_SECRET}
    depends_on:
      - postgres
      - fastapi

  react:
    build: ./react-kride
    ports:
      - "3000:3000"
    environment:
      REACT_APP_SPRING_URL: http://localhost:8080
      REACT_APP_NAVER_CLIENT_ID: ${NAVER_CLIENT_ID}

volumes:
  postgres_data:
```

### 5-2. 운영 배포 옵션

| 서비스 | 옵션 A (무료) | 옵션 B (유료) | 권장 |
|-------|-------------|-------------|------|
| React | Vercel (무료) | AWS CloudFront | Vercel |
| SpringBoot | Render (무료 750h/월) | AWS ECS | Render (개발) → AWS (운영) |
| FastAPI | Railway (무료 500h/월) | AWS ECS | Railway (개발) → AWS (운영) |
| PostgreSQL + PostGIS | Neon (무료 500MB) | AWS RDS | Neon (개발) → RDS (운영) |
| 모델 파일 | Hugging Face Hub (무료) | AWS S3 | HF Hub |

### 5-3. 환경 변수 관리

```
# .env (로컬 개발용 — .gitignore에 추가)
DB_PASSWORD=your_password
JWT_SECRET=your_jwt_secret
OPENAI_API_KEY=sk-...
KMA_API_KEY=...
ASOS_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...           # 네이버 지역 검색 API
KAKAO_REST_API_KEY=...            # 카카오 로컬 API (맛집 POI 수집)
GOOGLE_SEARCH_API_KEY=...         # Phase 8 SNS 크롤링
HF_REPO_ID=your-name/kride-models

# Render/Railway 환경 변수: 대시보드에서 동일 변수 설정
# Vercel 환경 변수: vercel.json 또는 대시보드
```

---

## 6. 데이터 마이그레이션 계획 (CSV → PostgreSQL)

### 6-1. 마이그레이션 스크립트 순서

| 순서 | 스크립트 | 원본 | 대상 테이블 | 선행 조건 |
|-----|---------|------|-----------|---------|
| 1 | migrate_road.py | road_scored_v2.csv | road_segment | PostGIS 설치 |
| 2 | migrate_poi.py | tour_poi.csv | poi | — |
| 3 | migrate_facility.py | facility_clean.csv | facility | — |
| 4 | migrate_travel.py | AI Hub 4개 테이블 | travel_log | — |
| 5 | migrate_weather.py | weather_asos_daily.csv | weather_daily | — |
| 6 | migrate_accident.py | TAAS 사고 데이터 | road_accident | TAAS 다운로드 후 |

```python
# migrate_road.py 핵심 코드
import geopandas as gpd
from sqlalchemy import create_engine

engine = create_engine(os.environ["DATABASE_URL"])

# road_scored_v2.csv → GeoDataFrame (LineString 생성)
df = pd.read_csv("data/raw_ml/road_scored_v2.csv")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.start_lon, df.start_lat),
    crs="EPSG:4326"
)

# PostGIS에 삽입
gdf.to_postgis(
    "road_segment",
    engine,
    if_exists="replace",
    index=False,
    dtype={"geom": Geometry("POINT", srid=4326)}
)
```

### 6-2. 전국 데이터 마이그레이션 예상 용량

| 테이블 | 행 수 | 예상 용량 |
|-------|-------|---------|
| road_segment (전국) | ~15,000 | ~50MB (PostGIS 공간 인덱스 포함) |
| poi (전국) | ~250,000 | ~200MB |
| facility (전국) | ~30,000 | ~30MB |
| travel_log (전국) | ~15,000 | ~10MB |
| visit_area (전국) | ~150,000 | ~100MB |
| weather_daily | ~50,000 | ~30MB |
| road_accident | ~50,000 | ~40MB |
| **합계** | | **~460MB** |

Neon 무료 플랜(500MB)으로 전국 데이터 커버 가능.

---

## 7. API 통신 규격 (SpringBoot ↔ FastAPI)

### 7-1. 내부 통신 규격

SpringBoot가 FastAPI를 호출할 때 사용하는 내부 API 규격.

#### 경로 추천

```
POST http://fastapi:8001/recommend/route

Request:
{
  "start_lat": 37.5665,
  "start_lon": 126.9780,
  "end_lat": 37.5000,
  "end_lon": 126.9500,
  "safety_weight": 0.6,
  "tourism_weight": 0.4,
  "bike_type": "electric",
  "travel_date": "2026-04-17"
}

Response:
{
  "path": [{"lat": 37.566, "lon": 126.978}, ...],
  "distance_km": 8.5,
  "estimated_time_min": 45,
  "avg_safety_score": 0.74,
  "avg_tourism_score": 0.52,
  "predicted_weather": {"label": "맑음", "rain_prob": 0.05},
  "safety_weight_adjusted": 0.60,
  "estimated_cost_krw": 8500,
  "pois_on_route": [
    {"name": "한강공원", "category": "관광지", "lat": 37.555, "lon": 126.977}
  ],
  "facilities_on_route": [
    {"type": "수리점", "name": "서울 자전거수리소", "lat": 37.562, "lon": 126.975}
  ]
}
```

#### 소비 예측

```
POST http://fastapi:8001/predict/consume

Request:
{
  "sgg_code": "11230",
  "duration_h": 6.5,
  "distance_km": 25.3,
  "companion_cnt": 2,
  "season": 2,
  "has_lodging": false,
  "income_tier": 1,
  "age_grp": "30대",
  "gender": "F"
}

Response:
{
  "predicted_amt": 42500,
  "confidence_note": "MAE ±30,000원 / R²=0.25",
  "breakdown": {
    "transport": 8000,
    "activity": 34500
  },
  "model_version": "v2"
}
```

#### POI 추천

```
POST http://fastapi:8001/recommend/poi

Request:
{
  "seeds": ["경복궁", "창덕궁"],
  "lat": 37.5760,
  "lon": 126.9770,
  "radius_km": 15,
  "top_n": 10,
  "intent": "관광"
}

Response:
{
  "recommendations": [
    {
      "name": "북촌한옥마을",
      "jaccard_score": 0.182,
      "category": "관광지",
      "lat": 37.582,
      "lon": 126.983,
      "dist_km": 1.2,
      "attraction_score": 0.87
    }
  ],
  "model_recall_at5": 0.1260
}
```

---

## 8. 전체 구현 우선순위 타임라인

### Phase R-1: ML 즉시 개선 (이번 주) — ✅ 완료

| 작업 | 파일 | 결과 |
|------|------|------|
| ✅ 소비 모델 v2 (이상치+피처+타겟 재정의) | build_consume_model_v2.py | MAE=129,653원, R²=0.1277 (test) — v1 R²=0.0053 대비 24배 향상 |
| ✅ Co-occurrence 지리 필터 v2 | build_poi_recommender_v2.py | Recall@5=0.1372 / Recall@10=0.1811 (test, 부스트) — 베이스라인(0.0370) 대비 3.7배 향상 |
| ✅ 경기도 레저스포츠 POI 재수집 / 전국 수집 | step3_tour_collect_v2.py | 전국 17개 시도 × 4개 유형(숙박 포함) → 15,905건 수집, 오류 없음 |

#### 소비 모델 v2 실행 통계 (2026-04-16)

#### POI 추천 모델 v2 실행 통계 (2026-04-16)

| 항목 | 수치 |
| --- | --- |
| 여행 로그 입력 | 전체 21,384행 / 2,560여행 / 9,881장소 |
| 비관광 제거 후 | 14,480행 잔존 |
| 좌표 보유 장소 | 7,748개 |
| TourAPI 카테고리 연동 | 15,851개 매핑 (전국 수집 데이터 활용) |
| 학습 어휘(vocab) | 1,646개 (min_trip_freq=2 필터 후) |
| 데이터 분할 | train=1,775 / val=507 / test=254 여행 |
| Co-occurrence 행렬 | 1,646 × 1,646 / 비영 셀 14,522개 |
| Jaccard 평균(비영) | 0.1775 |
| Val Recall@5 / @10 (부스트 O) | 0.1166 / 0.1827 |
| Test Recall@5 / @10 (부스트 O) | **0.1372 / 0.1811** |
| Test Recall@5 / @10 (부스트 X) | 0.1260 / 0.1821 |
| 카테고리 부스트 효과 | Recall@5 +0.0112 (부스트 O > X) |
| 인기도 베이스라인 Recall@5 / @10 | 0.0370 / 0.0498 |
| 모델 대비 베이스라인 향상 | Recall@5 0.1372 vs 0.0370 (**3.7배**) |
| 저장 파일 | poi_cooccurrence_v2.pkl / poi_rec_meta_v2.json |

| 항목 | 수치 |
| --- | --- |
| 학습 데이터 | train=1,755 / val=376 / test=377 (총 2,508행, 51행 이상치 제거) |
| 조기종료 | epoch 114 (best epoch=94) |
| Val MAE / R² | 114,202원 / 0.1155 |
| Test MAE / R² | 129,653원 / 0.1277 |
| income_tier 분포 | 짠순이 867 (34.6%) / 보통 1427 (56.9%) / 호캉스 214 (8.5%) |
| 계절 분포 | 여름 2075 (82.7%) — 불균형 여전히 존재 (Step F 전국 데이터로 해소 예정) |

### Phase R-2: 데이터 확보 (2주차)

| 작업 | 담당 |
|------|------|
| 전국 자전거도로 data.go.kr 다운로드 | 사용자 |
| TAAS 자전거 사고 데이터 다운로드 | 사용자 |
| AI Hub 전국 여행로그 신청 | 사용자 |
| 한국관광공사 TourAPI 전국 수집 스크립트 | 작성 |
| ✅ 카카오 REST API 키 | 사용 가능 — 로컬 REST API 비즈니스 등록 불필요, FD6/CE7 반경 검색 (300,000콜/일) |
| ✅ 네이버 Client ID/Secret 발급 | 완료 — krider 앱 등록, DataLab 트렌드 API 활성화 |
| 네이버 지역 검색 API 추가 | 사용자 — krider 앱 API 설정 탭에서 '지역(검색)' 추가 신청 |
| ✅ 맛집 POI 수집 스크립트 (step3_food_collect.py) | 완료 (2026-04-16) — 카카오 FD6/CE7 반경 검색 기본, --provider naver 대체 옵션 |
| ✅ DataLab 트렌드 수집 스크립트 (step4_naver_trend.py) | 완료 (2026-04-16) — sns_mention_norm 산출, --offset 배치 지원, 5,000 POI/일 |

### Phase R-3: 전국 파이프라인 (3~4주차)

| 작업 | 파일 |
|------|------|
| 전국 자전거도로 전처리 | step2_road_clean_v2.py |
| 전국 OSM 경로 그래프 (시도별) | build_route_graph_v2.py |
| TAAS 사고 데이터 연동 | build_safety_model_v2.py |
| 전국 소비 모델 재학습 | build_consume_model_v3.py |

### Phase R-4: FastAPI 서버 구축 (4~5주차)

| 작업 | 파일 |
|------|------|
| FastAPI 기본 서버 | fastapi_server/main.py |
| 모델 로더 | fastapi_server/models/loader.py |
| 경로/안전/소비/POI 라우터 | fastapi_server/routers/*.py |
| Docker Compose 구성 | docker-compose.yml |

### Phase R-5: SpringBoot 연동 (5~6주차)

| 작업 |
|------|
| SpringBoot 프로젝트 초기화 (Spring Initializr) |
| JWT 인증/인가 구현 |
| FastAPI 프록시 연동 (WebFlux) |
| PostgreSQL + PostGIS 연동 |
| 사용자 프로파일 API |

### Phase R-6: React Frontend (6~7주차)

| 작업 |
|------|
| React 프로젝트 초기화 (Vite + TypeScript) |
| 네이버 지도 or Leaflet 지도 컴포넌트 |
| 경로 탐색 UI |
| AI 비서 채팅 UI (Phase 8 연동) |
| Vercel 배포 |

### Phase R-7: AI 비서 통합 (Phase 8, 병행)

Phase 8 (plan.md)의 kride_ai/ 모듈을 FastAPI /ai-chat 엔드포인트로 서빙.
React에서 WebSocket으로 스트리밍 응답 수신.

---

## 9. 최소 실행 환경 (배포 기준)

| 항목 | 최소 사양 | 권장 사양 |
|------|---------|---------|
| FastAPI 서버 | RAM 4GB, CPU 2코어 | RAM 8GB, CPU 4코어 |
| SpringBoot 서버 | RAM 2GB, CPU 2코어 | RAM 4GB, CPU 4코어 |
| PostgreSQL | RAM 2GB, SSD 10GB | RAM 4GB, SSD 50GB |
| LLM (Phase 8) | OpenAI API (서버리스) | — |
| 모델 파일 용량 | ~500MB (현재) | ~2GB (전국 확대 후) |

Render 무료 플랜 기준 FastAPI + SpringBoot 동시 배포 가능.
모델 파일은 HuggingFace Hub에서 서버 시작 시 자동 다운로드.

---

## 10. 폐기 / 보류 항목

| 항목 | 이유 | 대안 |
| ---- | ---- | ---- |
| GRU 방문지 시퀀스 모델 | 구조적 한계 (vocab 희소성, top5=0.14 랜덤 이하) | Co-occurrence로 대체 완료 |
| 여기어때 / 야놀자 API 연동 | 공개 API 없음, B2B 파트너 계약 필요 | 한국관광공사 숙박 TourAPI (contentTypeId=32) |
| 카카오 지도 SDK (JavaScript) | 지도 표시용 SDK는 비즈니스 신청 필요 | Naver Cloud Maps Dynamic Map으로 대체 |
| 또간집 / 망고플레이트 / 블루리본 크롤링 | 공개 API 없음, 약관 위반 위험 | 카카오 로컬 REST API (FD6/CE7) |
| K컬처 / 아이돌 브이로그 방문지 크롤링 | 우선순위 낮음, 데이터 수집 복잡도 높음 | Phase 8 이후 SNS 크롤링 모듈에서 재검토 |
| 네이버 로드뷰 CNN | 약관 금지 | AI Hub 자전거도로 이미지 데이터셋 |
| Neural CF 개인화 추천 | 데이터 부족 + 구현 복잡도 | Co-occurrence MVP → 전국 데이터 확보 후 ALS |
| TabNet Safety (Phase 5) | RF R²=0.9539으로 충분 | 전국 데이터 15,000행 확보 후 재검토 |
