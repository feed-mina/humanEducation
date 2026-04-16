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
소비 R²=0.005                  → 소비 R²≥0.25             → 소비 R²≥0.35+
날씨 Acc=73%                   → 날씨 Acc≥80%             → 실시간 LSTM 예보
Co-occ Recall@5=0.126          → Recall@5≥0.20            → Recall@5≥0.30 (전국)
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
| 4 | 피처 추가 (income, age_grp, gender, travel_purpose) | 1시간 |
| 5 | TargetEncoding(sgg_code) + 계절 가중치 | 30분 |
| 6 | TabNet 재학습 (3분할: 70/15/15) | 1~2시간 |
| 7 | 결과 비교 (old vs new) | 30분 |

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
| 한국관광공사 TourAPI (전국) | data.visitkorea.or.kr | API 키 발급 후 전국 수집 | [ ] 일부만 |
| 전국 ASOS 기상 관측소 | data.kma.go.kr | API 키 발급됨 → 관측소 확대 | [ ] 서울/경기만 |

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
    income_grp: Optional[int] = None   # 1~8 소득분위
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
GOOGLE_SEARCH_API_KEY=...   # Phase 8 SNS 크롤링
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
  "income_grp": 5,
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

### Phase R-1: ML 즉시 개선 (이번 주)

| 작업 | 파일 | 담당 |
|------|------|------|
| 소비 모델 v2 (이상치+피처+타겟 재정의) | build_consume_model_v2.py | 작성 + 사용자 실행 |
| Co-occurrence 지리 필터 v2 | build_poi_recommender_v2.py | 작성 + 사용자 실행 |
| 경기도 레저스포츠 POI 재수집 | step3_tour_collect_v2.py | 작성 + 사용자 실행 |

### Phase R-2: 데이터 확보 (2주차)

| 작업 | 담당 |
|------|------|
| 전국 자전거도로 data.go.kr 다운로드 | 사용자 |
| TAAS 자전거 사고 데이터 다운로드 | 사용자 |
| AI Hub 전국 여행로그 신청 | 사용자 |
| 한국관광공사 TourAPI 전국 수집 스크립트 | 작성 |

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
