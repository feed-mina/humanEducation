# K-Ride 개발 계획

> 업데이트: 2026-04-15 (Phase 8 AI 비서 아키텍처 추가 — LLM + LangChain + RAG + Gradio + 목적 분류 모듈)
>
> 목표: 안전점수 모델 + 관광 모델 + **경로탐지** + **편의시설 표시** + **여행 코스 추천** + **딥러닝 보강** + **LLM 지능형 AI 비서** → 통합 서비스 (Streamlit → Vercel React 연동)
>
> 딥러닝 추가 목표: KLUE-BERT 감성분석·TabNet 안전예측·CNN 도로이미지 분류를 단계적으로 도입해 모델 품질을 향상시킨다.
>
> **서비스 목표 (전체)**: 자전거 여행 + 안전성 + 경로탐지 + 관광지/여행 추천 + 자전거 이용자 편의 제공
>
> **FastAPI 서버 구현은 후순위** — 모델 파이프라인 완성 후 진행

---

## 현재까지 완료된 작업

### 머신러닝 파이프라인 (전체 완료)
- [X] 자전거도로 데이터 전처리 (`road_clean.csv`)
- [X] 편의시설 데이터 전처리 (`facility_clean.csv`)
- [X] 관광 POI 수집 및 정제 (`tour_poi.csv`)
- [X] Spatial Join → 통합 학습 데이터 생성 (`road_features.csv`)
- [X] 안전 모델 탐색 (LinearRegression, RandomForest)
- [X] 데이터 모수 문제 분석 (노선명 PK 불가 / 사고 좌표 없음 / 경기도 xls = 명세서)
- [X] `preprocess_road.py` 실행 → `road_clean_v2.csv` 생성
- [X] `build_safety_model.py` 실행 → `safety_regressor.pkl`, `safety_classifier.pkl`, `safety_scaler.pkl`, `safety_meta.pkl` 생성
- [X] safety_index_v2 설계 완료 (`(1-district_danger)×0.6 + road_attr_score×0.4`)
- [X] `build_tourism_model.py` 실행 → `tourism_scaler.pkl`, `road_scored.csv` 생성

### 경로탐지 그래프 (완료)
- [X] `build_route_graph.py` 실행 → `models/route_graph.pkl` 생성 (osmnx, 120,775 노드 / 169,136 엣지, 완전 연결)

### 딥러닝 파이프라인 (완료)
- [X] ASOS 일자료 API 키 발급 + `fetch_weather_data.py` 실행 → `weather_asos_daily.csv` (5,480행)
- [X] `build_weather_lstm.py` 실행 → `models/dl/weather_lstm.pt` 생성 (test_acc=73.28%, 계절별 그룹 + train/val/test 3분할)
- [X] `weather_kma.py` 작성 → KMA 단기예보 API 연동 모듈 (실시간 날씨 → safety_score 보정)
- [X] `build_event_ner.py` 작성 → 이벤트 NER 스크립트 (zero-shot 분류 방식)
- [X] `build_consume_model.py` 수정 완료 → AI Hub 다중 테이블 자동 머지 (`load_aihub_data()`)
  - 기본 경로: `data/ai-hub/국내 여행로그 수도권_2023/02.라벨링데이터/`
  - TRAVEL_ID 기준으로 activity_consume + travel + companion + visit_area 4개 테이블 머지

### Streamlit 앱 (Tab 1~5 구현 완료, UI 수정 대기)
- [X] Tab 1~4 기본 구현 (안전등급 예측 / 경로 추천 / 데이터 탐색 / 경로 탐색)
- [X] Tab 4 경로 탐색: A→B 최적 경로 + folium 폴리라인 오버레이
- [X] Tab 4 순환 코스 생성 + folium 지도
- [X] 편의시설 마커 레이어 (경로 탐색 시 반경 500m 내 표시)
- [X] 사이드바 날씨 연동 (2026-04-09)
  - `KMA_API_KEY` `.env` 자동 로드
  - 30분 캐시(`@st.cache_data(ttl=1800)`) 날씨 조회
  - 날씨 아이콘 / 기온 / 강수확률 / 풍속 표시
  - "날씨 기반 안전 가중치 자동 조정" 토글 → `EFFECTIVE_WEIGHTS` 반영
- [X] Tab 5 "관광지 추천" 구현 (2026-04-10) — Co-occurrence + Jaccard 기반 POI 추천, Folium 지도 연동
- [X] Tab1 district_danger → 시군구 selectbox 교체 (UI 이슈 #A — 2026-04-10 완료)
- [X] Tab2 순위 기준 caption + 도로명 컬럼 추가 (UI 이슈 #B — 2026-04-10 완료)
- [X] Tab3 관광점수 설명 expander 추가 (UI 이슈 #C — 2026-04-10 완료)
- [X] Tab5 로딩 속도 개선 (UI 이슈 #D — spinner + sorted_places 캐시 2026-04-10 완료)
- [X] 사이드바 KMA_API_KEY 경고 제거 (UI 이슈 #E — 2026-04-10 완료)

### PDF 보고서
- [X] `generate_report.py` 작성 완료 (2026-04-10) — 9페이지 landscape A4, KoPubDotum 폰트
- [ ] PDF 실행 결과 확인 및 디자인 수정

### 데이터 확보
- [X] AI Hub 국내 여행로그 수도권_2023 (2026-04-09)
  - `tn_activity_consume_his` (11,739행) / `tn_travel` (2,560행)
  - `tn_companion_info` (3,537행) / `tn_visit_area_info` (21,384행)

---

## Phase 0: 데이터 재전처리 (우선 진행)

> 기존 `road_clean.csv`는 좌표 결측으로 학습 데이터가 전국의 8.1%뿐. `preprocess_road.py` 실행으로 재전처리.

### 재전처리 핵심 변경사항

- **필터 기준 변경**: `시도명` → `시군구명` (더 세밀한 지역 분류)
- **PK 설계**: `노선명` 단독 PK 불가 (유일값 17,544 / 전체 20,262) → `노선명 + 시군구명` 복합키
- **좌표 보완**: `기점지번주소` 있는 행 → 주소 기반 fallback 처리
- **출력**: `road_clean_v2.csv`

### 모델 3개 비교 계획

| 모델 | 입력 피처 | 타겟 |
| --- | --- | --- |
| LinearRegression | width_m, length_km | safety_index |
| PolynomialRegression (degree=2) | width_m, length_km | safety_index |
| RandomForestRegressor | width_m, length_km, start_lat, start_lon | safety_index |

### 실행 파일

```bash
python kride-project/preprocess_road.py
# 출력: data/raw_ml/road_clean_v2.csv
```

---

## Phase 1: 모델 완성 (오늘)

### 1-1. 안전 모델 (Safety Model) ← 코드 완성, 실행 대기

**목표**: road segment → safety_score (0~1) + 위험등급 (0/1/2)

```text
실행: python kride-project/preprocess_road.py   (road_clean_v2.csv 생성)
      python kride-project/build_safety_model.py (pkl 4개 생성)

입력 피처: width_m, length_km, district_danger, road_attr_score [+ start_lat/lon]
타겟 (회귀): safety_index_v2 = (1-district_danger)×0.6 + road_attr_score×0.4
타겟 (분류): danger_level (0=안전/1=보통/2=위험, 삼분위 기준)
모델: RandomForestRegressor + RandomForestClassifier (class_weight="balanced")

출력:
  models/safety_regressor.pkl    ← 연속 safety_score 예측
  models/safety_classifier.pkl   ← 3등급 위험등급 예측
  models/safety_scaler.pkl       ← 추론용 MinMaxScaler
  models/safety_meta.pkl         ← features/q33/q66/R²/F1 메타
  data/raw_ml/district_danger.csv ← 구별 위험도 참고 테이블
```

### 1-2. 관광 모델 (Tourism Score) ← 다음 단계

**목표**: road segment → tourism_score (0~1)

```text
실행: python kride-project/build_tourism_model.py

입력: road_features.csv (tourist_count, cultural_count, leisure_count, facility_count)
      + safety_regressor.pkl (safety_score 계산)

방식: 규칙 기반 가중합
  raw_tourism = tourist_count × 0.5 + cultural_count × 0.3 + leisure_count × 0.2
  tourism_score = MinMaxScaler().fit_transform(raw_tourism)
  facility_bonus = min(facility_count_norm × 0.1, 0.1)
  tourism_score_final = min(tourism_score + facility_bonus, 1.0)

혼잡도 proxy: tourist_count (MVP — 실시간 없음)

출력:
  models/tourism_scaler.pkl      ← MinMaxScaler
  data/raw_ml/road_scored.csv    ← safety_score + tourism_score + final_score
```

### 1-3. 통합 점수 (Composite Score)

```python
# 사용자 선호도에 따라 가중치 조절 가능
final_score = w_safety * safety_score + w_tourism * tourism_score

# 기본값 (안전 우선)
w_safety  = 0.6
w_tourism = 0.4
```

**핵심**: 두 점수 모두 MinMaxScaler로 정규화한 뒤 합산해야 스케일 충돌 없음.

---

## Phase 2: Streamlit 앱 (오늘)

**파일**: `kride-project/streamlit_app/app.py`

### 기능 목록

1. **지도 시각화** (folium + streamlit-folium)
   - 서울+경기 도로 세그먼트를 final_score에 따라 색상 표시
   - 관광 POI 마커 오버레이
   - **편의시설 레이어** (공기 주입소, 자전거 수리점, 대여소, 화장실 마커 표시)
   - **추천 경로 오버레이** (경로탐지 결과를 지도에 폴리라인으로 표시)

2. **사이드바 컨트롤**
   - 안전 / 관광 가중치 슬라이더 (w_safety, w_tourism)
   - 지역 필터: 서울 / 경기 / 전체, **구(區) 단위로 세분화**
     - 위경도 → 구 이름 역지오코딩 또는 행정동 경계 GeoJSON 활용
     - 서울 25개 구, 경기 주요 시·구 목록 드롭다운
   - **코스 거리 필터**: 10km / 20km / 30km 선택
   - **편의시설 레이어 토글**: 체크박스로 표시/숨김

3. **추천 결과 테이블**
   - 상위 10개 세그먼트 표시 항목: 구 이름, 도로 유형(road_type), 길이(km), safety_score, tourism_score, final_score
   - 세그먼트가 "어느 구의 어떤 성격 도로인지" 한눈에 파악 가능하도록 구성

4. **여행 코스 추천 패널**
   - 시작점(위경도 또는 구 선택) 입력 → 거리 조건에 맞는 추천 코스 생성
   - 코스 내 포함된 POI(관광지, 편의시설) 목록 함께 표시

5. **모델 정보 패널**
   - 현재 모델 성능 (R², feature importance)

### 배포

- **Streamlit Cloud**: `requirements.txt` 기반 자동 배포
- GitHub `kride-project/streamlit_app/` 폴더 연결

---

## Phase 3: FastAPI + React 연동 (Vercel)

### 백엔드 (FastAPI)

```text
POST /api/recommend
  입력: { lat: float, lon: float, radius_km: float, w_safety: float, w_tourism: float }
  출력: { segments: [ { start_lat, start_lon, safety_score, tourism_score, final_score } ] }

GET /api/pois
  입력: lat, lon, radius_km
  출력: { pois: [ { title, mapx, mapy, contentTypeId } ] }

POST /api/route
  입력: { start_lat: float, start_lon: float, end_lat: float, end_lon: float,
          w_safety: float, w_tourism: float }
  출력: { path: [ { lat, lon } ], total_distance_km: float,
          avg_safety_score: float, avg_tourism_score: float,
          pois_on_route: [ { title, lat, lon } ] }

POST /api/course
  입력: { start_lat: float, start_lon: float, distance_km: float,
          w_safety: float, w_tourism: float }
  출력: { course: [ { lat, lon } ], total_distance_km: float,
          facilities_on_course: [ { type, lat, lon } ],
          pois_on_course: [ { title, lat, lon } ] }

GET /api/facilities
  입력: lat, lon, radius_km
  출력: { facilities: [ { type, name, lat, lon } ] }
```

**배포**: Railway 또는 Render (무료 플랜)

### 프론트엔드 (React + Vercel)

**지도 라이브러리 선택 기준**:

1. **네이버 지도 API 우선 시도** — 한국 도로 데이터 정확도 높음, 무료 쿼터 넉넉함 (개인 키 발급 가능)
2. **네이버 연동 실패 시 Leaflet(react-leaflet) 사용** — 오픈소스, 별도 인증 불필요
3. 카카오맵은 비즈니스 앱 등록 필요 → 현재 단계에서 제외

```text
pages/index.tsx
  └─ 네이버 지도 or react-leaflet 컴포넌트
  └─ 가중치 슬라이더 (Radix UI or shadcn)
  └─ 구 단위 필터 드롭다운
  └─ fetch('/api/recommend') → 지도 위에 세그먼트 오버레이
```

**배포**: Vercel (GitHub 연동 자동 배포)

### CORS 설정 (FastAPI)

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["https://kride.vercel.app"])
```

---

## 아키텍처 요약

```text
[road_features.csv]  [facility_clean.csv]  [tour_poi.csv]
    │                        │                    │
    ├─ Safety Model (RF) ──┐  └── 편의시설 레이어   └── POI 레이어
    └─ Tourism Scaler ─────┤
                           ↓
                   Composite Score (road_scored.csv)
                           │
                   Route Graph (.pkl)  ← networkx 그래프
                           │
          ┌────────────────┼──────────────────────┐
          ↓                ↓                      ↓
   Streamlit App     FastAPI Server          (추후) DB 연동
   (Streamlit Cloud) (Railway/Render)        (PostgreSQL/PostGIS)
   - 지도 시각화        /api/recommend
   - 편의시설 레이어    /api/route          ← 경로탐지
   - 경로 오버레이      /api/course         ← 여행 코스
   - 코스 추천          /api/facilities
   - 날씨 표시          /api/pois
                           │
               날씨 API (KMA) 연동
                           │
               네이버지도 or Leaflet
                 React Frontend
                    (Vercel)
```

---

## Phase 3-5: 경로탐지 엔진 (Route Planning) ← 핵심 누락 기능

> **목표**: 도로 세그먼트 그래프를 구성하고 final_score 기반 최적 경로 탐색 → 자전거 여행 코스 생성

### 3-5-1. 그래프 구성 (networkx)

```python
import networkx as nx

G = nx.Graph()
for _, row in road_scored.iterrows():
    # 엣지 가중치: final_score 높을수록 우선 경로 (1 - score)
    G.add_edge(
        (row.start_lat, row.start_lon),
        (row.end_lat, row.end_lon),
        weight=1 - row.final_score,
        road_id=row.road_id,
        safety_score=row.safety_score,
        tourism_score=row.tourism_score,
        length_km=row.length_km
    )
```

**전제 조건**: `road_scored.csv`에 `end_lat` / `end_lon` 컬럼 존재 여부 확인 필요.

- 있을 경우: 바로 그래프 구성 가능
- 없을 경우: `road_clean_v2.csv`의 `종점위도` / `종점경도` 컬럼을 조인해서 보완

### 3-5-2. 최적 경로 탐색

```python
import heapq

def find_best_route(G, start_coord, end_coord, w_safety=0.6, w_tourism=0.4):
    """
    start_coord, end_coord: (lat, lon) 튜플
    가중치를 w_safety, w_tourism에 맞게 재계산 후 Dijkstra 탐색
    """
    # 입력 좌표와 가장 가까운 노드 탐색
    start_node = min(G.nodes, key=lambda n: haversine(n, start_coord))
    end_node   = min(G.nodes, key=lambda n: haversine(n, end_coord))

    # 엣지 가중치 재계산 (사용자 가중치 반영)
    for u, v, data in G.edges(data=True):
        data["weight"] = 1 - (w_safety * data["safety_score"]
                              + w_tourism * data["tourism_score"])

    path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
    return path
```

### 3-5-3. 여행 코스 생성 (거리 기반 순환 코스)

```python
def generate_course(G, start_coord, target_km=20, w_safety=0.6, w_tourism=0.4):
    """
    시작점 주변에서 final_score 높은 세그먼트를 따라
    target_km에 근접하는 순환 코스 생성 (DFS 기반)
    """
    start_node = min(G.nodes, key=lambda n: haversine(n, start_coord))
    course = []
    total_km = 0.0
    visited = set()
    stack = [(start_node, [start_node], 0.0)]

    while stack:
        node, path, dist = stack.pop()
        if dist >= target_km:
            course = path
            break
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                edge_data = G[node][neighbor]
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor], dist + edge_data["length_km"]))

    return course
```

```text
출력:
  models/route_graph.pkl   ← 사전 구성된 networkx 그래프 (서버 시작 시 로드)
  실행 파일: kride-project/build_route_graph.py
```

### 3-5-4. 경로 위 편의시설 / POI 조회

```python
def get_facilities_on_route(path_coords, facility_df, radius_m=500):
    """경로 좌표 목록 기준 반경 500m 내 편의시설 반환"""
    results = []
    for coord in path_coords:
        nearby = facility_df[
            facility_df.apply(
                lambda row: haversine(coord, (row.lat, row.lon)) <= radius_m / 1000, axis=1
            )
        ]
        results.append(nearby)
    return pd.concat(results).drop_duplicates()
```

### Task (Phase 3-5)

1. [X] `road_scored.csv`에 `end_lat` / `end_lon` 포함 여부 확인 → 19컬럼 모두 존재, 결측 없음
2. [X] `build_route_graph.py` 실행 → `models/route_graph.pkl` 생성 완료 (2026-04-08)
3. [X] **그래프 단절 문제 해결** — osmnx 기반으로 재설계, 연결 컴포넌트 1개(100%) 달성 (2026-04-08)
   - 원인: 공공 자전거도로 행정 데이터는 교차점 토폴로지 없음 → osmnx(OSM 자전거 네트워크)로 전환
   - 결과: 120,775 노드 / 169,136 엣지, 완전 연결 그래프
4. [ ] (후순위) FastAPI `/api/route` 엔드포인트 작성
5. [ ] (후순위) FastAPI `/api/course` 엔드포인트 작성
6. [ ] (후순위) FastAPI `/api/facilities` 엔드포인트 작성
7. [X] Streamlit 앱에 경로 입력 UI + folium 폴리라인 오버레이 추가 (Tab 4 구현 완료)
8. [X] Streamlit 앱에 편의시설 마커 레이어 추가 (경로 탐색 결과에 반경 500m 마커 표시)
9. [ ] Streamlit 독립 편의시설 레이어 토글 (경로 무관하게 전체 지도 위에 표시)

---

## Phase 3-6: 날씨 연동 (안전 가중치 동적 조정)

> **목표**: 기상청 단기예보 API → 강수확률/풍속에 따라 w_safety 자동 상향 → 악천후 시 안전 우선 경로 추천

```python
import requests

def get_weather_weight(lat, lon):
    """
    기상청 단기예보 API (KMA Open API) 호출
    강수확률(POP) 기반으로 안전 가중치 보정값 반환
    """
    # KMA API 호출 (base_date, base_time, nx, ny 필요)
    rain_prob = fetch_kma_pop(lat, lon)  # 0~100
    w_safety_adj = min(0.6 + (rain_prob / 100) * 0.2, 0.8)  # 최대 0.8
    w_tourism_adj = 1.0 - w_safety_adj
    return w_safety_adj, w_tourism_adj

# 사용 예시
# 비 올 확률 70% → w_safety=0.74, w_tourism=0.26
# 비 올 확률 0%  → w_safety=0.60, w_tourism=0.40
```

```text
Task:
  [X] KMA 단기예보 API 연동 모듈 작성 (weather_kma.py) — latlon_to_grid + get_weather_weight 포함
  [X] KMA_API_KEY .env 저장 + Streamlit 자동 로드 (2026-04-09)
  [X] Streamlit 앱 사이드바에 현재 날씨 상태 표시 — 날씨 아이콘/기온/강수확률/풍속 + 자동 가중치 조정 토글 (2026-04-09)
  [ ] (후순위) FastAPI /api/recommend, /api/route에 날씨 보정 파라미터 추가
```

> **Phase 3-6 확장**: 단순 API 연동을 넘어 LSTM 시계열 모델로 날씨를 예측하는 방안은 Phase 3-8 참조.

---

## Phase 3-7: 이벤트 감지 → 경로 혼잡도 실시간 알림

> **목표**: 스포츠 경기·아이돌 콘서트·재난 뉴스를 감지하여 경로 근처 이벤트 알림 제공  
> **데이터**: AI Hub 생성형AI 한국어 다중 이벤트 추출 데이터 (17,625건, dataSetSn=71729)

### 3-7-1. 이벤트 NER 파이프라인

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# STEP 1: KLUE-BERT NER 파인튜닝 (이벤트 추출 데이터 활용)
# 라벨: O / B-LOC / I-LOC / B-EVT / I-EVT
model = AutoModelForTokenClassification.from_pretrained(
    "klue/bert-base", num_labels=5
)
# entity_value(장소명) + event_type → NER 학습 데이터 변환 후 파인튜닝

# STEP 2 (빠른 대안): zero-shot classification으로 이벤트 유형 분류
classifier = pipeline("zero-shot-classification", model="snunlp/KR-FinBert-SC")
event_types = ["스포츠_경기", "공연_행사", "재난_사고", "도로_공사", "기타"]
result = classifier(news_text, candidate_labels=event_types)
```

### 3-7-2. 장소 → 경로 영향 계산

```python
# 이벤트 장소명 → Geocoding → lat/lon
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="kride")
location = geolocator.geocode("잠실종합운동장")

# 경로 세그먼트 반경 1km 이내 이벤트 감지
event_impact = {
    "스포츠_경기":  {"w_tourism": -0.2, "congestion_alert": True},
    "공연_행사":    {"w_tourism": -0.2, "congestion_alert": True},
    "재난_사고":    {"safety_score": -0.3, "danger_alert": True},
    "도로_공사":    {"safety_score": -0.15, "detour_alert": True},
}
```

### 3-7-3. FastAPI 엔드포인트

```text
GET /api/events
  입력: lat, lon, radius_km, date (YYYY-MM-DD)
  출력: { events: [ { type, title, venue, lat, lon, impact_level } ] }

# /api/route 응답에 events_on_route 필드 추가
```

### 3-7-4. Streamlit UI

```python
# 경로 지도 위 이벤트 마커 (경고색 표시)
event_layer = folium.FeatureGroup(name="이벤트 알림")
for evt in events_on_route:
    folium.Marker(
        [evt.lat, evt.lon],
        icon=folium.Icon(color="red", icon="exclamation-sign"),
        tooltip=f"{evt.type}: {evt.title}"
    ).add_to(event_layer)
```

### Task (Phase 3-7)

```text
  [X] build_event_ner.py 작성 완료 — zero-shot (mDeBERTa) + geopy 파이프라인 포함
  [ ] (데이터 필요) 이벤트 추출 데이터 (AI Hub) 다운 → 파인튜닝 시 활용
  [ ] (후순위) FastAPI /api/events 엔드포인트 작성
  [ ] (후순위) /api/route 응답에 events_on_route 필드 추가
  [ ] Streamlit 이벤트 알림 레이어 추가
```

---

## Phase 3-8: 날씨 LSTM 예측 (날짜 + 경로 → 날씨 예측)

> **목표**: 사용자가 여행 날짜를 입력하면 경로별 예상 날씨를 LSTM으로 예측 → safety_score 자동 보정  
> **데이터**: 기상청 과거 날씨 CSV (data.kma.go.kr, 무료, 즉시 다운)

### 3-8-1. 데이터 준비

```text
기상자료개방포털 (data.kma.go.kr)
  → 지상관측자료 → 서울/경기 주요 관측소
  → 일별 기온/강수량/풍속/습도 CSV (최소 3년치 권장)

시군구 매핑:
  관측소 코드 → 시군구명 → road_features.csv sgg_code와 연결
```

### 3-8-2. LSTM 날씨 예측 모델

```python
import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 3)  # 맑음=0 / 흐림=1 / 비·눈=2

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 입력 시퀀스: 과거 14일치
# [월, 일, 요일, 기온, 강수량, 풍속, 습도, 시군구코드_임베딩]
# 출력: 다음 날 날씨 3분류
```

### 3-8-3. safety_score 날씨 보정

```python
WEATHER_PENALTY = {0: 0.0, 1: -0.1, 2: -0.3}  # 맑음/흐림/비눈

def adjust_safety_for_weather(safety_score, predicted_weather):
    return max(safety_score + WEATHER_PENALTY[predicted_weather], 0.0)

# Streamlit UI: "선택 날짜 예상 날씨: 비 ☔ → 안전점수 자동 하향 조정"
```

### 3-8-4. FastAPI 엔드포인트 확장

```text
POST /api/route (기존) → 파라미터에 travel_date 추가
  입력 추가: { travel_date: "2026-04-15" }
  출력 추가: { predicted_weather: "비", safety_score_adjusted: 0.62 }

GET /api/weather_forecast
  입력: sgg_code, travel_date
  출력: { weather: "흐림", temp_avg: 14.2, rain_prob: 0.45 }
```

### Task (Phase 3-8)

```text
  [X] 공공데이터포털 ASOS 일자료 API 키 발급 (2026-04-08)
       - API: 기상청_지상(중관,ASOS) 일자료 조회서비스
       - EndPoint: https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList
       - 관측소: 서울(108), 수원(119), 인천(112), 양평(202), 이천(203)
  [X] fetch_weather_data.py 작성 완료 → data/dl/kma_weather_raw/weather_asos_daily.csv
  [X] WeatherLSTM 학습 스크립트 작성 (build_weather_lstm.py) — 기존 파일 존재
  [X] safety_score 날씨 보정 함수 작성 (weather_kma.py에 포함)
  [X] fetch_weather_data.py API 403 오류 수정 — serviceKey URL 직접 삽입 방식으로 변경 (2026-04-08)
       원인: requests params={}로 전달 시 자동 URL 인코딩 → 공공데이터포털 인증 거부
       해결: url = f"{API_URL}?serviceKey={api_key}" 후 나머지만 params로 전달
  [X] .env 파일 생성 (ASOS_API_KEY 저장) + .gitignore에 추가
  [X] fetch_weather_data.py 재실행 → weather_asos_daily.csv 생성 확인 (2026-04-08, 5,480행)
  [X] build_weather_lstm.py 실행 → models/dl/weather_lstm.pt 생성
      test_acc=73.28% / 계절별 층화 3분할(70/20/10) / 클래스 가중치 CrossEntropyLoss 적용
      ※ 초기 val_acc 79.43%는 test set 미분리 상태였음. 73.28%가 실제 일반화 성능.
  [ ] (후순위) FastAPI /api/weather_forecast 엔드포인트 작성
  [ ] (후순위) Streamlit 날짜 선택 UI + 예상 날씨 표시 추가
```

---

## Phase 3-9: 소비 패턴 예측 (경로별 예상 지출)

> **목표**: 여행로그 소비 데이터로 경로 선택 시 예상 지출 금액 제공  
> **데이터**: AI Hub 여행로그(수도권) TL_csv.zip (2.89MB) — TN_ACTIVITY_CONSUME_HIS 테이블

### 3-9-1. 소비 예측 모델 (TabNet Regressor)

```python
from pytorch_tabnet.tab_model import TabNetRegressor

# 입력 피처
features = [
    'sgg_code',           # 시군구 코드 (TC_SGG)
    'travel_duration_h',  # 여행 시간 (시간 단위)
    'distance_km',        # 이동 거리
    'companion_cnt',      # 동반자 수
    'season',             # 계절 (1~4)
    'day_of_week',        # 요일 (0~6)
    'has_lodging',        # 숙박 여부 (0/1)
]
# 타겟: 활동 소비 금액 (TN_ACTIVITY_CONSUME_HIS)

regressor = TabNetRegressor(n_d=16, n_a=16, n_steps=3)
regressor.fit(X_train, y_train.reshape(-1, 1))
```

### 3-9-2. 경로 추천 카드 통합

```python
# 경로 추천 결과에 예상 지출 추가
route_card = {
    "path": [...],
    "avg_safety_score": 0.74,
    "avg_tourism_score": 0.61,
    "predicted_weather": "흐림",
    "events_nearby": ["잠실야구장 경기 18:00"],
    "estimated_cost_krw": 12500,   # ← 소비 예측 TabNet 출력
    "cost_breakdown": {
        "transport": 3000,
        "activity": 9500
    }
}
```

### Task (Phase 3-9)

```text
  [X] build_consume_model.py 작성 완료 (--use_dummy 플래그로 합성 데이터 대체 가능)
  [X] AI Hub 여행로그(수도권) 데이터 확보 (2026-04-09)
      → data/ai-hub/국내 여행로그 수도권_2023/02.라벨링데이터/ (activity: 11,739행)
  [X] build_consume_model.py 수정 → load_aihub_data() 다중 테이블 머지 지원 (2026-04-09)
  [X] build_consume_model.py 실행 → models/consume_regressor.zip 생성 (2026-04-09)
      학습 결과: 2,530행 / best_epoch=51 / MAE=125,302원 / R²=0.0053
      (R² 낮음 → 소비금액 분산이 크고 특성 수 부족, 향후 특성 추가로 개선 가능)
  [X] Streamlit 경로 결과 카드에 예상 지출 표시 (2026-04-09)
      A→B 경로 + 순환 코스 양쪽에 "예상 지출 (참고)" 카드 추가
      동반자 수 입력 연동, cached_predict_consume() 캐시 래퍼 사용
  [ ] (후순위) FastAPI /api/route 응답에 estimated_cost 필드 추가
```

---

## 아키텍처 요약 (전체 업데이트)

```text
[사용자 입력]
출발지 + 도착지 + 날짜 + 가중치

        ┌──────────────────────────────────────────────┐
        │              입력 전처리                      │
        └──┬──────────────┬──────────────┬─────────────┘
           ▼              ▼              ▼
   날씨 LSTM 예측    이벤트 NER 감지   소비 TabNet 예측
   (맑음/흐림/비)   (스포츠/콘서트    (예상 지출 금액)
                    /재난/공사)
           │              │              │
   safety_score     혼잡/위험         비용 카드
   자동 보정         경고 알림
           │              │              │
           └──────────────┼──────────────┘
                          ▼
              Route Graph (networkx)
              Dijkstra 최적 경로 탐색
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   Streamlit App     FastAPI Server    React (Vercel)
   - 경로 오버레이   /api/route        네이버지도
   - 이벤트 마커     /api/events       경로 표시
   - 날씨 표시       /api/weather      통합 정보 카드
   - 예상 지출       /api/course
   - 편의시설        /api/facilities
```

---

## Phase 3-10: 관광 딥러닝 고도화 (AI Hub 여행로그 기반) ← 신규 (2026-04-09)

> **목표**: 이미 보유한 AI Hub 여행로그 데이터로 tourism_score를 딥러닝 기반으로 고도화.
> 현재 tourism_score는 POI 밀도 규칙 기반이며, 실제 방문자 만족도·재방문 의향이 반영되지 않음.

### 현재 tourism_score 한계

```text
현재 계산식 (build_tourism_model.py):
  raw = tourist_count × 0.5 + cultural_count × 0.3 + leisure_count × 0.2
  tourism_score = MinMaxScaler(raw) + facility_bonus (cap 0.1)

문제점:
  - POI 개수 기반 → 실제 방문자 경험 미반영
  - 카테고리 가중치가 임의 설정 (0.5/0.3/0.2)
  - 계절·시간대·인구통계 무시
```

---

### 3-10-1. POI 매력도 예측 (TabNet) ← 즉시 구현 가능

**데이터**: `tn_visit_area_info_방문지정보_E.csv` (21,384행, 이미 보유)

```text
타겟 (매력도 점수 합산):
  DGSTFN          → 만족도 점수
  REVISIT_INTENTION → 재방문 의향
  RCMDTN_INTENTION  → 추천 의향
  → attraction_score = (DGSTFN + REVISIT_INTENTION + RCMDTN_INTENTION) / 3  (정규화)

입력 피처:
  VISIT_AREA_TYPE_CD  → 방문지 유형 (음식점/관광지/자연 등)
  RESIDENCE_TIME_MIN  → 체류 시간 (오래 머물수록 매력적)
  SGG_CD              → 시군구 코드
  VISIT_CHC_REASON_CD → 방문 선택 이유
  VISIT_ORDER         → 여행 내 방문 순서 (첫 방문지 vs 마지막)
  X_COORD, Y_COORD    → 위치 (공간 클러스터링)
```

```python
# build_attraction_model.py 핵심 구조
from pytorch_tabnet.tab_model import TabNetRegressor

# 타겟: 매력도 점수 (0~5 정규화)
target = "attraction_score"

regressor = TabNetRegressor(n_d=16, n_a=16, n_steps=3)
regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# 출력: POI좌표 → attraction_score 맵핑
# → Spatial Join으로 도로 세그먼트 주변 attraction_score 집계
# → tourism_score_v2 = 0.7 × 기존 tourism_score + 0.3 × attraction_score
```

```text
출력 파일:
  models/attraction_regressor.zip   ← TabNet 가중치
  models/attraction_scaler.pkl      ← 피처 스케일러
  data/raw_ml/poi_attraction.csv    ← POI별 attraction_score (X_COORD, Y_COORD 포함)
  data/raw_ml/road_scored_v2.csv    ← tourism_score_v2 반영 업데이트
```

```text
Task:
  [X] build_attraction_model.py 작성 (2026-04-09)
  [X] build_attraction_model.py 실행 → attraction_regressor.zip 생성 (2026-04-09)
      MAE=0.6558 / R²=0.0662 / best_epoch=31 / POI 8,454개
  [X] build_tourism_score_v2.py 실행 → road_scored_v2.csv 생성 (2026-04-09)
      1,647 세그먼트 / 902개 POI 매칭(54.8%) / tourism_score +0.08 / final_score +0.03
  [X] route_graph.pkl 재빌드 (2026-04-09, v2 점수 반영)
      172,656 노드 / 238,962 엣지 / 완전 연결 1개 컴포넌트
  [ ] Streamlit 경로 카드에 "관광 매력도 v2" 점수 표시
```

---

### 3-10-2. 방문지 시퀀스 추천 (LSTM) ← 순환 코스 고도화

**데이터**: `tn_visit_area_info_방문지정보_E.csv`의 `VISIT_ORDER` 시퀀스

```text
목표: "이 경로에서 출발하면 다음에 많이 방문하는 POI는?"
     → 순환 코스 생성 시 "추천 경유지" 자동 삽입

학습 입력:
  TRAVEL_ID별 VISIT_ORDER 순서로 정렬된 POI 시퀀스
  예: [강남역 → 코엑스 → 봉은사 → 선릉] → 다음 POI 예측

모델: GRU (시퀀스 짧아서 LSTM보다 가벼움)
  Embedding(poi_id, 32) → GRU(64) → Dense(poi_vocab_size) → Softmax
```

```text
출력 파일:
  models/dl/visit_seq_gru.pt    ← GRU 가중치
  models/dl/poi_encoder.pkl     ← POI ID → 인덱스 인코더
  data/dl/visit_sequences.csv   ← 전처리된 시퀀스 데이터
```

```text
Task:
  [ ] build_visit_sequence_model.py 작성 (GRU 시퀀스 예측)
  [ ] Streamlit 순환 코스 탭에 "추천 경유지 POI" 표시
```

---

### 3-10-3. 여행자 스타일 기반 개인화 (후순위)

**데이터**: `tn_traveller_master_여행객 Master_E.csv` (35컬럼)

```text
활용 가능 피처:
  GENDER, AGE_GRP, INCOME, HOUSE_INCOME    → 인구통계
  TRAVEL_STYL_1~8                          → 8가지 여행 스타일 점수
  TRAVEL_LIKE_SIDO_1~3, TRAVEL_LIKE_SGG_1~3 → 선호 지역
  TRAVEL_MOTIVE_1~3                        → 여행 동기

모델 방향: Neural CF (사용자 임베딩 + POI 임베딩)
→ 사용자 스타일 입력 → 맞춤 POI 추천
→ 경로 선택 시 "나와 비슷한 사람들이 선호한 경로" 표시

난이도: 높음 (사용자 세션 관리 필요)
우선순위: 후순위 (Phase 5 이후)
```

---

### 3-10 구현 우선순위 및 일정

| 작업 | 난이도 | 데이터 | 우선순위 |
|------|--------|--------|----------|
| POI 매력도 TabNet (3-10-1) | 낮음 | ✅ 보유 | 1순위 |
| 방문지 시퀀스 GRU (3-10-2) | 중간 | ✅ 보유 | 2순위 |
| 감성 분석 KLUE-BERT (Phase 4) | 중간 | ❌ 수집 필요 | 3순위 |
| 여행자 개인화 Neural CF (3-10-3) | 높음 | ✅ 보유 | 4순위 (후순위) |

---

## Phase 4: 딥러닝 — NLP 감성 분석 (tourism_score 정교화) ← 1순위

> **목표**: 네이버·카카오맵 리뷰 텍스트를 KLUE-BERT로 감성 분석 → `sentiment_weight` 산출 → `tourism_score` 보정

### 4-1. 리뷰 데이터 수집

```text
수집 대상:
  - 네이버 지도 검색 API: 관광지·자전거도로 주변 장소 → 별점·리뷰 수
  - 카카오 로컬 API: 키워드 검색 → 장소 정보
  - Selenium/Playwright: 공식 API 미제공 리뷰 텍스트 크롤링 (robots.txt 준수)

저장 경로:
  data/dl/review_raw.csv   ← 수집 원본
  data/dl/review_clean.csv ← 전처리 후 (결측·중복 제거, 한글 정규화)
```

### 4-2. KLUE-BERT 감성 분석 모델

```python
# 사전학습 모델 (추가 학습 없이 zero-shot 활용 가능)
from transformers import pipeline

sentiment = pipeline(
    "text-classification",
    model="snunlp/KR-FinBert-SC"   # 한국어 감성 분류
)

# 리뷰 텍스트 → 긍정 확률
result = sentiment("이 자전거길 정말 넓고 쾌적해요!")
# → [{'label': 'positive', 'score': 0.94}]

# tourism_score 보정
sentiment_weight = positive_prob * 0.2   # 최대 20% 보정
tourism_score_dl = min(tourism_score_final + sentiment_weight, 1.0)
```

```text
출력:
  models/dl/sentiment_pipeline/   ← Hugging Face pipeline 저장
  data/dl/review_sentiment.csv    ← poi_title + sentiment_score
  data/raw_ml/road_scored_v2.csv  ← tourism_score_dl 컬럼 추가
```

### 4-3. 감성 분석 연동 파이프라인

```text
review_clean.csv
  └─ KLUE-BERT (snunlp/KR-FinBert-SC)
        └─ review_sentiment.csv
              └─ Spatial Join (POI 좌표 ↔ 도로 세그먼트 1km 반경)
                    └─ sentiment_score → tourism_score_dl 보정
```

**실행 파일**: `kride-project/build_sentiment_model.py`

---

## Phase 5: 딥러닝 — TabNet 안전 예측 (safety_score 고도화) ← 2순위

> **목표**: RandomForest(R²=0.95) 대체 또는 앙상블용으로 TabNet 도입 → 구조화 데이터에서 더 정밀한 경계 학습

### 5-1. 모델 선택 근거

| 항목 | RandomForest | TabNet |
| --- | --- | --- |
| 구조화 데이터 처리 | ✅ 강점 | ✅ 강점 (딥러닝 방식) |
| 피처 중요도 해석 | ✅ 직관적 | ✅ attention map 제공 |
| 위경도 비선형 학습 | 보통 | **우수** (attention 메커니즘) |
| 학습 데이터 요구량 | 적음 | 중간 (1,647행 → 증강 필요) |
| 현재 단계 적합성 | MVP 완료 | **고도화 단계 적합** |

### 5-2. TabNet 구현

```python
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

# 회귀 모델
regressor = TabNetRegressor(
    n_d=16, n_a=16,          # 임베딩 차원
    n_steps=5,               # attention 단계
    gamma=1.5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2)
)
regressor.fit(X_train, y_train.reshape(-1, 1),
              eval_set=[(X_val, y_val.reshape(-1, 1))])

# 입력 피처: width_m, length_km, district_danger, road_attr_score, start_lat, start_lon
# 타겟:      safety_index_v2 (연속값)
```

```text
출력:
  models/dl/tabnet_safety_regressor.zip    ← TabNet 가중치
  models/dl/tabnet_safety_classifier.zip   ← 3등급 분류
  models/safety_meta_v2.pkl                ← RF + TabNet 앙상블 메타
```

**실행 파일**: `kride-project/build_safety_model_dl.py`

### 5-3. RF + TabNet 앙상블 전략

```python
# 소프트 앙상블 (가중 평균)
final_safety = 0.5 * rf_pred + 0.5 * tabnet_pred
```

---

## Phase 6: 딥러닝 — CNN 도로 이미지 분류 (safety_score 보완) ← 3순위

> **목표**: 도로 이미지를 CNN으로 분류 → 도로 포장 상태·너비·장애물 여부 피처화

> ⚠️ **2026-04-09 조사 결과**: 네이버 로드뷰는 이미지 추출용 REST API를 공식 제공하지 않으며 이용약관상 이미지 저장·ML 학습이 금지됨 → **AI Hub 자전거도로 데이터셋으로 대체**

### 6-1. 데이터 수집

```text
❌ 제외: 네이버 로드뷰
  - 이미지 추출 REST API 미존재 (JavaScript 뷰어만 제공)
  - 이용약관: 이미지 저장·AI 학습 목적 사용 금지 (저작권 보호)

✅ 대체 1순위: AI Hub — 자전거도로 주행 데이터 (무료, 신청 후 사용)
  URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71629
  - 한국 자전거도로 특화 이미지 (장애물·노면 상태·편의시설 포함)
  - good/normal/bad 3등급 라벨 구조와 직접 매핑 가능
  - 공식 정부 AI 학습 데이터셋, 연구 목적 승인 가능성 높음

✅ 대체 2순위: NAVER LABS HD Map Dataset (CC-BY-NC-SA 4.0, 무료)
  URL: https://hdmap.naverlabs.com/
  - 서울 지역 (판교·상암·여의도·마곡) 3D 도로 구조 + LiDAR 포함
  - 비상업적 연구/교육 목적 허용

✅ 대체 3순위: Mapillary Vistas (전이학습 보조용, 즉시 다운로드)
  - 오픈소스 스트리트뷰 이미지 25,000장, semantic segmentation 라벨

저장 경로: data/dl/roadview_images/
  ├─ good/    ← 포장 양호, 장애물 없음
  ├─ normal/  ← 보통
  └─ bad/     ← 파손·협소·장애물
```

### 6-2. 전이학습 (EfficientNet-B0)

```python
import torchvision.models as models

# 사전학습 가중치 활용 (ImageNet)
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 3)   # 3등급 분류

# 학습: 직접 라벨링 또는 GPT-4V 자동 라벨링 활용
# 입력: 224×224 RGB 이미지
# 출력: road_condition (good/normal/bad)
```

```text
출력:
  models/dl/cnn_road_condition.pt   ← 가중치
  data/dl/road_condition.csv        ← road_id + road_condition_score
  → safety_index에 +road_condition_score×0.1 반영
```

**실행 파일**: `kride-project/build_road_image_model.py`

---

## 딥러닝 아키텍처 (전체 개요)

```text
[텍스트 데이터]   [구조화 데이터]         [이미지 데이터]
리뷰 텍스트        도로 피처 CSV       AI Hub 자전거도로 이미지
    │                   │                (Naver 로드뷰 ❌ 약관)
KLUE-BERT            TabNet               EfficientNet-B0
(감성분석)          (안전예측)              (도로상태분류)
    │                   │                       │
sentiment_weight   safety_score_dl      road_condition_score
    └──────────────┬────┘                       │
                   ▼                             │
            Composite Score ◄────────────────────┘
           (최종 추천 점수)
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
Streamlit       FastAPI       (추후) DB
```

---

## 전체 진행 현황 (2026-04-09 기준)

### 완료된 작업

| 단계 | 파일 | 결과 |
|------|------|------|
| ML 파이프라인 전체 | preprocess_road.py → build_safety_model.py → build_tourism_model.py | road_scored.csv 생성 |
| 경로 그래프 | build_route_graph.py (osmnx, simplify=False, 서울 bbox) | route_graph.pkl 120,767 노드 / 169,118 엣지, 100% 연결 |
| 날씨 LSTM | build_weather_lstm.py | weather_lstm.pt, test_acc=73.28%, 계절별 3분할 |
| 소비 TabNet | build_consume_model.py | consume_regressor.zip, MAE=125,302원 |
| 이벤트 NER | build_event_ner.py | zero-shot 파이프라인 완성, 데이터 수집 대기 |
| POI 매력도 TabNet | build_attraction_model.py | poi_attraction.csv (8,454 POI), MAE=0.6558 |
| tourism_score_v2 스크립트 | build_tourism_score_v2.py | 작성 완료, **실행 대기** |
| Streamlit 앱 | streamlit_kride.py | Tab 1~4, 날씨 연동, 소비 예측 카드 포함 |

---

## 다음 진행 순서 (즉시 실행 가능한 것부터)

### [사용자 실행] Step A — tourism_score_v2 파이프라인 완성

```bash
# 1. Spatial Join → road_scored_v2.csv 생성 (30초~2분)
python kride-project/build_tourism_score_v2.py

# 2. v2 기반 route_graph.pkl 재빌드 (OSM 캐시 있으므로 빠름)
python kride-project/build_route_graph.py
```

완료 후 체크:
```text
[ ] road_scored_v2.csv 생성 확인 (tourism_score_v2, final_score_v2 컬럼 포함)
[ ] route_graph.pkl 재빌드 (소스: road_scored_v2.csv, "tourism_score_v2 사용" 메시지 확인)
```

---

### [Claude 작성] Step B — build_visit_sequence_model.py (GRU 방문지 시퀀스)

```text
우선순위: 2순위 (데이터 보유, 즉시 구현 가능)
목표: TRAVEL_ID별 VISIT_ORDER 시퀀스 → 다음 방문지 예측
모델: Embedding(poi_id, 32) → GRU(64) → Dense(poi_vocab)
분할: 계절별 또는 TRAVEL_ID 기준 70/15/15 3분할 (DL 품질 기준 준수)
출력: models/dl/visit_seq_gru.pt, models/dl/poi_encoder.pkl
```

---

### [Claude 작성] Step C — Streamlit 개인화 경로 추천 UI (Phase 3-11)

```text
우선순위: 3순위 (route_graph.pkl v2 완성 후)
목표: 사이드바 "내 프로파일" 섹션 + BIKE_PROFILES 가중치 로직
변경 파일: streamlit_kride.py
```

---

### [후순위] Step D — FastAPI 서버 / React 연동

```text
모든 모델 파이프라인 안정화 후 진행
```

---

## Task 순서 (원본, 업데이트됨)

1. [X] `preprocess_road.py` 코드 작성
2. [X] `build_safety_model.py` 코드 작성 (safety_index_v2 + district_danger 반영)
3. [X] `build_tourism_model.py` 코드 작성 (tourism_score + final_score + road_scored.csv)
4. [X] `preprocess_road.py` 실행 → `road_clean_v2.csv` 생성 확인
   - LinearRegression R²=1.0 (leakage — safety_index가 width/length 선형조합이므로 당연)
   - RandomForest R²=0.859 (피처중요도: width_m 0.46 > start_lat 0.34 > start_lon 0.18 > length_km 0.02)
5. [X] `build_safety_model.py` 실행 → pkl 4개 + district_danger.csv 확인
   - 회귀 R²=0.9539 / 분류 F1-macro=0.9864
6. [X] `build_tourism_model.py` 실행 → road_scored.csv 확인 (1,647행 × 19컬럼)
7. [X] `build_route_graph.py` 작성 및 실행 → `models/route_graph.pkl` 생성
   - osmnx 서울 bbox(37.413~37.715N, 126.764~127.185E), simplify=False
   - 120,767 노드 / 169,118 엣지, 연결 컴포넌트 1개(100%)
8. [X] `streamlit_kride.py` 작성 — Tab 1~4 + 날씨 연동 + 소비 예측 카드
9. [ ] `build_tourism_score_v2.py` 실행 → `road_scored_v2.csv` 생성 ← **현재 단계**
10. [ ] `build_route_graph.py` 재실행 → route_graph.pkl v2 (tourism_score_v2 반영)
11. [ ] `build_visit_sequence_model.py` 작성 및 실행 (GRU 방문지 시퀀스)
12. [ ] Streamlit 개인화 경로 추천 UI 추가 (Phase 3-11)
13. [ ] FastAPI `ml-server/main.py` 작성 (후순위)
14. [ ] React 연동 — 네이버 지도 (후순위)

### 딥러닝 Task (Phase 4~6, 후순위)

1. [ ] 리뷰 데이터 수집 → `build_sentiment_model.py` (KLUE-BERT, 데이터 없음)
2. [ ] `build_safety_model_dl.py` — TabNet 안전 예측 (RF 앙상블)
3. [ ] AI Hub 자전거도로 데이터셋 신청 → `build_road_image_model.py` (EfficientNet-B0)
4. [ ] 딥러닝 모델 FastAPI 엔드포인트 추가

---

---

## Phase 3-11: 사용자 프로파일 기반 개인화 경로 추천 ← 이번주 (2026-04-09~)

> **목표**: 키·몸무게·나이·성별·자전거 종류·사는곳 입력 → 수도권 경로에서 맞춤 가중치 적용 → 개인화된 A→B·순환 코스 추천

### 기술 판단

| 입력 항목 | 활용 방식 | 구현 위치 |
|----------|----------|----------|
| **나이** (AGE_GRP) | 고령(60+) → safety_weight ↑, 젊은층(20~30대) → tourism_weight ↑ | Streamlit 사이드바 |
| **성별** | 통계 기반 선호도 반영 (참고용) | Streamlit 사이드바 |
| **자전거 종류** | 로드바이크 → 속도/거리 우선, 시티/하이브리드 → 안전 우선 | 경로 가중치 조정 |
| **키·몸무게** (BMI) | 적정 주행 거리 자동 계산 (BMI 기반 체력 지수) | 순환 코스 거리 제안 |
| **사는 곳** | 출발점 자동 설정 (시군구 → 좌표 변환) | 경로 탐색 출발점 |

### 자전거 종류별 가중치 프로파일

```python
BIKE_PROFILES = {
    "로드바이크":     {"safety_w": 0.4, "tourism_w": 0.3, "dist_factor": 1.5},
    "MTB":           {"safety_w": 0.5, "tourism_w": 0.3, "dist_factor": 1.2},
    "시티/하이브리드": {"safety_w": 0.6, "tourism_w": 0.3, "dist_factor": 1.0},
    "전기자전거":     {"safety_w": 0.5, "tourism_w": 0.4, "dist_factor": 1.8},
    "접이식":        {"safety_w": 0.6, "tourism_w": 0.2, "dist_factor": 0.7},
}

AGE_SAFETY_BOOST = {
    "10대":  0.0, "20대": 0.0, "30대": 0.0,
    "40대":  0.05, "50대": 0.1, "60대 이상": 0.2,
}

# BMI → 권장 순환 거리 (km)
def recommend_distance(height_cm, weight_kg):
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi < 18.5:   return 15   # 저체중 → 무리 없이
    elif bmi < 25.0: return 20   # 정상
    elif bmi < 30.0: return 15   # 과체중
    else:            return 10   # 비만 → 단거리
```

### Streamlit UI 변경사항

```text
사이드바에 "내 프로파일" 섹션 추가:
  - 나이대 (select: 10대~60대 이상)
  - 성별 (select: 남/여/미선택)
  - 자전거 종류 (select: 로드/MTB/시티/전기/접이식)
  - 키(cm), 몸무게(kg) (number_input)
  - 사는 곳 (text_input: 시군구명 → 좌표 변환)

경로 탐색 탭:
  - "내 프로파일 반영" 토글 ON → EFFECTIVE_WEIGHTS 자동 재계산
  - 순환 코스 거리 슬라이더 default = recommend_distance(height, weight)
```

### Task

```text
[ ] streamlit_kride.py에 사용자 프로파일 사이드바 섹션 추가
[ ] BIKE_PROFILES / AGE_SAFETY_BOOST 가중치 로직 구현
[ ] BMI 기반 권장 거리 함수 + 순환 코스 자동 반영
[ ] 사는 곳 → 좌표 변환 (주소 검색 또는 행정구역 매핑)
[ ] (장기) Neural CF 개인화 모델 (tn_traveller_master TRAVEL_STYL_1~8 활용)
```

---

## Phase 7: 셀카 → 자전거 패션·용품 추천 ← 다음 Phase (제품 카탈로그 필요)

> **목표**: 전신 셀카 사진 업로드 → 체형 분류 → 자전거 용품 사이즈·스타일 추천

### 기술 판단 (2026-04-09)

| 구성 요소 | 기술 | 가능성 |
|----------|------|--------|
| 셀카 입력 | Streamlit `st.camera_input()` | ✅ 즉시 가능 |
| 체형 추정 | MediaPipe BlazePose (17개 키포인트) | ✅ 가능 (정밀도 제한) |
| 키·몸무게 정밀 추정 | 단일 2D 사진으로는 오차 큼 | ⚠️ 참고용으로만 |
| 체형 분류 | 키/몸무게 직접 입력 → 사이즈 추천 | ✅ 규칙 기반 가능 |
| 패션 스타일 매칭 | CLIP 임베딩 → 제품 이미지 유사도 | ✅ 기술 가능 |
| **제품 카탈로그** | 자전거 의류/용품 DB (없음) | ❌ **핵심 선결 조건** |

### 단계별 구현 계획

```text
Phase 7-1 (단기, 카탈로그 없이 가능):
  - 키·몸무게 직접 입력 → 자전거 헬멧/의류 사이즈 표 추천 (규칙 기반)
  - 신체 사이즈 기준: XS/S/M/L/XL 매핑 테이블

Phase 7-2 (중기, 카탈로그 구축 필요):
  - 쿠팡/무신사 자전거 카테고리 제품 크롤링 → 제품 DB
  - MediaPipe 체형 분류 + CLIP 스타일 매칭 → 상위 5개 추천

Phase 7-3 (장기, 이미지 DL):
  - EfficientNet 기반 체형 분류 (학습 데이터 필요)
  - 개인화: 선호 색상·브랜드 히스토리 반영
```

### 선결 조건

```text
❌ 현재 구현 불가 이유:
  1. 제품 카탈로그(DB) 없음
  2. 체형 → 제품 매핑 규칙 미정의
  3. 학습용 체형-패션 페어 데이터 없음

→ 이번주: 설계 문서화만 완료
→ 다음 Phase 착수 조건: 제품 카탈로그 최소 100개 이상 확보
```

---

## DL 품질 기준 (2026-04-09 신규 적용)

> **피드백 반영**: 앞으로 모든 딥러닝 모델은 Train/Validation/Test 3분할을 명시적으로 구현해야 한다.

### 필수 기준

```python
# 모든 DL 모델 공통 분할 비율
from sklearn.model_selection import train_test_split

# 1단계: train+val vs test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
# 2단계: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.18, random_state=42  # 전체의 15%
)
# 결과: train 70% / val 15% / test 15%

# 최종 성능 리포트: test set 기준으로만 보고
# val set: 학습 중 early stopping 용도
# test set: 최종 평가 (학습 중 절대 사용 금지)
```

### 이미지 DL 품질 기준

```text
이미지 모델 (EfficientNet, ResNet 등):
  - 최소 학습 이미지: 클래스당 500장 이상
  - 데이터 증강 필수: RandomFlip, RandomRotation, ColorJitter
  - 평가 지표: Accuracy + Confusion Matrix + Per-class F1
  - 전이학습 활용: ImageNet → Fine-tuning (cold start 방지)

현재 상태:
  - road image CNN: AI Hub 자전거도로 데이터셋 신청 전 → 구현 불가
  - tourism image: tn_tour_photo 실제 이미지 미확보 → 구현 불가
  → 이미지 DL은 데이터 확보 후 진행 (무의미한 더미 구현 금지)
```

---

## 모델 개선 백로그 (시간 여유 시)

- TAAS 자전거 사고 다발지 데이터 → safety_index 타겟 품질 향상
- 날씨 API 연동 → 실시간 가중치 조정
- 사용자 피드백 데이터 수집 → **Neural Collaborative Filtering** 개인화 추천
- 경기도 레저스포츠 POI 재수집 (타임아웃 건)
- KLUE-BERT 미세조정 (자전거·관광 도메인 특화 데이터 확보 시)
- GNN(Graph Neural Network) — 도로 연결 구조를 그래프로 모델링 (데이터 충분 시)

## 딥러닝 라이브러리 목록 (requirements 추가 예정)

```text
torch>=2.0.0
transformers>=4.38.0          # KLUE-BERT / KR-FinBert
pytorch-tabnet>=4.1.0          # TabNet
torchvision>=0.15.0            # EfficientNet CNN
sentencepiece                  # BERT 토크나이저
selenium                       # 리뷰 크롤링
webdriver-manager              # Selenium 드라이버 관리
```

---

## Streamlit UI 수정 계획 (2026-04-10 캡처 기반)

> 상세 원인 분석은 research.md 섹션 34 참조.

### 수정 항목 목록

| # | 탭 | 이슈 | 상태 |
|---|-----|------|------|
| A | Tab1 | `district_danger` 슬라이더 → 시군구 selectbox로 교체 | [ ] 미완 |
| B | Tab2 | 순위 기준 caption 추가 + 도로명 컬럼 표시 | [ ] 미완 |
| C | Tab3 | 관광 점수 낮은 이유 설명 expander 추가 | [ ] 미완 |
| D | Tab5 | 로딩 속도 개선 (sparse 행렬 + spinner) | [ ] 미완 |
| E | 사이드바 | KMA_API_KEY 없을 때 경고 숨김 처리 | [ ] 미완 |

### A. Tab1 — 시군구 selectbox

```text
파일: streamlit_kride.py
변경:
  1. load_district() 캐시 함수 추가 (district_danger.csv 로드)
  2. Tab1 col2에서 district_danger 슬라이더를 selectbox로 교체
     - "직접 입력" 옵션도 유지 (서울/경기 외 지역 대응)
  3. 선택 시 해당 지역 danger_score 자동 반영
```

### B. Tab2 — 순위 기준 명시 + 도로명

```text
파일: streamlit_kride.py
변경:
  1. 테이블 위에 st.info() 추가:
     "1위 = route_score 최고 경로 | route_score = 안전점수×{w} + 관광점수×{1-w}"
  2. road_scored.csv에 "노선명" 컬럼 있으면 display_cols에 추가
  3. st.dataframe의 column_config로 컬럼 설명 추가
```

### C. Tab3 — 관광 점수 설명

```text
파일: streamlit_kride.py
변경:
  1. st.expander("📌 관광 점수가 낮은 이유") 추가
  2. 내용: POI 희소성, 레저스포츠 83개 한계, MinMaxScaler 동작 방식 설명
```

### D. Tab5 — 로딩 속도

```text
파일: streamlit_kride.py, build_poi_recommender.py
변경 1 (빠른 수정):
  - load_poi_rec() 내에서 _sorted_places 미리 계산 후 반환
  - Tab5 진입 시 st.spinner("불러오는 중...") 추가
변경 2 (근본 해결, 별도 실행 필요):
  - build_poi_recommender.py에서 jaccard 행렬을 scipy sparse로 저장
  - poi_cooccurrence.pkl 재생성 필요
```

### E. 사이드바 — KMA 경고 제거

```text
파일: streamlit_kride.py
변경: KMA_KEY가 없으면 st.warning() 대신 조용히 날씨 섹션 숨김
```

---

## Phase 8: 지능형 AI 자전거 여행 비서 (LLM + LangChain + RAG + Gradio) ← 신규 (2026-04-15)

> **목표**: 기존 ML/DL 파이프라인 위에 LLM 레이어를 얹어 "대화형 자전거 여행 비서"로 진화.
> 사용자가 "BTS 탐방하며 내일 전기자전거로 서울 돌고 싶어"라고 말하면,
> 에이전트가 목적을 분류하고 → 날씨/경로/안전/SNS 크롤링 도구를 순서대로 호출하고 → 자연어 리포트 + 지도로 응답한다.

### 설계 결정 요약 (2026-04-15 사용자 확정)

| 항목 | 결정 |
|------|------|
| 목적 분류 방식 | UI 버튼 우선 + LLM 자동 추론 + 불확실 시 되묻기 + 첫 턴 에이전트 질문 (B+C+D) |
| 목적 복수 선택 | 메인 1개 + 서브 최대 2개 (C) |
| 데이터 소스 | 목적별 전용 소스 매핑 — 호캉스=야놀자, K문화=위버스/팬카페, 인생샷=인스타 해시태그 등 (A) |
| 경로 파라미터 세팅 | LLM이 목적+사용자 조건 보고 동적 생성 (B) |
| 디버그 표시 | 개발 중에만 [K문화][커플여행] 태그 표시, 이후 숨김 (D) |
| 아키텍처 모델 | Option C — 완전 하이브리드 (LLM이 오케스트레이터, 알고리즘이 실행자) |
| LangChain 방식 | Text-to-API + ReAct Agent (A+B) |
| Gradio 포지션 | 개발자 데모 샌드박스 + 멀티모달(음성) 프론트엔드 (C+D) |
| 핵심 우려사항 | 할루시네이션 + 레이턴시 |

---

### 전체 아키텍처 (Option C 하이브리드)

```
[사용자 입력 텍스트 or 음성(Whisper)]
        |
        v
[Phase 8-0: 목적 분류 모듈 IntentClassifier]
  UI버튼 -> LLM추론 -> 재확인 -> 파라미터 생성
        |
        v (목적 + 파라미터)
[Phase 8-1: LangChain ReAct Agent 오케스트레이터]
  목적에 따라 도구 선택/호출/합성
        |
   +----+----+----+----+
   |    |    |    |    |
[날씨][경로][안전][SNS][POI RAG]
Tool  Tool  Tool  크롤링 ChromaDB
   |    |    |    |
[KMA][Dijkstra][RF/TabNet][Google Search]
 API  [Graph]            [각 목적별 소스]

[기존 K-Ride 모델들 재사용 — LLM은 설명만 담당]
        |
        v
[Phase 8-4: Gradio UI]
  채팅 + Folium 지도 + 디버그 태그 + 음성 입력
```

---

### Phase 8-0: 목적 분류 모듈 (IntentClassifier)

파일: `kride_ai/intent/classifier.py`, `kride_ai/intent/category_config.py`

#### 목적 카테고리 정의 (초안 — 추후 확장 가능)

| 카테고리 | 키워드 | 전용 크롤링 소스 | 필수 결과 항목 |
|---------|--------|----------------|--------------|
| 관광 | 명소, 랜드마크, 역사, 박물관 | 네이버 블로그, 구글 Places | 명소명, 운영시간, 입장료 |
| 호캉스 | 호텔, 풀빌라, 스파, 루프탑 | 야놀자, 여기어때 리뷰 | 숙소명, 등급, 자전거보관 여부 |
| K문화 | 아이돌, 드라마 촬영지, K-pop | 위버스, 팬카페, 나무위키 | 멤버명/콘텐츠명, 방문 근거 출처 |
| 인생샷 | 포토스팟, 인스타, 감성카페, 뷰맛집 | 인스타 해시태그, 핀터레스트 | GPS좌표, 골든타임, 주차 여부 |
| 액티비티 | 서핑, 클라이밍, 카약, 번지점프 | 클룩, 에어비앤비 체험 | 예약 필요 여부, 연령/체력 제한, 비용 |
| 가족여행 | 키즈카페, 동물원, 안전, 쉬운 코스 | 네이버 지도 어린이 친화 | 경사도, 화장실 위치, 응급실 거리 |
| 커플여행 | 데이트코스, 야경, 분위기, 로맨틱 | 네이버 블로그 데이트코스 | 야경 명소, 분위기 카페 |
| 혼자여행 | 힐링, 조용한, 사색, 나만의 시간 | 네이버 블로그 혼행 | 한적한 구간, 혼밥 가능 식당 |
| 당일치기 | 2-4시간, 짧은, 가까운 | 공통 (거리 필터링) | 총 소요 시간, 대중교통 연계 |
| 자연즐기기 | 산, 강, 공원, 숲길, 한적한 | 한국관광공사 API, 국립공원 | 자연경관 등급, 생태 정보 |

#### 3중 분류 흐름

```
[입력]
  1. UI 버튼 클릭 -> 즉시 카테고리 확정 (가장 빠름)
  2. 텍스트 입력 -> LLM 자동 추론
       confidence >= 0.8 -> 확정
       confidence < 0.8  -> 되묻기
  3. 첫 턴 빈 입력 -> 에이전트가 "어떤 여행을 원하시나요?" 질문

[목적 결정 구조]
  메인 1개 (필수) + 서브 최대 2개 (선택)
  예: 메인=K문화, 서브=[커플여행, 인생샷]

[파라미터 동적 생성 — LLM 담당]
  {
    "safety_weight": 0.7,
    "tourism_weight": 0.3,
    "max_distance_km": 20,
    "prefer_flat": true,
    "bike_type": "electric",
    "crawl_sources": ["weverse", "fancafe"],
    "required_fields": ["멤버명", "방문 근거 출처"]
  }
```

#### 파일 구조

```
kride-project/kride_ai/
├── __init__.py
├── intent/
│   ├── __init__.py
│   ├── classifier.py        <- IntentClassifier 클래스
│   └── category_config.py   <- CATEGORIES dict (소스/키워드/필수항목)
├── agent/
│   ├── __init__.py
│   ├── kride_agent.py       <- ReAct Agent (오케스트레이터)
│   ├── system_prompt.py     <- 할루시네이션 방지 지시 포함
│   └── tools/
│       ├── __init__.py
│       ├── weather_tool.py  <- weather_kma.py 래핑
│       ├── route_tool.py    <- route_graph.pkl + Dijkstra 래핑
│       ├── safety_tool.py   <- safety_regressor.pkl 래핑
│       ├── crawl_tool.py    <- Google Search API + 소스별 크롤러
│       └── poi_rag_tool.py  <- ChromaDB 검색
├── rag/
│   ├── __init__.py
│   ├── vectorstore.py       <- ChromaDB 관리 (컬렉션별 목적 분리)
│   └── ingestion.py         <- 문서 임베딩 파이프라인
└── gradio_app.py            <- Gradio UI
```

#### Task (Phase 8-0)

```
[ ] kride_ai/ 디렉토리 및 __init__.py 생성
[ ] category_config.py 작성 — CATEGORIES dict (10개 카테고리 x 소스/키워드/필수항목)
[ ] classifier.py 작성 — IntentClassifier 클래스
    - classify_from_text(text) -> LLM 추론 (confidence 포함)
    - classify_from_button(category_name) -> 즉시 확정
    - ask_clarification() -> 재확인 질문 생성
    - build_parameters(intent, user_profile) -> LLM 동적 파라미터 생성
```

---

### Phase 8-1: LangChain ReAct Agent (오케스트레이터)

파일: `kride_ai/agent/kride_agent.py`, `kride_ai/agent/system_prompt.py`

#### 핵심 설계 원칙

```
1. LLM은 "무엇을 할지" 결정만 함 — 숫자 생성 금지
2. 경로 계산 -> 반드시 route_tool (Dijkstra) 호출 결과만 사용
3. 안전 점수 -> 반드시 safety_tool (RF/TabNet 모델) 결과만 사용
4. 날씨     -> 반드시 weather_tool (KMA API) 결과만 사용
5. SNS 장소 -> "~로 알려진 곳" 단서 + 출처 링크 필수
```

#### 할루시네이션 방어 시스템 프롬프트 핵심 규칙

```
[절대 규칙]
- "안전합니다"는 safety_tool 데이터 없이 절대 말하지 마세요.
- 경로 좌표는 route_tool 반환값만 사용하세요. 임의로 만들지 마세요.
- 날씨 정보는 weather_tool 호출 결과만 인용하세요.
- 아이돌/연예인 방문지는 반드시 출처 URL을 함께 제시하세요.
- 데이터가 없으면 "확인된 정보가 없습니다"라고 답하세요.
```

#### Task (Phase 8-1)

```
[ ] kride_agent.py 작성
    - create_kride_agent() 함수
    - gpt-4o-mini 사용 (비용 절감 + 빠른 응답)
    - streaming=True (레이턴시 체감 개선)
    - max_iterations=8 (무한루프 방지)
    - return_intermediate_steps=True (디버그용)
[ ] system_prompt.py 작성 — 할루시네이션 방지 절대 규칙 포함
[ ] .env에 OPENAI_API_KEY 추가
```

---

### Phase 8-2: Tool 구현 (기존 K-Ride 모델 래핑)

파일: `kride_ai/agent/tools/` 각 파일

#### 기존 K-Ride 코드 -> Tool 매핑

| Tool 파일 | 래핑 대상 기존 코드 | 담당 역할 |
|-----------|------------------|---------|
| weather_tool.py | weather_kma.py (완료) | 날씨 조회 + safety_score 보정값 |
| route_tool.py | models/route_graph.pkl + Dijkstra | 최적 경로 계산 (경로는 LLM이 만들지 않음) |
| safety_tool.py | models/safety_regressor.pkl (완료) | 도로 구간 안전점수 조회 |
| crawl_tool.py | Google Search API + BeautifulSoup | SNS/웹 크롤링 (목적별 소스 분기) |
| poi_rag_tool.py | ChromaDB + 기존 tour_poi.csv | POI 상세 정보 검색 |

#### crawl_tool 목적별 소스 분기 로직

```python
SOURCE_MAP = {
    "K문화":    ["site:weverse.io", "site:namu.wiki", "팬카페"],
    "인생샷":   ["인스타그램 포토스팟", "site:pinterest.com"],
    "호캉스":   ["site:yanolja.com", "site:goodchoice.kr"],
    "액티비티": ["site:klook.com/ko", "에어비앤비 체험"],
    "관광":     ["네이버 블로그", "site:visitkorea.or.kr"],
    "자연":     ["site:knps.or.kr", "한국관광공사"],
    "가족여행": ["어린이 자전거길", "키즈 친화"],
    "커플여행": ["데이트코스", "야경 명소"],
    "혼자여행": ["혼행", "혼밥", "조용한 자전거길"],
    "당일치기": ["당일치기", "서울 근교"],
}
```

#### Task (Phase 8-2)

```
[ ] weather_tool.py — weather_kma.py 래핑 (LangChain @tool 데코레이터)
[ ] route_tool.py   — route_graph.pkl 로드 + Dijkstra Tool 래핑
[ ] safety_tool.py  — safety_regressor.pkl 추론 Tool 래핑
[ ] crawl_tool.py   — Google Search API + SOURCE_MAP 분기
[ ] poi_rag_tool.py — ChromaDB 검색 Tool
```

---

### Phase 8-3: RAG 지식베이스 구축 (ChromaDB)

파일: `kride_ai/rag/vectorstore.py`, `kride_ai/rag/ingestion.py`

#### 지식베이스 구성 (목적별 컬렉션 분리)

```
ChromaDB 컬렉션 구조:
  kride_poi_general   <- 기존 tour_poi.csv + 관광지 설명
  kride_road_info     <- road_scored.csv 구간 속성 텍스트화
  kride_kculture      <- K문화 크롤링 데이터 (위버스, 나무위키 등)
  kride_lifestyle     <- 호캉스/인생샷/액티비티 크롤링 데이터
  kride_safety_rules  <- 지자체 자전거도로 가이드라인, 안전수칙

임베딩 모델 선택지:
  A. text-embedding-3-small (OpenAI, 유료, 고성능)
  B. jhgan/ko-sroberta-multitask (로컬 무료, 한국어 특화)
  -> 우선 B로 구축, 품질 부족 시 A로 전환
```

#### road_scored.csv 텍스트화 (구간 설명 생성)

```python
def road_to_text(row):
    return (
        f"{row.sigungu}의 자전거도로 구간 ({row.length_km:.1f}km). "
        f"안전점수 {row.safety_score:.2f}, 관광점수 {row.tourism_score:.2f}. "
        f"도로 너비 {row.width_m:.1f}m, 유형: {row.road_type}. "
        f"주변 관광지 {row.tourist_count}개, 편의시설 {row.facility_count}개."
    )
```

#### Task (Phase 8-3)

```
[ ] vectorstore.py 작성 — ChromaDB 초기화/컬렉션 관리/검색 함수
[ ] ingestion.py 작성
    - tour_poi.csv -> kride_poi_general 임베딩
    - road_scored.csv -> kride_road_info 텍스트화 후 임베딩
    - K문화 크롤링 결과 -> kride_kculture 임베딩
[ ] requirements에 chromadb, langchain-chroma 추가
[ ] 임베딩 모델 확정: jhgan/ko-sroberta 우선, 추후 OpenAI 전환 검토
```

---

### Phase 8-4: Gradio UI (데모 + 멀티모달)

파일: `kride_ai/gradio_app.py`
포지션: 개발자/기획자 내부 데모 샌드박스 + 음성 입력 시연

#### UI 구성

```
+----------------------------------------------------------+
|  K-Ride AI 비서 (Gradio 데모)                              |
+----------------------------------------------------------+
|  [목적 버튼] 관광 | 호캉스 | K문화 | 인생샷 | 액티비티      |
|             가족여행 | 커플여행 | 혼자여행 | 당일치기 | 자연  |
|  [서브 목적 체크박스 — 최대 2개]                             |
+---------------------+------------------------------------+
|  채팅 영역           |  Folium 지도 영역                   |
|  스트리밍 응답        |  경로 + POI 마커 표시               |
|                     |                                    |
|  [디버그 패널]       |  [구간별 안전점수 JSON]              |
|  [K문화][커플여행]   |  (Accordion — 개발 중에만 열림)     |
+---------------------+------------------------------------+
|  텍스트 입력 or [마이크 버튼 — Whisper 음성 입력]             |
+----------------------------------------------------------+
```

#### 스트리밍 + 레이턴시 개선 전략

```python
for chunk in agent_executor.stream({"input": message}):
    if "output" in chunk:
        yield chunk["output"]  # 첫 토큰부터 즉시 표시
    if "intermediate_steps" in chunk:
        for action, obs in chunk["intermediate_steps"]:
            if action.tool == "route_tool":
                map_html = render_folium_map(json.loads(obs))
                yield "", map_html  # 경로 나오면 지도 먼저 표시
```

#### Task (Phase 8-4)

```
[ ] gradio_app.py 작성
    - 목적 버튼 그리드 (10개 카테고리, gr.Button)
    - 서브 목적 체크박스 (최대 2개 선택 제한 로직 포함)
    - 스트리밍 채팅 + Folium 지도 동시 렌더링
    - Whisper 음성 입력 (gr.Audio + openai-whisper)
    - 디버그 패널 (gr.Accordion, 개발 중 기본 열림 / 배포 시 닫힘)
[ ] requirements에 gradio>=4.0, openai-whisper 추가
```

---

### Phase 8-5: 할루시네이션 & 레이턴시 방어 전략

#### 할루시네이션 방어 (4중 레이어)

```
방어층 1 (시스템 프롬프트): "데이터 없으면 모른다고 답하라" 절대 규칙
방어층 2 (Tool 설계): 안전/경로 정보는 Tool 반환값만 LLM에 전달
방어층 3 (출처 태깅): SNS 정보에 출처 URL 태그 자동 삽입
방어층 4 (테스트):
  - "존재하지 않는 도로" -> "데이터 없음" 응답 확인
  - "이 도로 완전 안전해?" -> safety_tool 데이터 인용 응답 확인
  - "내일 비 안 와?" -> weather_tool 호출 결과 인용 확인
```

#### 레이턴시 방어 (5중 레이어)

```
방어층 1 (스트리밍): 첫 토큰 즉시 표시 (streaming=True)
방어층 2 (병렬 호출): 날씨 + 경로 + SNS 크롤링 async 병렬 실행
방어층 3 (모델 선택): gpt-4o-mini (gpt-4o 대비 5~10배 빠름)
방어층 4 (캐싱): 동일 경로 요청 LRU 캐시 (TTL=30분)
방어층 5 (선 렌더링): route_tool 결과 나오는 즉시 지도 표시
```

#### Task (Phase 8-5)

```
[ ] test_hallucination.py 작성 (할루시네이션 테스트 케이스)
[ ] 캐싱 레이어 구현 (functools.lru_cache 또는 Redis)
[ ] 응답 시간 측정 로깅 추가
```

---

### Phase 8 필요 패키지 (requirements.txt 추가)

```
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
openai>=1.30.0
chromadb>=0.5.0
gradio>=4.36.0
openai-whisper>=20231117
google-search-results>=2.4.2
beautifulsoup4>=4.12.0
httpx>=0.27.0
sentence-transformers>=3.0.0
```

### Phase 8 전체 실행 순서 (우선순위 순)

```
Priority 1 — 환경 구성
  [ ] kride_ai/ 디렉토리 구조 생성 + __init__.py
  [ ] requirements.txt에 Phase 8 패키지 추가
  [ ] .env에 OPENAI_API_KEY 추가

Priority 2 — 목적 분류 모듈 (Phase 8-0)
  [ ] category_config.py
  [ ] classifier.py

Priority 3 — Tool 구현 (Phase 8-2)
  [ ] weather_tool.py / route_tool.py / safety_tool.py
  [ ] crawl_tool.py / poi_rag_tool.py

Priority 4 — Agent + RAG (Phase 8-1, 8-3)
  [ ] system_prompt.py + kride_agent.py
  [ ] vectorstore.py + ingestion.py

Priority 5 — Gradio UI (Phase 8-4)
  [ ] gradio_app.py

Priority 6 — 방어 및 최적화 (Phase 8-5)
  [ ] test_hallucination.py + 캐싱 + 로깅
```
