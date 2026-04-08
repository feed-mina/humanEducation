# K-Ride 개발 계획

> 업데이트: 2026-04-08
>
> 목표: 안전점수 모델 + 관광 모델 + **경로탐지** + **편의시설 표시** + **여행 코스 추천** + **딥러닝 보강** → 통합 서비스 (Streamlit → Vercel React 연동)
>
> 딥러닝 추가 목표: KLUE-BERT 감성분석·TabNet 안전예측·CNN 도로이미지 분류를 단계적으로 도입해 모델 품질을 향상시킨다.
>
> **서비스 목표 (전체)**: 자전거 여행 + 안전성 + 경로탐지 + 관광지/여행 추천 + 자전거 이용자 편의 제공

---

## 현재까지 완료된 작업

- [X] 자전거도로 데이터 전처리 (`road_clean.csv`)
- [X] 편의시설 데이터 전처리 (`facility_clean.csv`)
- [X] 관광 POI 수집 및 정제 (`tour_poi.csv`)
- [X] Spatial Join → 통합 학습 데이터 생성 (`road_features.csv`)
- [X] 안전 모델 탐색 (LinearRegression, RandomForest)
- [X] 데이터 모수 문제 분석 (노선명 PK 불가 / 사고 좌표 없음 / 경기도 xls = 명세서)
- [X] `preprocess_road.py` 코드 작성 (시군구명 기반 재전처리 + 모델 3개 비교)
- [X] `build_safety_model.py` 코드 작성 (사고데이터 district_danger 반영, pkl 4개 출력)
- [X] safety_index_v2 설계 완료 (`(1-district_danger)×0.6 + road_attr_score×0.4`)

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
4. [ ] FastAPI `/api/route` 엔드포인트 작성
5. [ ] FastAPI `/api/course` 엔드포인트 작성
6. [ ] FastAPI `/api/facilities` 엔드포인트 작성
7. [ ] Streamlit 앱에 경로 입력 UI + folium 폴리라인 오버레이 추가
8. [ ] Streamlit 앱에 편의시설 레이어(토글) 추가

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
  [ ] KMA Open API 키 발급 (공공데이터포털)
  [ ] fetch_kma_pop() 함수 작성
  [ ] FastAPI /api/recommend, /api/route에 날씨 보정 파라미터 추가
  [ ] Streamlit 앱 사이드바에 현재 날씨 상태 표시 (맑음/흐림/비)
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
  [ ] 이벤트 추출 데이터 (AI Hub) 다운 → NER 학습 데이터 변환
  [ ] KLUE-BERT NER 파인튜닝 스크립트 작성 (build_event_ner.py)
  [ ] (빠른 대안) zero-shot 이벤트 분류 함수 작성
  [ ] Geocoding 파이프라인 구성 (geopy)
  [ ] FastAPI /api/events 엔드포인트 작성
  [ ] /api/route 응답에 events_on_route 필드 추가
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
  [X] build_weather_lstm.py 실행 → models/dl/weather_lstm.pt 생성 (2026-04-08, val_acc=79.43%)
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
  [ ] 여행로그 TL_csv.zip (2.89MB) 다운 → TN_ACTIVITY_CONSUME_HIS 파싱
  [ ] 소비 예측 TabNet 학습 스크립트 작성 (build_consume_model.py)
  [ ] 모델 저장: models/consume_regressor.zip
  [ ] FastAPI /api/route 응답에 estimated_cost 필드 추가
  [ ] Streamlit 경로 결과 카드에 예상 지출 표시
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

> **목표**: 네이버 로드뷰 이미지를 CNN으로 분류 → 도로 포장 상태·너비·장애물 여부 피처화

### 6-1. 데이터 수집

```text
수집 방법: 네이버 지도 스트리트뷰(로드뷰) API
  - road_features.csv의 (start_lat, start_lon) → 파노라마 이미지 요청
  - 1,647개 좌표 × 1장 = 약 1,647장 (MVP 기준)

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
리뷰 텍스트        도로 피처 CSV            네이버 로드뷰
    │                   │                       │
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

## Task 순서

1. [X] `preprocess_road.py` 코드 작성
2. [X] `build_safety_model.py` 코드 작성 (safety_index_v2 + district_danger 반영)
3. [X] `build_tourism_model.py` 코드 작성 (tourism_score + final_score + road_scored.csv)
4. [X] `preprocess_road.py` 실행 → `road_clean_v2.csv` 생성 확인
   - LinearRegression R²=1.0 (leakage — safety_index가 width/length 선형조합이므로 당연)
   - RandomForest R²=0.859 (피처중요도: width_m 0.46 > start_lat 0.34 > start_lon 0.18 > length_km 0.02)
5. [X] `build_safety_model.py` 실행 → pkl 4개 + district_danger.csv 확인
   - 회귀 R²=0.9539 / 분류 F1-macro=0.9864
6. [ ] `build_tourism_model.py` 실행 → road_scored.csv 확인
6-1. [ ] `road_scored.csv`에 `end_lat` / `end_lon` 컬럼 포함 여부 확인
6-2. [ ] 없을 경우: `build_tourism_model.py`에서 종점 좌표 조인 추가 후 재실행
7. [ ] `build_route_graph.py` 작성 → `models/route_graph.pkl` 생성
8. [ ] `streamlit_app/app.py` 작성 (구 단위 필터 + 편의시설 레이어 + 경로 오버레이 + 코스 추천 포함)
9. [ ] `streamlit_app/requirements.txt` 작성
10. [ ] Streamlit Cloud 배포 확인
11. [ ] FastAPI `ml-server/main.py` 작성 (`/api/recommend`, `/api/route`, `/api/course`, `/api/facilities`, `/api/pois`)
12. [ ] KMA Open API 키 발급 → 날씨 연동 함수 작성
13. [ ] React 연동 — 네이버 지도 우선 시도

### 딥러닝 Task (Phase 4~6)

1. [ ] 리뷰 데이터 수집 스크립트 작성 (`step5_review_collect.py`)
2. [ ] `build_sentiment_model.py` 작성 — KLUE-BERT 감성 분석 파이프라인
3. [ ] `review_sentiment.csv` → road_scored와 Spatial Join → `road_scored_v2.csv`
4. [ ] `build_safety_model_dl.py` 작성 — TabNet 회귀/분류
5. [ ] RF + TabNet 앙상블 비교 실험 (R², F1 기록)
6. [ ] 네이버 로드뷰 이미지 수집 스크립트 작성
7. [ ] 이미지 라벨링 (수동 또는 GPT-4V 자동 라벨링)
8. [ ] `build_road_image_model.py` 작성 — EfficientNet-B0 전이학습
9. [ ] 딥러닝 모델 FastAPI 엔드포인트 추가 (`/api/safety_dl`, `/api/sentiment`)

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
