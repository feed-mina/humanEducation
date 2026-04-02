# K-Ride 개발 계획

> 최종 업데이트: 2026-04-02
> 목표: 안전점수 모델 + 관광 모델 → 통합 서비스 (Streamlit → Vercel React 연동)

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

2. **사이드바 컨트롤**
   - 안전 / 관광 가중치 슬라이더 (w_safety, w_tourism)
   - 지역 필터: 서울 / 경기 / 전체, **구(區) 단위로 세분화**
     - 위경도 → 구 이름 역지오코딩 또는 행정동 경계 GeoJSON 활용
     - 서울 25개 구, 경기 주요 시·구 목록 드롭다운

3. **추천 결과 테이블**
   - 상위 10개 세그먼트 표시 항목: 구 이름, 도로 유형(road_type), 길이(km), safety_score, tourism_score, final_score
   - 세그먼트가 "어느 구의 어떤 성격 도로인지" 한눈에 파악 가능하도록 구성

4. **모델 정보 패널**
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
[road_features.csv]
    │
    ├─ Safety Model (RF .pkl) ──┐
    └─ Tourism Scaler (.pkl) ───┤
                                ↓
                        Composite Score
                                │
          ┌─────────────────────┼───────────────────┐
          ↓                     ↓                   ↓
   Streamlit App          FastAPI Server       (추후) DB 연동
   (Streamlit Cloud)      (Railway/Render)     (PostgreSQL/PostGIS)
                                │
                    네이버지도 or Leaflet
                      React Frontend
                         (Vercel)
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
7. [ ] `streamlit_app/app.py` 작성 (구 단위 필터 포함)
8. [ ] `streamlit_app/requirements.txt` 작성
9. [ ] Streamlit Cloud 배포 확인
10. [ ] FastAPI `ml-server/main.py` 작성
11. [ ] React 연동 — 네이버 지도 우선 시도

---

## 모델 개선 백로그 (시간 여유 시)

- TAAS 자전거 사고 다발지 데이터 → safety_index 타겟 품질 향상
- 날씨 API 연동 → 실시간 가중치 조정
- 사용자 피드백 데이터 수집 → 개인화 추천 (협업 필터링)
- 경기도 레저스포츠 POI 재수집 (타임아웃 건)

## 다음 주 백로그

- 네이버 지도 / 카카오맵 주요 명소 리뷰 크롤링
  - 자전거 도로 관련 사용자 리뷰 수집 → 도로 품질 피처로 활용
  - 관광지 리뷰 감성 분석 → tourism_score 정교화
