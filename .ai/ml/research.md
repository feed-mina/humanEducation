# K-Ride ML 기술 리서치

## 현재 진행 상황 요약

- TAAS 사고 데이터 탐색 완료 (`0330`, `0331` 노트북)
- 안전지수 산출 공식 초안 완성
- 자전거도로 피처 기반 회귀/분류 모델 실험 중

---

## 1. 데이터 소스

### 1.1 TAAS (교통사고분석시스템)

- **URL:** <https://taas.koroad.or.kr>
- **데이터:** 자전거 관련 교통사고 (위치, 유형, 피해 정도)
- **현황:** `data/raw_accident_data/` 에 저장 완료
- **주요 컬럼:** 사고일시, 위도/경도, 사망자수, 중상자수, 경상자수, 사고유형

### 1.2 공공데이터포털 자전거도로

- **URL:** <https://www.data.go.kr>
- **데이터:** 전국 자전거도로 노선, 너비, 관리기관
- **현황:** `data/raw_ml/` 에 일부 저장 (서울 데이터 중심)
- **주요 컬럼:** 자전거도로너비, 안전지수, 위도/경도, 관리기관명

### 1.3 AI Hub (딥러닝용)

- **URL:** <https://aihub.or.kr>
- **데이터셋 후보:**
  - 차량 주행 영상 (자전거도로 주행 영상)
  - 보행자 인식 데이터
  - 도로 장애물 탐지 데이터
- **라이선스:** 비상업적 연구 목적 사용 가능

---

## 2. 안전지수 산출 방법

### 현재 공식 (노트북 기준)

```python
safety_score = (사망자수 * 50) + (중상자수 * 20) + (경상자수 * 10)
```

점수가 높을수록 위험. 경로별 100m 반경 내 사고 집계 후 정규화 예정.

### 개선 방향

- 시간대 가중치 추가 (야간 사고 = 1.5배)
- 사고 유형 가중치 (자전거-차량 충돌 > 단독 사고)
- 최근 3년 데이터 가중 평균 (최근 연도 가중치 증가)

---

## 3. 모델 선택 근거

### 3.1 안전지수 예측 (회귀)

| 모델 | 장점 | 단점 | 채택 여부 |
| ---- | ---- | ---- | --------- |
| LinearRegression | 해석 용이 | 비선형 관계 포착 불가 | 기준선 |
| RandomForest | 비선형, 이상치 강건 | 학습 느림 | 채택 |
| XGBoost | 높은 성능, 피처 중요도 | 과적합 주의 | 채택 |

**선택:** RandomForest + XGBoost 앙상블, LinearRegression은 기준선(baseline)으로 사용.

### 3.2 위험등급 분류 (분류)

- 타겟: 안전지수 기준 3등급 (안전/보통/위험)
- 클래스 불균형: 대부분 경로가 "안전" → SMOTE 오버샘플링 적용
- 평가: F1-macro (불균형 데이터에 적합)

### 3.3 경로 추천 (코사인 유사도)

```python
from sklearn.metrics.pairwise import cosine_similarity

# 경로 피처 벡터: [안전지수, 도로폭, 사고건수, 경사도, 관광지접근성]
similarity_matrix = cosine_similarity(route_features)
# 사용자 선호도 벡터와 유사도 계산 → Top-5 추천
```

---

## 4. 공간 데이터 처리 (Spatial Join)

### 방법: geopandas + shapely

```python
import geopandas as gpd
from shapely.geometry import Point

# 사고 데이터 → GeoDataFrame
accident_gdf = gpd.GeoDataFrame(
    accident_df,
    geometry=gpd.points_from_xy(accident_df.경도, accident_df.위도),
    crs="EPSG:4326"
)

# 자전거도로 버퍼 (100m) → 사고 건수 집계
route_gdf["buffer"] = route_gdf.geometry.buffer(0.001)  # ~100m
joined = gpd.sjoin(accident_gdf, route_gdf, how="left", predicate="within")
route_safety = joined.groupby("route_id")["safety_score"].sum()
```

### PostGIS 쿼리 (서버 측)

```sql
SELECT r.route_id, COUNT(a.accident_id) as accident_count
FROM bicycle_routes r
LEFT JOIN accidents a
  ON ST_DWithin(r.geom::geography, a.geom::geography, 100)
GROUP BY r.route_id;
```

---

## 5. YOLOv8 객체 탐지

### 탐지 대상 클래스

| 클래스 | 설명 | 위험 가중치 |
| ------ | ---- | ----------- |
| pedestrian | 보행자 | 높음 |
| bicycle | 자전거 | 중간 |
| vehicle | 차량 | 높음 |
| obstacle | 장애물 (공사, 라바콘) | 높음 |
| congestion | 혼잡 구간 | 중간 |

### 학습 전략

```text
Base model: YOLOv8n (nano) — CPU 환경 대응
Fine-tuning: AI Hub 자전거 주행 영상 데이터
Epochs: 50 (조기 종료 patience=10)
Image size: 640x640
Batch size: 16
```

### 평가 목표

- mAP@0.5 > 0.60
- 추론 속도: < 100ms / frame (CPU)

---

## 6. 경로 최적화 알고리즘

### 수정된 Dijkstra (다중 요인)

```python
# 엣지 가중치 = 안전점수 * w1 + 거리 * w2 + 혼잡도 * w3
# 사용자 모드별 가중치
weights = {
    "safe":    {"safety": 0.6, "distance": 0.2, "congestion": 0.2},
    "fast":    {"safety": 0.2, "distance": 0.6, "congestion": 0.2},
    "tourist": {"safety": 0.3, "distance": 0.3, "congestion": 0.4},
}
```

---

## 7. 프로파일링 리포트 분석 결과

> 분석 파일: `kride-project/report/`
> 생성일: 2026-03-31

### 7.1 자전거 도로 데이터 (`bicycle_data_profiling_report.html`)

**기본 정보** *(2026-03-31 실제 확인값으로 수정)*

| 항목 | 값 |
| ---- | -- |
| 행 수 | **20,262** (프로파일링 시점 11,771 → 파일 업데이트됨) |
| 열 수 | 23 |
| 결측 셀 | 미재측정 |
| 중복 행 | 0 |

**사용 가능한 핵심 컬럼**

| 컬럼 | 타입 | 결측률 | ML 활용 |
| ---- | ---- | ------ | ------- |
| 기점위도 / 기점경도 | Numeric | 낮음 | Spatial Join 기준점 |
| 종점위도 / 종점경도 | Numeric | 낮음 | Spatial Join 종점 |
| 자전거도로너비(m) | Numeric | 낮음 | 안전 피처 |
| 총길이(km) | Numeric | 낮음 | 경로 거리 피처 |
| 시도명 | Categorical | 낮음 | 지역 그룹핑 |
| 자전거도로종류 | Categorical | 3.2% | 도로 유형 피처 |
| 자전거도로고시유무 | Boolean | 2.4% | 공식 인증 여부 |

**제거 대상 컬럼 (결측률 높음)**

| 컬럼 | 결측률 | 처리 방법 |
| ---- | ------ | --------- |
| 노선번호 | 87.6% | 제거 |
| 기점지번주소 / 종점지번주소 | 85.3% | 제거 |
| 주요경유지 | 87.1% | 제거 |
| 일반도로너비(m) | 95.1% | 제거 |

**시도명 전체 분포** *(2026-03-31 노트북 실측 확인값)*

| 시도 | 행 수 | 비율 | 비고 |
| ---- | ----- | ---- | ---- |
| 경상북도 | 9,066 | 44.7% | 낙동강 자전거길, 농촬 전용도로 구간 많음 |
| 경기도 | 4,939 | 24.4% | 한강임진강 자전거길 |
| 경상남도 | 761 | 3.8% | |
| 울산광역시 | 639 | 3.2% | |
| 대구광역시 | 628 | 3.1% | |
| 강원특별자치도 | 593 | 2.9% | |
| 광주광역시 | 586 | 2.9% | |
| 대전광역시 | 585 | 2.9% | |
| 인천광역시 | 507 | 2.5% | |
| 충청북도 | 410 | 2.0% | |
| **서울특별시** | **380** | **1.9%** | 도시형 차선 위주, 전용도로 공식 등록 적음 |
| 전라남도 | 288 | 1.4% | |
| 충청남도 | 261 | 1.3% | |
| 전북특별자치도 | 238 | 1.2% | |
| 부산광역시 | 220 | 1.1% | |
| 제주특별자치도 | 118 | 0.6% | |
| 전라북도 | 42 | 0.2% | |
| 강원도 | 1 | 0.0% | |
| **합계** | **20,262** | **100%** | |

**⚠️ 경상북도 9,066행이 많은 이유**

이 데이터셋은 주의 구간(세그먼트) 단위로 등록되어 있음. 경상북도는 한국에서 맨 넓은 주(서울의 31배).

- 낙동강 자전거길(수백 km) 등 강 주변 전용도로가 잠가 단위로 잉세히 등록됨
- 서울은 도시 특성상 차선형 자전거도가 대부분 → **연장은 존재하지만 전용도로로 공식 등록이 적음**
- 서울시는 별도 구돉별 GPX/GeoJSON 형태로 자체 관리
- **결론: 서울 380행은 오류가 아닌 정확한 데이터** 표기 모드 차이임

**필터링 전략**

- **선택: 서울특별시 + 경기도 = 5,319행** → K-Ride 수도권 서비스 중심, ML 학습에 충분

**경고 사항**

- `자전거도로종류` 불균형 83.9% — 단일 유형 압도적
- 위경도 컨럼들 간 높은 상관관계 (다중공선성 주의 — Spatial Join 용도이므로 무관)

**ML에서 해야 할 분석 작업**

- [x] 사용 컨럼 7개만 추출하여 정제 CSV 생성
- [x] `시도명` 기준으로 **서울특별시 + 경기도** 필터링 (5,319행) → `isin()` 사용
- [x] `자전거도로너비(m)` + `쳙길이(km)` 로 안전 피처 생성 (`is_wide_road`, `safety_index`)
- [ ] 기점/종점 위경도로 LineString 생성 후 TAAS 데이터와 Spatial Join

> 파일: `step2_road_clean.py` → 산출물: `road_clean.csv` (5,319행, 9컨럼)

---

### 7.2 편의시설 데이터 (`bicycle_data_facility_profiling_report.html`)

**기본 정보**

| 항목 | 값 |
| ---- | -- |
| 행 수 | 3,368 |
| 열 수 | 43 |
| 결측 셀 | 41,084 (28.4%) |
| 중복 행 | 0 |

**사용 가능한 핵심 컬럼**

| 컬럼 | 타입 | 결측률 | ML 활용 |
| ---- | ---- | ------ | ------- |
| x 좌표 / y 좌표 | Numeric | 0% | 편의시설 위치 (Spatial Join) |
| 거리 | Numeric | 0% | 편의시설 거리 피처 |
| is_24h | Boolean | 낮음 | 24시간 운영 여부 |
| has_restricted_hours | Boolean | 낮음 | 운영 제한 여부 |
| 상세정보 값 2 (설치지역) | Text | 31.5% | 시설 위치 보조 정보 |
| 상세정보 값 4 (설치유형) | Categorical | 70.9% | 시설 유형 분류 |

**제거 대상 컬럼 (상수 또는 고결측)**

| 컬럼 | 이유 |
| ---- | ---- |
| 사용유무, 테마타입, 좌표타입, 테마명 | 상수값 (분산 없음) |
| 이미지 URL1/2, 메인 이미지 | 결측 94~100%, ML 비활용 |
| 상세정보 값 6~10 | 결측 99% 이상 |

**경고 사항**

- `is_24h` 불균형 91.3%, `has_restricted_hours` 불균형 98.4%
- `상세정보 이름 7/8/9` 불균형 98~99% → 제거 권장
- `x 좌표`와 `거리` 간 높은 상관관계 (하나만 사용 권장)

**ML에서 해야 할 분석 작업**

- [x] 43개 컬럼에서 유효 컬럼 6개 추출 후 정제 CSV 저장 → `facility_clean.csv` 생성 완료
- [ ] x/y 좌표 → `EPSG:4326` 위경도 확인 (x=126.95, y=37.54 → WGS84로 확인됨, EPSG:5179 변환 불필요)
- [ ] 각 자전거도로 경로 기준 반경 500m 내 편의시설 수 집계 (새 피처)
- [ ] `설치유형` 원핫인코딩 → 경로 추천 피처로 활용

---

### 7.3 분석 파일 목록 및 작업 순서 *(2026-03-31 업데이트)*

| 순서 | 파일 | 작업 내용 | 산출물 | 상태 |
| ---- | ---- | --------- | ------ | ---- |
| 1 | `ml.ipynb` | TAAS 사고 데이터 위경도 정제 | `taas_clean.csv` | ⏳ |
| 2 | `step2_road_clean.py` | 자전거도로 서울+경기 필터 + 피처 생성 | `road_clean.csv` | ✅ **완료** (5,319행, 332KB) |
| 3 | `step1_facility_clean.py` | 편의시설 6개 컬럼 추출 | `facility_clean.csv` | ✅ **완료** (3,368행) |
| 4 | `step3_tour_collect.py` | TourAPI 호출 → 서울+경기 관광지 좌표 수집 | `tour_poi.csv` | ✅ **완료** (3,407건, 453KB) |
| 5 | `step4_spatial_join.py` | 도로 + 관광지/편의시설 Spatial Join | `road_features.csv` | ⏳ **다음** |
| 6 | `ml.ipynb` | 도로 + TAAS Spatial Join (100m 반경) | `route_safety_features.csv` | ⏳ |
| 7 | `ml.ipynb` | 최종 피처 병합 + 모델 학습 | `safety_model.pkl` | ⏳ |

---

### 7.4 TourAPI 관광지 POI 수집 계획 *(2026-03-31 신규 추가)*

**API 정보**

| 항목 | 내용 |
| ---- | ---- |
| 서비스명 | 한국관광공사 국문 관광정보 서비스_GW |
| End Point | `https://apis.data.go.kr/B551011/KorService2` |
| 포맷 | JSON (responseType=json) |
| 인증 | 일반 인증키, 자동승인, 이용제한 없음 |

**수집 전략**

- API: `areaBasedList2` (지역 코드 기반 전체 수집)
- 대상 지역: 서울(`areaCode=1`) + 경기도(`areaCode=31`)
- 콘텐츠 타입 필터 (`contentTypeId`)

  | 코드 | 유형 | 활용 피처 |
  | ---- | ---- | ---- |
  | 12 | 관광지 | `tourist_count` |
  | 14 | 문화시설 | `cultural_count` |
  | 28 | 레저스포츠 | `leisure_count` |

- 저장 컬럼: `title`, `mapx`(x좌표), `mapy`(y좌표), `contentTypeId`, `addr1`
- 산출물: `data/raw_ml/tour_poi.csv`

**Spatial Join 연계**

```python
# 각 자전거도로 경로(LineString) 기준 1km 반경 내 관광지 수 집계
# 스크립트: step4_spatial_join.py
# 산출 컬럼: tourist_count, cultural_count, leisure_count
```

---

## 8. 미결 사항 (Action Items)

- [ ] TAAS 위경도 컬럼명 통일 (일부 파일 컬럼명 불일치 확인 필요)
- [x] 편의시설 x/y 좌표계 확인 → WGS84(EPSG:4326) 확인됨, 변환 불필요
- [x] 자전거도로 필터링 기준 확정 → 서울특별시 + 경기도 (5,319행)
- [x] **road_clean.csv 생성 완료** → 5,319행, 332KB, 9컬럼 (`step2_road_clean.py`)
- [x] **facility_clean.csv 생성 완료** → 3,368행, 6컬럼 (`step1_facility_clean.py`)
- [x] 관광지 POI API 키 발급 완료 → 한국관광공사 국문 관광정보 서비스_GW
  - End Point: `https://apis.data.go.kr/B551011/KorService2`
  - 포맷: JSON+XML, 자동승인, 이용제한 없음
  - 활용 계획: 자전거 경로 1km 반경 내 관광지 수 → `tourist_count` 피처
  - 주요 API: `/areaBasedList2` (지역 기반), `/locationBasedList2` (위치 기반)
- [ ] **`step3_tour_collect.py` 작성** → TourAPI 호출로 서울+경기 관광지 좌표 수집
- [ ] `tour_poi.csv` 생성 후 Spatial Join 코드 작성
- [ ] AI Hub 데이터셋 신청 및 다운로드 (수동 작업)
- [ ] 경사도 데이터 수집 방법 확정 (카카오맵 API vs 공공데이터)
