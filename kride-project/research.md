# K-Ride Research: 데이터 분석 결과

> 최종 업데이트: 2026-04-08 (딥러닝 모델 리서치 추가 / 서비스 목표 갭 분석 추가)

---

## 1. 수집 및 전처리 데이터 현황

| 파일 | 행 수 | 컬럼 수 | 설명 |
|---|---|---|---|
| `road_clean.csv` | 5,319 | 10 | 서울+경기 자전거도로 (원본 20,262행 중 필터) |
| `facility_clean.csv` | 3,368 | 6 | 서울시 자전거 편의시설 |
| `tour_poi.csv` | 2,529 | 9 | 서울+경기 관광지/문화시설/레저 POI |
| `road_features.csv` | 1,647 | 14 | Spatial Join 최종 학습용 데이터 |

### road_features.csv 컬럼 구성
- **위치**: `start_lat`, `start_lon`, `end_lat`, `end_lon`
- **도로 속성**: `length_km`, `width_m`, `road_type`, `is_official`, `is_wide_road`
- **안전 타겟**: `safety_index` (도로너비 × 공식도로여부 기반 복합 지수, 평균 0.131)
- **관광 피처**: `tourist_count`, `cultural_count`, `leisure_count`, `facility_count`

---

## 2. 관광 POI 분포

| 타입 | 건수 | 비고 |
|---|---|---|
| 관광지 (contentTypeId=12) | 1,775 | 서울 581 + 경기 1,194 |
| 문화시설 (contentTypeId=14) | 671 | 서울 366 + 경기 305 |
| 레저스포츠 (contentTypeId=28) | 83 | 경기 타임아웃으로 서울분만 |
| **합계** | **2,529** | |

### Spatial Join 결과 (도로 세그먼트 기준)

| 피처 | 평균 | 0 이상인 세그먼트 비율 |
|---|---|---|
| tourist_count (1km 반경) | 1.63 | 62.6% |
| cultural_count (1km 반경) | 0.74 | 34.2% |
| leisure_count (1km 반경) | 0.10 | 6.0% |
| facility_count (500m 반경) | 1.09 | 11.3% |

---

## 3. 모델 성능 비교

### 안전 모델 (타겟: safety_index)

| 모델 | R² Score | 비고 |
|---|---|---|
| LinearRegression (단순: 도로너비) | 0.0096 | 단일 피처 한계 |
| LinearRegression (다중: 너비+위경도) | 0.0635 | 위치 추가 시 소폭 향상 |
| PolynomialFeatures + 선형회귀 | 0.0132 | 비선형 효과 미미 |
| **RandomForestRegressor** | **0.1890** | 현재 최고 성능 |

### RandomForest 피처 중요도

| 피처 | 중요도 |
|---|---|
| 기점위도 (start_lat) | 0.5028 |
| 기점경도 (start_lon) | 0.3903 |
| 자전거도로너비(m) | 0.1068 |

> **분석**: 위치(위경도)가 safety_index를 가장 잘 설명함 → 지역별 도로 환경 차이가 큼. 도로너비 단독으로는 안전을 충분히 설명하지 못함.

---

## 4. 주요 인사이트

1. **R² 0.19는 낮은 수치** — safety_index가 도로너비·공식여부만으로 계산된 단순 지수라 모델 설명력 한계. 사고 다발지 데이터(TAAS)와 결합하면 타겟 품질 개선 가능.

2. **위치가 핵심 피처** — 서울 강북/강남, 경기 지역 간 도로 환경이 크게 다름. 지역 클러스터링을 피처에 추가하면 성능 향상 기대.

3. **관광 데이터 결합 완료** — `road_features.csv`에서 road 세그먼트 단위로 관광 POI가 조인됨. tourist_count가 0인 세그먼트(37.4%)는 관광 추천에서 제외하거나 낮은 가중치 부여.

4. **facility_count 편차 큼** — 평균 1.09이지만 max=55, std=4.61로 편의시설이 특정 구간에 집중. 정규화 필수.

---

## 5. 데이터 모수(母數) 문제 — 전체 모수가 아닌 샘플임

### 데이터 손실 경로

| 단계 | 행 수 | 비율 | 이유 |
| --- | --- | --- | --- |
| 원본 전국 자전거도로 | 20,262 | 100% | `전국자전거도로표준데이터.csv` |
| 서울+경기 필터링 | 5,319 | 26.3% | `TARGET_SIDO = ["서울특별시", "경기도"]` |
| 위경도 결측 제거 후 Spatial Join | 1,647 | **8.1%** | `start_lat`/`start_lon` 결측 3,624행 (68%) 제거 |

> **결론**: 최종 모델은 전국 데이터의 **8.1%** 로 학습됨. 서울+경기 지역 데이터만 다루는 서비스라면 지역 범위는 타당하나, 서울+경기 내에서도 좌표 결측으로 약 2/3가 탈락한 것은 모델 대표성 문제임.

// [메모] 데이터 전처리를 다시 할 필요가 있습니다. 칼럼에서 노선명, 시군구명 와 기점지번주소를 기준으로 경기도와 서울 지역만 다시 전처리를 하여 모델 3개를 비교하겠습니다. 노선명이 유일값으로 추측되어지는데 만약 유일값이면 이를 pk로 기준삼으면 될것같습니다![1775110874194](image/research/1775110874194.png)
// [메모] 여기에 해당하는 py파일을 만들어주시면 제가 진행하겠습니다
### 전국자전거도로표준데이터.csv 전체 컬럼 목록 (23개)

| 컬럼명 | 설명 | 결측 여부 |
| --- | --- | --- |
| `노선명` | 자전거도로 노선 이름 | 없음 |
| `노선번호` | 노선 번호 | - |
| `시도명` | 시/도 | 없음 |
| `시군구명` | 시/군/구 | 없음 |
| `기점지번주소` | 시작점 지번주소 | **15,997행 결측** |
| `종점지번주소` | 끝점 지번주소 | - |
| `기점도로명주소` | 시작점 도로명주소 | - |
| `종점도로명주소` | 끝점 도로명주소 | - |
| `기점위도` | 시작점 위도 | **8,423행 결측** |
| `기점경도` | 시작점 경도 | **8,423행 결측** |
| `종점위도` | 끝점 위도 | - |
| `종점경도` | 끝점 경도 | - |
| `주요경과지` | 경유 지점 | - |
| `총길이(km)` | 도로 총 길이 | 없음 |
| `일반도로너비(m)` | 일반 구간 너비 | - |
| `자전거전용도로너비(m)` | 자전거전용 구간 너비 | - |
| `자전거전용도로종류` | 전용/우선/겸용 등 | - |
| `자전거전용도로이용가능여부` | 이용 가능 여부 | - |
| `관리기관명` | 담당 기관 | - |
| `관리기관전화번호` | 연락처 | - |
| `데이터기준일자` | 데이터 날짜 | - |
| `제공기관코드` | 기관 코드 | - |
| `제공기관명` | 기관명 | - |

### 노선명 PK 가능 여부 검토 결과

| 항목 | 값 |
| --- | --- |
| 전체 행 수 | 20,262 |
| 노선명 유일값 수 | **17,544** |
| 중복 행 수 | **2,718** |

> **결론: 노선명 단독 PK 불가.** `노선명 + 시군구명` 복합키를 사용해야 함.

### 왜 노선명만으로 PK가 안 되는가?

#### 이유 1 — 같은 이름이 여러 지역에 등록됨 (전국 표준명칭 없음)

자전거도로 명칭은 각 지자체가 자체적으로 부여하기 때문에, 서로 다른 시군구가 동일한 이름을 독립적으로 사용함.

```text
예시:
  노선명: "한강자전거길"
    → 시군구: 영등포구  (서울)
    → 시군구: 고양시    (경기)
    → 시군구: 남양주시  (경기)
  → 세 행이 모두 "한강자전거길"이지만 실제로는 다른 도로 구간

예시 2:
  노선명: "수변공원 자전거길"
    → 여러 시군구에서 동일 명칭 사용 가능
    → 각 지자체가 독립적으로 붙인 이름
```

#### 이유 2 — 하나의 노선이 여러 시군구에 걸쳐 분절 등록됨

긴 자전거도로 노선의 경우, 행정구역 경계를 넘어가면서 시군구별로 별도 행으로 등록됨.

```text
예시:
  "아라자전거길" (인천 → 경기 김포 → 서울 강서구)
    → 행 1: 노선명="아라자전거길", 시군구="김포시",  길이=12.3km
    → 행 2: 노선명="아라자전거길", 시군구="강서구",  길이=5.7km
  → 같은 노선명이지만 시군구 단위로 분할 등록
```

#### 이유 3 — 수치로 확인

```text
전체 20,262행에서 노선명 유일값이 17,544개
→ 2,718개 행에서 노선명이 중복 발생
→ 중복률: 13.4%

단독 PK 조건: 유일값 수 == 전체 행 수 (17,544 ≠ 20,262)
```

**해결책: `노선명 + 시군구명` 복합키**

```python
df_target["road_id"] = df_target["노선명"] + "_" + df_target["시군구명"]
# 예: "한강자전거길_영등포구", "한강자전거길_고양시" → 서로 다른 ID로 구분 가능
```

복합키 이후에도 동일 시군구 내 같은 노선명이 여러 행 존재하는 극단적 케이스가 있을 수 있으나, 서비스 범위(서울+경기)에서는 허용 범위로 판단.

### 전처리 재설계 방향 (메모 반영)

기존 `시도명` 필터 대신 `시군구명` 기준으로 재전처리하면 주소 정보를 함께 활용할 수 있음.

```python
# 기존 (시도명 기준)
TARGET_SIDO = ["서울특별시", "경기도"]
df_road = df[df["시도명"].isin(TARGET_SIDO)]

# 개선 (시군구명 + 기점지번주소 기반, 좌표 보완 병행)
# 시군구명 필터 → 기점지번주소 geocoding → 좌표 복원
```

### 왜 좌표가 이렇게 많이 빠지나?

`전국자전거도로표준데이터.csv`의 `기점위도`/`기점경도` 컬럼은 일부 지자체만 입력해서 전국적으로 결측이 많음 (서울+경기 5,319행 중 3,624행, 68% 결측). `기점지번주소` 컬럼도 15,997행 결측으로 geocoding 대상 자체도 적음.

### 좌표 결측 보완 방향 (우선순위)

1. `기점지번주소` 또는 `기점도로명주소`가 있는 행 → 지오코딩 API 호출
2. `서울시 자전거도로 현황` 별도 데이터 (`자전거도로+현황_서울.csv`) 와 `노선명+시군구명` 매칭 병합
3. 위 두 방법으로도 복원 불가 시 → 해당 행 제외 후 모델 한계 명시

---

## 6. safety_index 계산 방식 변천

### v1 공식 (ml.ipynb 기준, 구버전)

```python
# MinMaxScaler로 너비·길이를 0~1 정규화 후 가중합
safety_index = 너비(width_m) × 0.7 + 길이(length_km) × 0.3
```

- **사고 다발지 데이터를 전혀 사용하지 않음**
- 도로가 넓고 길수록 safety_index가 높아지는 구조 → 실제 위험도와 역방향일 수 있음 (넓은 도로 = 차량 속도 높아 사고 위험 증가 가능)

### v2 공식 (build_safety_model.py 기준, 현행)

**2단계 구조**: 구(區) 수준 위험도 + 도로 속성 가중합

```python
# STEP 1: 사고다발지_서울.xlsx → 구별 위험도 계산
danger_score = (발생건수 × 1.0 + 사망자수 × 5.0 + 중상자수 × 2.0 + 부상자수 × 1.0)
district_danger = MinMaxScaler().fit_transform(구별_danger_score)  # 0=안전, 1=위험

# STEP 2: 도로 속성 정규화 (기존 v1 공식)
road_attr_score = width_m_norm × 0.7 + length_km_norm × 0.3

# STEP 3: 통합 (district_danger가 높을수록 unsafe → 반전)
safety_index_v2 = (1 - district_danger) × 0.6 + road_attr_score × 0.4
```

| 구성 요소 | 가중치 | 데이터 출처 |
| --- | --- | --- |
| `(1 - district_danger)` | 0.6 | 서울 자전거 사고 다발지 xlsx |
| `road_attr_score` | 0.4 | 전국자전거도로표준데이터 (너비·길이) |

> **분류 타겟**: `safety_index_v2` 삼분위 기준 → `danger_level` (0=안전/1=보통/2=위험)  
> **회귀 타겟**: `safety_index_v2` 연속값 (지도 색상 그라디언트용)

### 경기도 미매핑 처리

경기도에는 자전거 사고 다발지 데이터가 없으므로, 구별 위험도 매핑 실패 행에는 `district_danger` 중앙값을 대입.

```python
median_danger = df_danger["district_danger"].median()
df_road["district_danger"] = df_road["district_danger"].fillna(median_danger)
```

### ml.ipynb에서 사고 데이터 로드 여부

`ml.ipynb` 내에서 아래 파일들을 **로드한 코드가 없음** (grep 확인):

- `다발지분석-24년 자전거 교통사고 다발지역_서울.xlsx`
- `자전거+교통사고_서울.csv`

0331_taas_data.ipynb에 "사고 좌표와 도로 좌표를 매칭하는 작업을 시작해야 합니다"라는 메모만 있고, 실제 매칭 로직은 **미구현 상태**.

---

## 7. 경기도 사고 다발지 — 자전거 데이터 없음

### raw_ml 폴더의 사고 관련 파일 현황

| 파일명 | 지역 | 사고 유형 | ml.ipynb 반영 여부 |
| --- | --- | --- | --- |
| `다발지분석-24년 자전거 교통사고 다발지역_서울.xlsx` | 서울 | **자전거** 사고 | 미반영 |
| `다발지분석-24년 보행자 교통사고 다발지역_경기도.xlsx` | 경기도 | **보행자** 사고 | 미반영 |
| `자전거+교통사고_서울.csv` | 서울 | 자전거 사고 | 미반영 |
| `한국도로교통공단_전국.csv` | 전국 | (확인 필요) | 미반영 |

> **결론**: 경기도에는 **자전거** 교통사고 다발지 데이터가 없고, 보행자 사고 데이터만 있음. 서울도 자전거 사고 데이터가 있으나 현재 safety_index에 반영되지 않은 상태.

### 사고 데이터 컬럼 분석 (실측)

**다발지분석-24년 자전거 교통사고 다발지역_서울.xlsx** (111행 × 8컬럼):

| 컬럼명 | 설명 | 활용 가능 여부 |
| --- | --- | --- |
| `지점` | 구+동+번호 (예: "서울 중구1") | 구 이름 추출 가능 |
| `위치설명` | 도로명 근처 묘사 (예: "중로 1가 (청구로 1호선 입구)") | Geocoding으로 좌표 변환 가능 |
| `발생건수` | 사고 발생 횟수 | **핵심 피처** |
| `사고건수` | - | 발생건수와 유사 |
| `사망자수` | 사망자 수 | 가중치 부여 가능 |
| `부상자수` | 부상자 수 | **핵심 피처** |
| `중상자수` | 중상자 수 | 위험도 가중치 |
| `경상취급자수` | 경상 처리자 수 | 참고용 |

> ⚠️ **좌표 컬럼 없음** — 위치설명 텍스트를 Geocoding API로 변환해야 Spatial Join 가능.

**다발지분석-24년 보행자 교통사고 다발지역_경기도.xlsx** (94행 × 8컬럼):
동일한 컬럼 구조. 지점명이 경기도 시군구 기준 (예: "용인 처인구 모현읍1").

### 보행자 사고 데이터 사용 여부 판단

| 항목 | 판단 |
| --- | --- |
| 서울 자전거 사고 (111건) | **사용** — 자전거 서비스에 직접 관련, safety_index 타겟으로 반영 |
| 경기도 보행자 사고 (94건) | **제한적 사용** — 자전거 사고 아니지만 도로 위험 구간 참고용으로 활용 가능 (가중치 0.3 이하 적용) |
| 경기도 자전거 사고 | **미확보** — 경기도 오픈API 명세서만 있고 실제 데이터 없음. API 호출 필요 |

### 사고 데이터 활용 방법 (safety_index 개선)

```python
# 1단계: 위치설명 → Geocoding → (lat, lon) 변환
# 2단계: 도로 세그먼트와 Spatial Join (반경 200m 이내 사고 집계)
# 3단계: 사고 위험도 지수 계산

danger_score = (발생건수 × 1.0 + 사망자수 × 5.0 + 중상자수 × 2.0 + 부상자수 × 1.0)
# MinMaxScaler 정규화 후 safety_index에 반영
# safety_index_v2 = safety_index × 0.5 + (1 - danger_score_norm) × 0.5
```

### 경기도 사고 데이터 보완 방향

- TAAS 사이트에서 경기도 자전거 사고 다발지 데이터 별도 수집 필요
- 서울 사고 데이터(`자전거+교통사고_서울.csv`)를 Spatial Join하여 safety_index 타겟 품질 향상
- 경기도는 보행자 사고 데이터를 우선 사용하되, 가중치 낮게 적용하거나 별도 위험 피처로 분리

---

## 8. 모델 파이프라인 단계 정리 (실행 순서)

### 안전 모델 완성 파이프라인

```text
[실행 순서]
STEP 0  python kride-project/preprocess_road.py
        → data/raw_ml/road_clean_v2.csv (서울+경기 자전거도로, 복합키 road_id)

STEP 1  python kride-project/build_safety_model.py
        → models/safety_regressor.pkl   (RandomForestRegressor, 연속 safety_score)
        → models/safety_classifier.pkl  (RandomForestClassifier, 3등급 danger_level)
        → models/safety_scaler.pkl      (MinMaxScaler, 추론 시 피처 정규화용)
        → models/safety_meta.pkl        (features 목록, q33/q66, R², F1)
        → data/raw_ml/district_danger.csv (구별 위험도 참고 테이블)

STEP 2  python kride-project/build_tourism_model.py     ← 다음 작성 예정
        → models/tourism_scaler.pkl     (MinMaxScaler)
        → data/raw_ml/road_scored.csv   (safety_score + tourism_score + final_score 포함)

STEP 3  streamlit run kride-project/streamlit_app/app.py  ← 이후 작성
```

### 입력 데이터 의존 관계

```text
전국자전거도로표준데이터.csv ──→ preprocess_road.py ──→ road_clean_v2.csv
다발지분석-24년 자전거교통사고다발지역_서울.xlsx ──→ build_safety_model.py
road_clean_v2.csv ──────────────────────────────────→ build_safety_model.py
road_features.csv ──────────────────────────────────→ build_tourism_model.py
safety_regressor.pkl ───────────────────────────────→ build_tourism_model.py (final_score 계산)
```

### 출력 모델 역할

| 파일 | 역할 | 출력 형태 |
| --- | --- | --- |
| `safety_regressor.pkl` | 도로 세그먼트 안전점수 예측 | 0~1 연속값 (지도 색상) |
| `safety_classifier.pkl` | 위험등급 분류 | 0/1/2 (마커 색상) |
| `safety_scaler.pkl` | 추론 시 피처 스케일링 | transform() |
| `tourism_scaler.pkl` | 관광점수 정규화 | transform() |

---

## 10. 남은 데이터 이슈 요약

| 이슈 | 심각도 | 개선 방법 |
| --- | --- | --- |
| 좌표 결측으로 학습 데이터 8.1% 수준 | 높음 | GeoJSON 원본 또는 지오코딩으로 보완 |
| safety_index가 사고 데이터 미반영 | 높음 | 서울 사고 CSV Spatial Join |
| 경기도 자전거 사고 다발지 없음 | 중간 | TAAS에서 별도 수집 |
| 경기도 레저스포츠 POI 타임아웃 | 낮음 | 재수집 |
| `road_type` 결측 979행 | 낮음 | 최빈값 대체 또는 제외 |

---

## 11. 지도 API 비교 및 선택 근거

| 항목 | 네이버 지도 API | 카카오맵 API | Leaflet (react-leaflet) |
| --- | --- | --- | --- |
| 개인 키 발급 | 가능 (네이버 클라우드) | 비즈니스 앱 등록 필요 | 불필요 (오픈소스) |
| 한국 도로 정확도 | 높음 | 높음 | 보통 (OSM 기반) |
| 무료 쿼터 | 넉넉함 | 제한적 | 무제한 |
| React 연동 | `@navermaps/react-naver-map` | `react-kakao-maps-sdk` | `react-leaflet` |
| **현재 단계 채택** | **1순위** | 제외 | 네이버 실패 시 대안 |

> **결론**: 카카오맵은 비즈니스 앱 등록 요건으로 현 단계에서 제외. 네이버 지도로 먼저 구현하고 문제 발생 시 Leaflet으로 전환.

---

## 12. 구(區) 단위 필터링 방법

현재 `road_features.csv`에는 구 이름 컬럼이 없고 `start_lat`/`start_lon`만 있음. 두 가지 방법으로 구 이름을 추가할 수 있음.

### 방법 A: 행정동 경계 GeoJSON (권장)

```python
import geopandas as gpd

# 서울시 구 경계 GeoJSON (서울 열린데이터 광장 제공)
gu_boundary = gpd.read_file("seoul_gu.geojson")  # EPSG:4326

gdf_road = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.start_lon, df.start_lat))
gdf_road = gdf_road.set_crs("EPSG:4326")

# 공간 조인으로 구 이름 추가
gdf_with_gu = gpd.sjoin(gdf_road, gu_boundary[["SIG_KOR_NM", "geometry"]], how="left")
```

- 데이터 출처: 서울 열린데이터광장 → "서울시 자치구 경계"
- 경기도는 시·구 경계 GeoJSON 별도 수집 필요

### 방법 B: 역지오코딩 API (간단하지만 API 호출 비용 발생)

```python
# 네이버 지도 역지오코딩 API
# coords=경도,위도 → 행정구역명 반환
```

- 1,647행 전체 호출 시 API 쿼터 소모 주의

### Streamlit 적용 방안

```python
# 사이드바 필터 예시
gu_list = sorted(df["gu_name"].dropna().unique())
selected_gu = st.sidebar.multiselect("구 선택", gu_list, default=gu_list)
df_filtered = df[df["gu_name"].isin(selected_gu)]
```

---

## 13. 다음 주 리뷰 데이터 수집 계획

**목표**: 사용자 리뷰 기반으로 관광지 quality 및 도로 환경 피처 보강

### 수집 대상

| 소스 | 데이터 | 활용 방안 |
| --- | --- | --- |
| 네이버 지도 리뷰 | 관광지/명소 별점, 리뷰 텍스트 | tourism_score 정교화 (감성 분석) |
| 카카오맵 리뷰 | 장소 리뷰, 혼잡도 | 관광지 품질 가중치 |
| 네이버 지도 로드뷰 | 자전거 도로 환경 사진 | 도로 상태 피처 (비정형) |

### 수집 방법

- 네이버 지도 검색 API (공식): 장소명 → 리뷰 수, 별점
- 카카오 로컬 API (공식): 키워드 검색 → 장소 정보
- Selenium / Playwright: 공식 API에서 제공 안 되는 리뷰 텍스트 크롤링 (robots.txt 준수)

### 활용 방안

```text
리뷰 감성 분석 (긍정/부정 비율)
  → tourism_score에 sentiment_weight 추가
  → 별점 낮은 관광지는 tourism_score 하향 조정

자전거 도로 관련 키워드 ("자전거", "라이딩", "도로 좁음" 등)
  → safety_index 보정 피처로 활용 (다음 주 모델 업데이트)
```

---

## 14. 서비스 목표 vs 현재 문서 갭 분석 (2026-04-08)

### 프로젝트 목표 5가지

1. 자전거 여행
2. 안전성
3. 경로탐지
4. 관광지/여행 추천
5. 자전거 이용자 편의 제공

### 목표별 현재 반영 상태

| 목표 | 반영 여부 | 현재 상태 | 보완 방향 |
| --- | --- | --- | --- |
| 안전성 | 완료 | safety_index_v2, RF 모델, TabNet 계획 | 딥러닝 고도화(Phase 5) |
| 관광지/여행 추천 | 완료 | tourism_score, POI Spatial Join | 감성분석 보정(Phase 4) |
| 편의 제공 | 부분 | facility_clean.csv 수집, tourism bonus만 반영 | 지도 마커 레이어로 시각화 필요 |
| 자전거 여행 | 간접 | 도로 세그먼트 점수화로 간접 커버 | 여행 코스 단위 추천 추가 필요 |
| **경로탐지** | **미반영** | 없음 | **Phase 3-5에서 신규 구현** |

### 가장 큰 갭 — 경로탐지

현재 문서는 도로 세그먼트 **점수화**에 집중되어 있으며, 출발지 → 목적지 **경로 탐색** 기능이 전혀 없음.

- 도로 세그먼트들이 그래프로 연결되어 있지 않음
- 사용자가 "A에서 B까지 안전한 자전거 경로"를 요청할 수 없음
- 여행 코스(순환 루트) 생성 로직 없음

**해결책**: `road_scored.csv`의 세그먼트를 networkx 그래프로 연결하고, Dijkstra 알고리즘으로 최적 경로 탐색 (plan.md Phase 3-5 참조)

### 편의시설 미활용 문제

`facility_clean.csv` (3,368행) 가 tourism_score의 +0.1 보너스로만 사용됨.
사용자 관점에서 필요한 것: **경로 위 편의시설 위치를 지도에서 직접 확인**

- 공기 주입소, 자전거 수리점, 대여소, 화장실 등 타입별 마커
- 경로/코스 탐색 시 반경 500m 내 편의시설 자동 조회

### 추가 권장 기능 (plan.md 반영 완료)

| 기능 | 우선순위 | 구현 위치 |
| --- | --- | --- |
| 경로탐지 (`/api/route`) | 필수 | Phase 3-5 |
| 여행 코스 생성 (`/api/course`) | 필수 | Phase 3-5 |
| 편의시설 지도 레이어 (`/api/facilities`) | 권장 | Phase 3-5 |
| 날씨 연동 (KMA API) | 권장 | Phase 3-6 |

---

## 15. 딥러닝 단기 업그레이드 계획 (내일까지 마감)

### 15-1. TabNet — 안전 예측 고도화

**역할**: 현재 RandomForest(R²=0.9539)를 TabNet으로 교체 또는 앙상블

```text
입력 피처: width_m, length_km, district_danger,
           road_attr_score, start_lat, start_lon

출력:
  ① safety_score (0~1 연속값)  ← TabNetRegressor
  ② danger_level (0/1/2 등급)  ← TabNetClassifier
```

**RandomForest 대비 TabNet 장점**:

| 항목 | RandomForest | TabNet |
| --- | --- | --- |
| 피처 중요도 시각화 | 막대 그래프 | Attention Map (스텝별) |
| 위경도 비선형 학습 | 보통 | 우수 (attention 메커니즘) |
| 딥러닝 요건 충족 | X | O |
| 데이터 추가 수집 필요 | X | X (기존 road_features.csv 사용) |

**구현 난이도**: 낮음 — `pytorch-tabnet` 라이브러리 하나로 완결, plan.md Phase 5 코드 그대로 실행

**예상 소요 시간**: 1~2시간

---

### 15-2. KR-FinBERT — 관광지 감성분석

**역할**: 관광지 리뷰 텍스트 → 긍정/부정 확률 → `tourism_score` 보정

```python
from transformers import pipeline
sentiment = pipeline("text-classification", model="snunlp/KR-FinBert-SC")
# Pretrained 모델 → 파인튜닝 없이 zero-shot 추론 가능
```

**예상 소요 시간**: 3~4시간

---

### 15-3. 따릉이 수요 LSTM — 혼잡도 예측

**역할**: 시간대 + 날씨 피처로 자전거 수요 예측 → 현재 `tourist_count` proxy 대체

**예상 소요 시간**: 4~5시간

---

### 15-4. 데이터셋 출처 정리

| 모델 | 데이터셋 | 출처 | 접근 방법 |
| --- | --- | --- | --- |
| TabNet (안전 예측) | `road_features.csv`, `district_danger.csv` | 이미 보유 | 추가 수집 불필요 |
| KR-FinBERT (감성분석) 검증용 | 네이버 영화 리뷰 (NSMC) 20만 건 | github.com/e9t/nsmc | `wget` 즉시 다운 |
| KR-FinBERT (감성분석) 관광 도메인 | 한국어 감성 및 기반 의견 분석 데이터 | AI Hub (aihub.or.kr) → 자연어 분야 | 회원가입 후 신청 |
| LSTM (따릉이 수요) | Seoul Bike Sharing Demand (날씨+시간대+계절 포함) | Kaggle | Kaggle 계정으로 즉시 다운 |
| LSTM (따릉이 원본) | 서울 공공자전거 따릉이 이용정보 | 공공데이터포털 (data.go.kr) | 바로 다운로드 |

---

### 15-5. AI Hub 데이터 신청 시 주의사항

**잘못된 선택**: `영상이미지` 분야 → `한국인 감정인식을 위한 복합 영상`
→ 얼굴 표정 인식용 영상 데이터로 KRide 감성분석 용도에 맞지 않음

**올바른 선택**: `자연어` 분야 → `한국어 감성 및 기반 의견 분석 데이터` 또는 `여행 도메인 특화 언어 모델 학습 데이터` 검색

**문의하기 내용 (템플릿)**:

```text
안녕하세요.

자전거 도로 안전 및 여행 추천 서비스(K-Ride) 개발을 위해
딥러닝 모델 학습 데이터를 탐색 중입니다.

문의 사항:
1. 해당 데이터셋의 라이선스 및 비상업적 연구/교육 목적 활용 가능 여부
2. 데이터 형식(파일 구조, 라벨 형식) 확인
3. 데이터 다운로드 절차 및 소요 기간

활용 목적: 대학 프로젝트 - 관광지/자전거 도로 관련 리뷰 텍스트
           감성분석 모델(KR-FinBERT) 학습 및 검증용

감사합니다.
```

---

### 15-6. 단기 구현 순서 (오늘~내일 마감 기준)

```text
오늘 오전  → TabNet 구현 (road_features.csv 그대로 사용, 1~2시간)
오늘 오후  → KR-FinBERT 감성분석 (NSMC 다운 → zero-shot 추론, 3~4시간)
오늘 저녁  → 따릉이 LSTM (Kaggle 데이터, 4~5시간)
내일 오전  → 세 모델 결과를 Streamlit 앱에 통합 시각화
```

---

## 16. AI Hub 데이터셋 분석 — KRide 딥러닝 활용 가능성

> 분석 기준: CNN/이미지, 시계열, 예측(추론) 3가지 딥러닝 관점

### 16-1. 데이터셋별 기본 정보

| # | 데이터셋명 | 유형 | 크기 | 구축년도 |
| --- | --- | --- | --- | --- |
| A | 관광지 소개 다국어 번역 데이터 | 텍스트 + 이미지 | 1.77GB | 2022 |
| B | 국내 여행로그 데이터(수도권) | 텍스트 + 이미지 + GPS | 94.9GB | 2023 |
| C | 생성형AI 한국어 다중 이벤트 추출 데이터 | 텍스트 | 225MB | 2023 |
| D | 관광분야 이미지-텍스트 쌍 데이터 | 이미지 + 텍스트 | 770GB | 2023 |
| E | 생성형AI 한국어 SNS 멀티턴 대화 데이터 | 텍스트 | 547MB | 2023 |
| F | K-Culture 관광 콘텐츠 특화 일본어 말뭉치 | 텍스트 + 이미지 | 938MB | 2023 |

---

### 16-2. CNN / 이미지 딥러닝 활용

#### B. 국내 여행로그 데이터(수도권) ★★★ 최우선 추천

```text
활용: 수도권 여행지 사진 18,229장 → EfficientNet-B0 전이학습
      장소 유형 4클래스 분류 (자연/도심/문화/레저)

연결: 분류 결과 → tour_poi.csv contentTypeId 매핑
      → tourism_score 계산 시 유형별 가중치 정교화

장점: GPS + 사진 동시 보유, 수도권 한정 (KRide 범위와 일치)
주의: 전체 94.9GB 중 사진만 선별 다운로드 권장 (TS_photo 샘플 1~2GB)
```

#### D. 관광분야 이미지-텍스트 쌍 데이터 ★★

```text
활용: 안내도/이정표/시간표 이미지 280,000장 → CNN + OCR 파이프라인
      관광지 안내 이미지 → 텍스트 추출 → tourism_score 보정 피처화

카테고리 분포:
  안내도  33.7% (94,400건)
  입장권  18.4% (51,584건)
  이정표  17.5% (48,915건)
  시간표   8.9% (24,768건)

주의: 770GB 대용량 → 내일 마감에는 전체 다운 불가
      서브라벨링 10,000장만 우선 활용 권장
```

---

### 16-3. 시계열 딥러닝 활용 (LSTM / GRU)

#### B-2. 국내 여행로그 데이터(수도권) — 시계열 ★★★ 최우선 추천

```text
활용: GPS 이동 경로 (타임스탬프 포함) → 시간대별 방문 패턴 예측

입력 시퀀스: [날짜, 시간대, 요일, 구역ID, 방문자수_t-1, t-2, t-3]
출력: 다음 시간대 방문자 수 → 혼잡도 예측

핵심 파일:
  TL_gps_data.zip  (89.55MB)  ← 학습용 GPS 경로
  VL_gps_data.zip  (10.81MB)  ← 검증용 GPS 경로
  TL_csv.zip        (2.89MB)  ← 방문지/소비 내역 라벨

KRide 연결:
  현재 tourist_count(정적) → LSTM 예측 혼잡도(동적)로 교체
  tourism_score에 실시간 혼잡 가중치 반영 가능
  혼잡 예상 구간 회피 경로 추천으로 확장
```

#### A. 관광지 소개 다국어 번역 데이터 (간접 활용)

```text
활용: 관광지 설명 텍스트 시퀀스 → Seq2Seq 또는 텍스트 분류
      POI 유형(travelType) 예측 모델 학습

주요 컬럼:
  POI_id       : 관광지 고유 ID (tour_poi.csv와 매핑 가능)
  travelType   : 관광 타입 (분류 타겟)
  k_context    : 관광지 설명 텍스트 (입력 피처)
  k_context_wordNum: 텍스트 길이 (0~200 어절)

KRide 연결:
  KR-FinBERT 감성분석 학습/검증 데이터로 활용
  관광지 설명 텍스트 → 긍정/부정 점수 → tourism_score 보정
```

---

### 16-4. 예측 / 추론 딥러닝 활용

#### E. 생성형AI 한국어 SNS 멀티턴 대화 데이터 ★★

```text
활용: 자전거 여행 관련 대화 학습 → 여행지 추천 챗봇 구현

구성: 196,235 대화 세션, 3,246,886 발화
      2인/3인 화자, 카테고리/키워드/화행 라벨 포함

KRide 연결:
  "한강 근처 자전거 타기 좋은 곳 추천해줘" 형태의 대화 생성 모델
  기존 추천 API (/api/recommend) + 챗봇 인터페이스 확장
  Streamlit 앱에 대화형 추천 탭 추가 가능

주의: 여행/자전거 도메인 특화 파인튜닝 필요
```

#### F. K-Culture 관광 콘텐츠 특화 일본어 말뭉치 ★

```text
활용: 레저스포츠 카테고리(18,210건) → 관광지 유형 분류 모델

카테고리 중 KRide 관련:
  레저스포츠  18,210건  ← 자전거/레저 관련 POI 분류에 활용 가능
  자연관광    46,011건  ← 자연 경로 tourism_score 보정

주의: 일본어 텍스트 → 한국어 서비스에 직접 사용 어려움
      카테고리 라벨만 추출해서 분류 모델 학습에 활용 권장
```

#### C. 생성형AI 한국어 다중 이벤트 추출 데이터 ★★ (재평가)

```text
초기 평가: 낮음 → 재평가: 중간 (아이디어 반영 후 상향)

도메인 분포:
  스포츠  14% → 야구/축구 경기장 주변 혼잡도 예측에 활용
  연예    13% → 아이돌 콘서트/공연 장소 근처 경로 위험도 상향
  라이프  15% → 날씨/재난 이벤트 추출 가능
  부동산  15% → 도로 공사/개발 공사 이벤트 추출 (경로 우회 필요)

이벤트 JSON 구조:
  entity_value  : 장소/팀/인물 이름 (예: "잠실 올림픽경기장", "BTS")
  event_type    : 이벤트 유형 (스포츠_경기, 공연_행사 등)
  trigger_value : 이벤트 트리거 동사 (개최하다, 열리다 등)

KRide 활용 방향 → 섹션 17 참조
```

---

### 16-5. KRide 딥러닝 적용 우선순위 종합

| 순위 | 데이터셋 | DL 모델 | KRide 기능 | 난이도 | 내일 마감 적합 |
| --- | --- | --- | --- | --- | --- |
| 1 | 여행로그(수도권) GPS | LSTM | 시간대별 혼잡도 예측 → tourism_score 동적 반영 | 중 | ✅ (GPS만 89MB) |
| 2 | 관광지 소개 번역 | KR-FinBERT | 관광지 설명 텍스트 감성분석 → tourism_score 보정 | 낮 | ✅ (1.77GB) |
| 3 | 여행로그(수도권) 사진 | CNN (EfficientNet) | 여행지 유형 분류 → tourism_score 가중치 정교화 | 중 | ⚠️ (샘플만) |
| 4 | 관광분야 이미지-텍스트 | CNN + OCR | 관광 안내 이미지 분류 | 높 | ❌ (770GB) |
| 5 | SNS 멀티턴 대화 | Seq2Seq | 자전거 여행 추천 챗봇 | 높 | ❌ (파인튜닝 필요) |
| 6 | 다중 이벤트 추출 | NLP | 관련성 낮음 | - | ❌ |

### 16-6. 내일 마감 기준 실행 계획

```text
[즉시 다운 가능]
  관광지 소개 번역 (1.77GB)
    → KR-FinBERT 감성분석 학습/검증 데이터로 바로 사용
    → k_context(관광지 설명) + travelType(분류 타겟) 컬럼 활용

[GPS만 선택 다운 — 89MB]
  여행로그(수도권) TL_gps_data.zip + TL_csv.zip
    → LSTM 시간대별 방문 패턴 학습
    → tourism_score 동적 보정 파이프라인 구성

[사진 샘플만 — 1~2GB]
  여행로그(수도권) TS_photo 일부
    → EfficientNet-B0 CNN 장소 유형 분류 (빠른 검증용)
```

---

## 17. 신규 아이디어 — 이벤트 감지 + 소비 예측 + 날씨 예측

> 아이디어 출처: 여행로그 소비/시군구 데이터 + 이벤트 추출 텍스트 데이터 활용
>
> 핵심 개념: 자전거 경로 탐색 시 **갑작스런 이벤트(스포츠/콘서트/재난/날씨)** 를 감지하고,  
> 날짜+경로 입력 시 **예상 날씨와 혼잡도**를 미리 알려주는 지능형 경로 추천

---

### 17-1. 아이디어 1 — 시군구 소비 패턴 예측

**배경**: 여행로그 데이터에 시군구 코드(TC_SGG) + 소비 테이블 4종 존재

```text
소비 테이블 구조:
  TN_MVMN_CONSUME_HIS   → 이동수단 소비 (교통비)
  TN_LODGE_CONSUME_HIS  → 숙박 소비
  TN_ADV_CONSUME_HIS    → 사전 소비 (여행 전 준비)
  TN_ACTIVITY_CONSUME_HIS → 활동 소비 (현장 지출)

연결 고리:
  TC_SGG (시군구 코드) ↔ TN_GPS_COORD ↔ TN_MOVE_HIS
  → 어느 구에서 얼마나 썼는지 집계 가능
```

**딥러닝 모델**: TabNet Regressor (정형 데이터 예측)

```python
# 입력 피처
features = [
    'sgg_code',          # 시군구 코드
    'travel_duration',   # 여행 시간(시간 단위)
    'distance_km',       # 이동 거리
    'companion_cnt',     # 동반자 수
    'season',            # 계절 (1~4)
    'day_of_week',       # 요일
]
# 타겟: 활동 소비 금액 예측
target = 'activity_consume_amount'

# KRide 연결
# → 경로 선택 시 "이 경로 예상 지출: 약 15,000원" 표시
# → 시군구별 평균 지출 히트맵으로 지도에 오버레이
```

**KRide 서비스 기능**: 경로 추천 결과에 **예상 여행 비용** 함께 표시

---

### 17-2. 아이디어 2 — 이벤트 감지 → 경로 혼잡도 실시간 알림

**배경**: 이벤트 추출 데이터(dataSetSn=71729)의 스포츠/연예 도메인 활용

```text
이벤트 예시:
  스포츠: "잠실야구장에서 LG-KIA 경기 개최"
           → entity_value: "잠실야구장"
           → event_type: "스포츠_경기"
  연예:   "KSPO돔에서 아이돌 콘서트 열려"
           → entity_value: "KSPO돔"
           → event_type: "공연_행사"
  재난:   "강남구 도림천 범람 위험"
           → entity_value: "도림천", "강남구"
           → event_type: "재난_사고"
```

**파이프라인**:

```text
뉴스 텍스트
    │
KLUE-BERT NER  ← 이벤트 추출 데이터로 파인튜닝
    │
장소명 추출 (entity_value)
    │
Geocoding (장소명 → lat/lon)
    │
Spatial Join (경로 세그먼트 반경 1km 이내 이벤트 감지)
    │
혼잡도 가중치 적용
  스포츠/콘서트 → w_tourism 하향 (혼잡 우회)
  재난/공사     → safety_score 하향 (위험 구간 회피)
    │
경로 재탐색 or 경고 알림 출력
```

**딥러닝 모델**: KLUE-BERT (NER 파인튜닝) + 이진 분류 (경로 영향 여부)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# KLUE-BERT NER 파인튜닝
model = AutoModelForTokenClassification.from_pretrained(
    "klue/bert-base",
    num_labels=5  # O, B-LOC, I-LOC, B-EVT, I-EVT
)
# 이벤트 추출 데이터 17,625건으로 파인튜닝
# entity_value(장소) + event_type(유형) → NER 라벨 생성
```

**KRide 서비스 기능**: 경로 탐색 시 **"이 경로 근처 오늘 이벤트: 잠실야구장 경기 (18:00)"** 알림 표시

---

### 17-3. 아이디어 3 — 날짜 + 경로 → 날씨 예측 (LSTM 시계열)

**배경**: 날짜와 출발 시군구를 입력하면 해당 날의 날씨를 예측 → safety_score 자동 보정

**데이터 출처**:

| 데이터 | 출처 | 접근 방법 |
| --- | --- | --- |
| 기상청 과거 날씨 (기온/강수/풍속) | 기상자료개방포털 (data.kma.go.kr) | 무료 다운로드 |
| 단기예보 API | 공공데이터포털 기상청 단기예보 | API 키 발급 |
| 여행로그 GPS + 날씨 매핑 | AI Hub 여행로그 + KMA 매핑 | 날짜 기준 조인 |

**딥러닝 모델**: LSTM (시계열 날씨 예측)

```python
import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 맑음/흐림/비

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 입력 시퀀스 (과거 14일)
# [월, 일, 요일, 기온, 강수량, 풍속, 습도, 시군구코드]
# 출력: 다음 날 날씨 3분류 (맑음=0 / 흐림=1 / 비·눈=2)
```

**날씨 → safety_score 연동**:

```python
weather_safety_penalty = {0: 0.0, 1: -0.1, 2: -0.3}  # 비·눈이면 안전점수 -0.3

safety_score_weather = max(
    safety_score + weather_safety_penalty[predicted_weather], 0.0
)
# 최종 경로 추천 시 날씨 보정 safety_score 사용
```

**KRide 서비스 기능**: "선택하신 날짜({date})의 {시군구} 예상 날씨: 비 (안전점수 자동 하향 조정)"

---

### 17-4. 세 아이디어 통합 파이프라인

```text
사용자 입력: 출발지(lat/lon) + 도착지(lat/lon) + 날짜

                    ┌─────────────────────┐
                    │   날짜 입력          │
                    └──────┬──────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   날씨 예측 LSTM    이벤트 감지        소비 예측 TabNet
   (맑음/흐림/비)    KLUE-BERT NER     (예상 지출 금액)
          │                │                │
   safety_score       혼잡도 가중치     경로 비용 표시
   자동 보정           경고 알림 생성
          │                │                │
          └────────────────┼────────────────┘
                           ▼
              최종 경로 추천 + 통합 정보 카드
              ┌──────────────────────────────┐
              │ 경로: 한강로 → 잠실          │
              │ 안전점수: 0.72 (비 예보 -0.1)│
              │ 예상 날씨: 흐림              │
              │ 주변 이벤트: 잠실야구장 경기 │
              │ 예상 지출: 약 12,000원       │
              └──────────────────────────────┘
```

---

### 17-5. 구현 가능성 및 모델 요약

| 아이디어 | 모델 유형 | 데이터 출처 | 난이도 | 내일 마감 |
| --- | --- | --- | --- | --- |
| 소비 예측 | TabNet (정형/예측) | 여행로그 TN_ACTIVITY_CONSUME_HIS | 낮음 | ✅ |
| 날씨 예측 | LSTM (시계열) | KMA 과거 날씨 + data.kma.go.kr | 중간 | ✅ |
| 이벤트 감지 | KLUE-BERT NER (텍스트) | 이벤트 추출 데이터 17,625건 | 높음 | ⚠️ zero-shot 우선 |

### 17-6. 내일 마감 기준 단계별 구현 전략

```text
STEP 1 (1~2시간)
  날씨 예측 LSTM
  → data.kma.go.kr에서 서울 과거 날씨 CSV 다운 (무료, 즉시 가능)
  → 14일 시퀀스 → 다음날 날씨 3분류 학습
  → safety_score 자동 보정 파이프라인 연결

STEP 2 (2~3시간)
  소비 예측 TabNet
  → 여행로그 TL_csv.zip (2.89MB) 다운
  → 시군구 + 이동거리 + 동반자수 → 활동 소비 금액 예측
  → 경로 추천 결과 카드에 예상 지출 추가

STEP 3 (여유 시 / zero-shot 우선 적용)
  이벤트 감지
  → KLUE-BERT zero-shot classification으로 뉴스 이벤트 유형 분류
  → 파인튜닝은 시간 여유 시 추가 (이벤트 추출 데이터 17,625건 사용)
  → 장소명 geocoding → Spatial Join → 경로 근처 이벤트 알림
```

---

## 18. build_route_graph.py 실행 결과 (2026-04-08)

### 실행 결과 요약

```text
STEP 1: road_scored.csv 로드
  shape: (1647, 19)
  컬럼: start_lat, start_lon, end_lat, end_lon, length_km, width_m,
        road_type, is_official, is_wide_road, safety_index,
        tourist_count, cultural_count, leisure_count, facility_count,
        tourism_score, district_danger, road_attr_score, safety_score, final_score
  좌표 결측: 없음 (1,647 → 1,647행 유지)

STEP 2: 그래프 구성
  추가된 엣지: 1,370개
  스킵된 세그먼트: 186개 (self-loop 또는 zero-length)
  노드 수: 2,245
  엣지 수: 1,370

STEP 3: 연결성 분석
  연결 컴포넌트 수: 881개
  최대 컴포넌트 노드 수: 32개 (전체의 1.4%)
  최대 컴포넌트 엣지 수: 31개
  G_main: 32 노드 / 31 엣지

STEP 4: route_graph.pkl 저장 완료
  → kride-project/models/route_graph.pkl
```

### 핵심 문제: 그래프 단절 심각

연결 컴포넌트 881개, 최대 연결 컴포넌트가 전체의 **1.4%** 에 불과함.
이 상태에서는 G_main으로 경로 탐색 시 커버 가능한 구간이 극히 제한적.

**원인 분석**:

- `road_scored.csv`의 도로 세그먼트들이 좌표 단위로 연결되지 않음
- 동일 구간이라도 좌표 정밀도 차이(소수점 4자리 반올림 기준 11m)로 인해 노드 병합 실패
- 자전거 도로 원천 데이터가 노선별 독립 행으로 기록되어 교차점 연결 정보 없음

**해결 방향 (Phase 3-5 보완)**:

| 방법 | 설명 | 난이도 |
| --- | --- | --- |
| COORD_PRECISION 완화 | 소수점 3자리(~111m)로 낮춰 더 많은 노드 병합 | 낮음 |
| OSM 도로 데이터 보완 | OpenStreetMap 자전거 도로 네트워크로 대체 or 보완 | 중간 |
| 구(區) 단위 서브그래프 | 전국 탐색 포기 → 특정 구 내 경로 탐색만 구현 | 낮음 |

**MVP 임시 전략**: G 전체(2,245 노드)를 지도 시각화 용도로만 쓰고,  
경로 탐색은 COORD_PRECISION을 3으로 낮춰 재빌드 후 재평가.
