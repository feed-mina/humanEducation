# K-Ride ML 기술 리서치

## 현재 진행 상황 요약 (2026-04-02 기준)

- ✅ 전처리 완료: `road_features.csv` (1,647행, 14컬럼)
- ✅ Spatial Join 완료: 관광지(1km), 편의시설(500m) 피처 생성
- ✅ 회귀 모델 실험 완료 → 한계 확인, **분류로 전략 전환**
- ✅ **pkl 5종 생성 완료** (`safety_classifier`, `safety_regressor`, `safety_scaler`, `safety_meta`, `tourism_scaler`)
- ✅ `road_scored.csv` 생성 (1,647행, 19컬럼) — safety/tourism/final_score 포함
- ✅ `streamlit_kride.py` 작성 완료 (탭 3종: 안전등급 예측 / 경로 추천 Top-10 / 데이터 탐색)
- ⏳ **다음 목표: Streamlit Cloud 배포 + model_comparison.md 작성 (Day 5)**

---

## 1. 데이터 소스

### 1.1 TAAS (교통사고분석시스템)

- **URL:** <https://taas.koroad.or.kr>
- **데이터:** 자전거 관련 교통사고 (위치, 유형, 피해 정도)
- **현황:** `data/raw_accident_data/` 에 저장 완료
- **주요 컬럼:** 사고일시, 위도/경도, 사망자수, 중상자수, 경상자수, 사고유형
- **⚠️ Spatial Join 미완료** → P2 (시간 부족 시 2주차 이월)

### 1.2 공공데이터포털 자전거도로

- **URL:** <https://www.data.go.kr>
- **현황:** `road_features.csv` 완성 (1,647행, 14컬럼)
- **핵심 피처:** width_m, length_km, is_wide_road, safety_index, tourist_count, cultural_count, leisure_count, facility_count

### 1.3 한국관광공사 TourAPI

- **End Point:** `https://apis.data.go.kr/B551011/KorService2`
- **수집 완료:** 서울+경기 2,529건 (관광지 1,775 / 문화시설 671 / 레저스포츠 83)
- **활용:** tourist_count, cultural_count, leisure_count 피처로 Spatial Join 완료

---

## 2. 전략 전환: 회귀 → 분류

### 2.1 회귀 실패 원인 분석

| 모델 | R² | 원인 |
| ---- | -- | ---- |
| LinearRegression | 0.0096 | safety_index가 width_m, length_km로만 계산 → 선형 관계 포착 불가 |
| 다중회귀 | 0.0635 | 피처 추가해도 타겟 자체의 분산 설명력 낮음 |
| RandomForest | 0.1890 | 현재 최고, 그러나 여전히 낮음 |
| 다항회귀 | 0.0132 | 과적합 없이 낮음 |

**핵심 문제:** `safety_index = f(width_m, length_km)` 공식 기반 타겟 → 입력 피처와 타겟이 동어반복 구조. TAAS 사고 데이터 없이는 회귀 성능 개선 어려움.

### 2.2 분류 전환 근거

- **3등급 구간화로 타겟 재정의:** 연속값 → 범주형
  ```python
  safety_grade = pd.qcut(safety_index, q=3, labels=[0, 1, 2])
  # 0=위험(하위33%), 1=보통(중간33%), 2=안전(상위33%)
  ```
- **분류는 불균형에 강건:** F1-macro로 평가 시 클래스 불균형 완화
- **사용자에게 직관적:** "이 경로는 안전/보통/위험" 출력이 수치보다 이해하기 쉬움

---

## 3. 모델 설계

### 3.1 모델 1 — 안전등급 분류 (safety_classifier.pkl)

**목적:** 도로 피처 입력 → 안전등급(0/1/2) 예측

**피처셋:**

### 3.1-A 사고 데이터 기반 안전지수 반영 설계 (구 단위 → 노선 단위 연결)

**핵심 문제**: 사고다발지 데이터는 구(시군구) 단위로 집계되어 있고, 도로 노선은 `노선명+시군구명` 복합키를 사용함.
→ 같은 구에 속한 노선들은 동일한 구 위험도를 공유하되, 노선별 도로 폭으로 미세 차등 가능.

#### 2단계 안전지수 계산 방식 (개선안 v2)

```python
# 1단계: 구(시군구) 단위 위험도 계산
danger_score = (발생건수 * 1.0 + 사망자수 * 5.0 + 중상자수 * 2.0 + 부상자수 * 1.0)
# 구별 집계 후 MinMaxScaler 정규화 → district_danger (0=안전, 1=위험)

# 2단계: road_id의 시군구명으로 매핑
df["district_danger"] = df["sigungu"].map(danger_by_sigungu)

# 3단계: 도로 속성 (기존 safety_index)
road_attr_score = width_m_norm * 0.7 + length_km_norm * 0.3

# 4단계: 통합 안전지수 (위험도는 반전: 1-danger)
safety_index_v2 = (1 - district_danger) * 0.6 + road_attr_score * 0.4
```

| 레이어 | 데이터 소스 | 가중치 | 역할 |
| --- | --- | --- | --- |
| 구 위험도 (district_danger) | 사고다발지 엑셀 (서울) | 0.6 | 지역 기반 위험 환경 반영 |
| 도로 속성 (road_attr_score) | 전국자전거도로 너비+길이 | 0.4 | 노선별 차등화 |

> **왜 회귀 대신 분류를 고려해야 하나**: 사고다발지 집계가 구 단위이기 때문에 노선 개별 회귀는 의미가 낮음. 구 위험등급(안전/보통/위험 3단계)을 먼저 분류한 뒤, 같은 등급 내에서 도로 폭으로 순위를 정하는 계층적 접근이 더 현실적임.

#### 분류 모델 적용 방안

```text
타겟: 위험등급 (0=안전 / 1=보통 / 2=위험)
  - 구별 사고 건수 기준 삼분위(33%/66%) → 등급 자동 부여
  - 같은 구 내: 도로 폭 < 1.5m → 한 단계 상향

피처:
  - district_danger (구 위험도 정규화값)
  - width_m (도로 폭)
  - is_wide_road (너비 2.0m 이상 여부)
  - length_km (노선 길이)
  - tourist_count (관광지 밀도 → 혼잡 가능성)

모델: RandomForestClassifier (현재 회귀 모델 대체 가능)
평가: F1-macro (3클래스 불균형 대응)
```

> **현재 데이터 한계**: 서울 사고다발지 111건만 있고 경기도 자전거 사고 데이터 미확보. 서울 25구 중 사고 기록이 있는 구만 위험도 적용, 나머지(경기 포함)는 도로 속성 점수만 사용.

### 3.2 위험등급 분류 (분류)
| 피처 | 설명 | 전처리 |
| ---- | ---- | ------ |
| `width_m` | 도로 너비 | StandardScaler |
| `length_km` | 도로 길이 | StandardScaler |
| `is_wide_road` | 너비 2m 이상 여부 | 그대로 (0/1) |
| `facility_count` | 500m 내 편의시설 수 | StandardScaler |
| `tourist_count` | 1km 내 관광지 수 | StandardScaler |
| `cultural_count` | 1km 내 문화시설 수 | StandardScaler |
| `road_type_OHE` | 도로 유형 원핫인코딩 | 결측 → 'unknown' |

**타겟:** `safety_grade` (0=위험, 1=보통, 2=안전)

**모델 선택:**

| 모델 | 채택 | 이유 |
| ---- | ---- | ---- |
| RandomForestClassifier | ✅ 채택 | 회귀에서도 최고 성능, 피처 중요도 제공 |
| LogisticRegression | 기준선 | 해석 용이, baseline |
| XGBoostClassifier | 비교용 | 성능 비교 후 채택 여부 결정 |

**평가:**
- F1-macro (클래스 불균형 대응)
- 목표: F1-macro > 0.6

**저장:**
```python
joblib.dump(model, 'kride-project/ml-server/models/safety_classifier.pkl')
joblib.dump(scaler, 'kride-project/ml-server/models/safety_scaler.pkl')
```

---

### 3.2 모델 2 — 관광 경로 추천 스코어 (tourist_scorer.pkl)

**목적:** 사용자 모드(안전/관광/균형) 선택 → 경로별 추천 점수 계산 → Top-10 출력

**접근법: 가중치 기반 점수 + MinMaxScaler**

```python
# 각 피처를 0~1로 정규화 (MinMaxScaler)
# tourist_scorer.pkl = MinMaxScaler 저장

weight_modes = {
    "safe":     {"safety_grade": 0.6, "tourist_count": 0.15, "cultural_count": 0.15, "facility_count": 0.1},
    "tourist":  {"safety_grade": 0.3, "tourist_count": 0.4,  "cultural_count": 0.2,  "facility_count": 0.1},
    "balanced": {"safety_grade": 0.4, "tourist_count": 0.3,  "cultural_count": 0.2,  "facility_count": 0.1},
}

# route_score = Σ(정규화된_피처 × 가중치)
```

**왜 ML 모델이 아닌가?**
- 추천 레이블(어떤 경로가 "좋은" 경로인지) 데이터 없음
- 사용자 선호도는 도메인 지식 기반 가중치가 더 설명 가능
- MinMaxScaler를 pkl로 저장 → 새 데이터에도 동일 정규화 적용 가능

---

## 4. Streamlit 통합 앱 설계 (streamlit_kride.py)

```
사이드바
└── 모드 선택: 안전 우선 / 관광 우선 / 균형

탭 1: 안전등급 예측
├── 입력: 도로폭(슬라이더), 도로길이, 편의시설 수
├── 처리: safety_scaler → safety_classifier 예측
└── 출력: 안전등급 (🟢안전 / 🟡보통 / 🔴위험)

탭 2: 경로 추천
├── 처리: tourist_scorer(MinMaxScaler) → 가중치 점수 계산
└── 출력: Top-10 경로 테이블 (점수, 도로폭, 관광지 수 등)

탭 3: 데이터 탐색
└── road_features 분포 차트 (히스토그램, 상관관계 히트맵)
```

---

## 5. 공간 데이터 처리 (Spatial Join) — 완료

### 완료 내용

```python
# 관광지 1km 반경 집계 (EPSG:5179 변환 후)
tourist_count  평균: 1.63
cultural_count 평균: 0.74
leisure_count  평균: 0.10

# 편의시설 500m 반경 집계
facility_count 평균: 1.09
```

### PostGIS 쿼리 (서버 측 — 2주차 연동용)

```sql
SELECT r.route_id, COUNT(a.accident_id) as accident_count
FROM bicycle_routes r
LEFT JOIN accidents a
  ON ST_DWithin(r.geom::geography, a.geom::geography, 100)
GROUP BY r.route_id;
```

---

## 6. YOLOv8 객체 탐지 — 2주차 이월

### 탐지 대상 클래스

| 클래스 | 설명 | 위험 가중치 |
| ------ | ---- | ----------- |
| pedestrian | 보행자 | 높음 |
| bicycle | 자전거 | 중간 |
| vehicle | 차량 | 높음 |
| obstacle | 장애물 (공사, 라바콘) | 높음 |

---

## 7. 피처 중요도 현황 (RandomForest 회귀 기준)

### 혼잡도 데이터 확보 방법

혼잡도는 실시간 데이터와 정적 대리 지표 두 가지로 나뉨. 현 MVP 단계에서는 정적 대리 지표로 근사.

#### 실시간 혼잡도 (향후)

| 소스 | API | 내용 | 비용 |
| --- | --- | --- | --- |
| 카카오 모빌리티 | 카카오맵 교통 API | 링크별 현재 소통 상태 | 유료 |
| TOPIS | 서울시 교통정보시스템 | 링크속도, 혼잡등급 | 무료(서울) |
| 공공데이터포털 | 실시간 도시데이터 | 생활인구, 이동인구 | 무료 |

#### 정적 대리 지표 (현재 MVP 적용 가능)

```text
1. tourist_count (관광지 수)
   → 관광지 밀도 높으면 주말 혼잡 가능성 높음
   → 이미 road_features.csv에 포함됨

2. facility_count (편의시설 수)
   → 자전거 보관소/대여소 집중 구간 = 이용자 많음

3. 따릉이 대여소 이용 통계 (서울시 공공데이터)
   → 대여소별 시간대별 이용 건수 → 노선 근접 대여소 혼잡도 proxy
   → 파일: "서울시+교통공사+지하철역+편의시설위치정보+자전거보관함+현황.xls" 활용 가능

4. 시간대 가중치 (정적)
   → 주말/공휴일: congestion_weight = 1.5
   → 평일 출퇴근(07-09, 18-20): congestion_weight = 1.3
   → 기타: congestion_weight = 1.0
```

#### Dijkstra 혼잡도 반영 방안 (MVP)

```python
# 현재 단계: tourist_count를 혼잡도 proxy로 사용
congestion_proxy = MinMaxScaler().fit_transform(df[["tourist_count"]])

# 엣지 가중치에 반영
edge_weight = safety_score * w1 + distance * w2 + congestion_proxy * w3
```

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


#### 서울/경기 별도 CSV vs 전국 CSV 비교 타당성 검토

> **결론: 두 소스는 데이터 정의 자체가 달라 직접 비교 부적절. 전국 표준데이터가 베이스라인, 서울시 별도 데이터는 보완재로 사용.**

| 항목 | 전국자전거도로표준데이터 | 서울시 별도 데이터 |
| --- | --- | --- |
| 관리 주체 | 국토교통부 공공데이터포털 | 서울시 열린데이터광장 |
| 서울 행 수 | **380행** | 수천 행 (차선형 포함) |
| 포함 범위 | 국가 표준 등록 **전용도로만** | 차선형 자전거도, 겸용도로 포함 |
| 경기도 행 수 | 4,939행 | 경기도 별도 수집 필요 |
| 좌표 포함 여부 | 부분 결측 (~32%) | 대부분 포함 |

**서울 380행이 적은 이유 (오류 아님)**:

- 전국표준데이터는 `자전거전용도로고시` 여부가 있는 공식 등록 노선만 포함
- 서울은 도시 특성상 **차선형(도로 우측 표시 방식)** 자전거도가 대부분
- 차선형은 별도 노선으로 등록하지 않고 도로 속성으로만 관리 → 전국 CSV에 미포함
- 서울시는 자체 GPX/GeoJSON 형태로 관리 (서울 열린데이터광장)

**타당성 결론**:

- 전국 CSV의 서울 380행 + 경기 4,939행 = 5,319행은 **공식 전용도로** 기준의 완전한 데이터
- 서울 별도 CSV는 **차선형 포함 확장 데이터** → 두 소스는 목적이 다름
- 모델 학습 시: 전국 표준데이터 사용, 서울 별도 데이터는 좌표 보완용으로 병합 가능
- `노선명 + 시군구명`으로 매칭하면 중복 없이 병합 가능

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
- [x] x/y 좌표 → `EPSG:4326` 위경도 확인 (x=126.95, y=37.54 → WGS84로 확인됨)
- [x] 각 자전거도로 경로 기준 반경 500m 내 편의시설 수 집계 → `facility_count` 완료 (평균 1.09)
- [ ] `설치유형` 원핫인코딩 → 경로 추천 피처로 활용

#### ⚠️ 컬럼명 주의

- `facility_clean.csv` 좌표 컬럼명: `x 좌표`, `y 좌표` (공백 포함) — `"x"`, `"y"` 로 접근 시 KeyError 발생

---

### 7.3 분석 파일 목록 및 작업 순서 *(2026-04-01 업데이트)*

| 순서 | 파일 | 작업 내용 | 산출물 | 상태 |
| ---- | ---- | --------- | ------ | ---- |
| 1 | `ml.ipynb` | TAAS 사고 데이터 위경도 정제 | `taas_clean.csv` | ⏳ |
| 2 | `ml.ipynb` (step2) | 자전거도로 서울+경기 필터 + 피처 생성 | `road_clean.csv` | ✅ **완료** (5,319행, 10컬럼, 332KB) |
| 3 | `ml.ipynb` (step1) | 편의시설 6개 컬럼 추출 | `facility_clean.csv` | ✅ **완료** (3,368행, 6컬럼) |
| 4 | `ml.ipynb` (step3) | TourAPI 호출 → 서울+경기 관광지 좌표 수집 | `tour_poi.csv` | ✅ **완료** (2,529건, 335KB) |
| 5 | `ml.ipynb` (step4) | 도로 + 관광지/편의시설 Spatial Join | `road_features.csv` | ✅ **완료** (1,647행, 14컬럼, 171KB) |
| 6 | `ml.ipynb` | 도로 + TAAS Spatial Join (100m 반경) | `route_safety_features.csv` | ⏳ **다음** |
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
| 피처 | 중요도 |
| ---- | ------ |
| 기점위도 | 0.5028 |
| 기점경도 | 0.3903 |
| 자전거도로너비(m) | 0.1068 |

> 위경도가 압도적 → 분류 모델에서는 위경도 제외하고 도메인 피처만 사용 예정

---

## 8. 미결 사항 (Action Items)

- [x] **[P0]** safety_grade 타겟 생성 (3분위 구간화) — q33=0.4661, q66=0.4787
- [x] **[P0]** RandomForestClassifier 학습 + safety_classifier.pkl 저장 — F1=0.9864
- [x] **[P0]** RandomForestRegressor 학습 + safety_regressor.pkl 저장 — R²=0.9539
- [x] **[P0]** MinMaxScaler 학습 + tourism_scaler.pkl 저장
- [x] **[P0]** road_scored.csv 저장 (1,647행, 19컬럼)
- [x] **[P1]** streamlit_kride.py 작성 및 로컬 실행 확인
- [ ] **[P1]** Streamlit Cloud 배포 (공개 URL 확보)
- [ ] **[P1]** model_comparison.md 작성 (Day 5)
- [ ] [P2] TAAS 위경도 컬럼명 통일 및 Spatial Join (100m 반경 사고 집계) — 2주차 이월
- [x] 편의시설 x/y 좌표계 확인 → WGS84(EPSG:4326) 확인됨
- [x] 자전거도로 필터링 기준 확정 → 서울특별시 + 경기도 (5,319행)
- [x] road_features.csv Spatial Join 완료 (1,647행, 14컬럼)

---

## 9. 딥러닝(YOLOv8) 서빙 연동 리서치 *(2026-04-08 신규)*

### 9.1 학습 데이터 추천 — 우선순위별

자전거 주행 환경에서 보행자·장애물·차량 탐지를 위한 데이터셋을 우선순위 순으로 정리.

#### 🥇 1순위: AI Hub — 자전거 주행 영상

| 항목 | 내용 |
| ---- | ---- |
| **출처** | 한국지능정보사회진흥원(NIA) AI Hub |
| **URL** | <https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=189> |
| **데이터 규모** | 영상 약 50시간, 객체 어노테이션 포함 |
| **라벨 형식** | JSON (COCO 유사) → YOLO txt 변환 필요 |
| **수집 대상** | 보행자, 이륜차, 차량, 장애물 (한국 도로 환경) |
| **비용** | 무료 (회원가입 후 다운로드) |
| **장점** | 한국 자전거도로 실환경 데이터, 프로젝트 도메인 최적 |
| **단점** | 로그인 및 신청 절차 필요, 용량 클 수 있음 |

**라벨 변환 방법:**

```python
# AI Hub JSON → YOLO txt 변환 예시
import json, os

def convert_aihub_to_yolo(json_path, img_w, img_h, class_map):
    with open(json_path) as f:
        data = json.load(f)
    lines = []
    for ann in data["annotations"]:
        cls = class_map.get(ann["category_id"], -1)
        if cls == -1:
            continue
        x, y, w, h = ann["bbox"]
        # YOLO 형식: 중심점 + 정규화
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines)

CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
# 1=pedestrian, 2=bicycle, 3=vehicle, 4=obstacle
```

---

#### 🥈 2순위: Roboflow Universe — bicycle-safety 공개 데이터셋

| 항목 | 내용 |
| ---- | ---- |
| **출처** | Roboflow Universe (커뮤니티 공개) |
| **URL** | <https://universe.roboflow.com/search?q=bicycle+obstacle> |
| **추천 데이터셋** | `bicycle-safety`, `cyclist-detection`, `road-hazard-detection` |
| **라벨 형식** | YOLO 형식 직접 export (변환 불필요) |
| **비용** | 무료 플랜으로 다운로드 가능 |
| **장점** | 즉시 사용 가능, YOLO 형식 지원, 온라인 augmentation |
| **단점** | 해외(서양) 도로 환경 중심, 한국 환경과 도메인 차이 존재 |

**다운로드 예시 (Roboflow API):**

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("bicycle-safety")
dataset = project.version(1).download("yolov8")
# → data/ 폴더에 images/, labels/, data.yaml 자동 생성
```

---

#### 🥉 3순위: Open Images v7 — 자전거 관련 클래스

| 항목 | 내용 |
| ---- | ---- |
| **출처** | Google Open Images Dataset v7 |
| **URL** | <https://storage.googleapis.com/openimages/web/index.html> |
| **관련 클래스** | `Bicycle`, `Person`, `Car`, `Motorcycle`, `Traffic cone` |
| **라벨 형식** | CSV 바운딩박스 → YOLO 변환 필요 |
| **비용** | 무료 |
| **장점** | 대규모(9M 이미지), 다양한 환경, 고품질 어노테이션 |
| **단점** | 클래스 선택 다운로드 도구 필요, 용량 매우 큼 |

**선택적 다운로드 (FiftyOne 사용):**

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Bicycle", "Person", "Car", "Traffic cone"],
    max_samples=3000,
)
# YOLO 형식으로 export
dataset.export(
    export_dir="./open_images_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
)
```

---

#### 보조 데이터: CCTV 영상 (공공안전데이터)

| 항목 | 내용 |
| ---- | ---- |
| **출처** | 공공데이터포털 교통안전 CCTV 영상 |
| **URL** | <https://www.data.go.kr> (검색: 자전거도로 CCTV) |
| **활용 방법** | 비라벨 영상 → YOLOv8 pseudo-labeling → 학습 데이터 증강 |
| **비고** | 직접 어노테이션이 없으므로 보조 활용 권장 |

---

### 9.2 데이터셋 비교 요약

| 데이터셋 | 한국 도메인 | 즉시 사용 | 규모 | 라벨 품질 | 추천 용도 |
| -------- | :---------: | :-------: | ---- | :-------: | --------- |
| AI Hub 자전거 주행 영상 | ✅ 최적 | ⚠️ 신청 필요 | 중 | ⭐⭐⭐⭐⭐ | 본 프로젝트 메인 |
| Roboflow bicycle-safety | ❌ 해외 중심 | ✅ 즉시 | 소~중 | ⭐⭐⭐⭐ | AI Hub 다운로드 전 빠른 프로토타입 |
| Open Images v7 | ❌ 글로벌 | ⚠️ 도구 필요 | 대 | ⭐⭐⭐⭐⭐ | 데이터 보강(augmentation) |
| CCTV 공공데이터 | ✅ 한국 | ⚠️ 어노테이션 필요 | 중 | ⭐⭐ | pseudo-labeling |

> **권장 전략**: AI Hub 신청 → 승인 대기 중 Roboflow로 프로토타입 학습 → AI Hub 데이터로 파인튜닝 재학습

---

### 9.3 FastAPI 서빙 아키텍처

```text
[클라이언트 / Spring Boot]
         │  POST /predict (multipart/form-data: image)
         ▼
[FastAPI ml-server :8001]
  ├── 이미지 디코딩 (PIL / OpenCV)
  ├── YOLOv8 추론 (best.pt)
  │     └── detections: [{class, conf, bbox}]
  ├── danger_score 계산
  │     = Σ(confidence × DANGER_WEIGHTS[class]) / n
  └── JSON 응답 반환
         │
         ▼
[Spring Boot] → danger_score → 경로 안전등급 업데이트
```

**엔드포인트 설계:**

| 메서드 | 경로 | 설명 | 응답 |
| ------ | ---- | ---- | ---- |
| `POST` | `/predict` | 이미지 → 객체 탐지 결과 | `{danger_score, detections, time_ms}` |
| `GET`  | `/health`  | 서버 상태 확인 | `{status: "ok"}` |
| `GET`  | `/model-info` | 모델 메타데이터 | `{model, classes, trained_at}` |

**danger_score 위험 클래스 가중치:**

| 클래스 | 가중치 | 근거 |
| ------ | ------ | ---- |
| `vehicle` | 0.9 | 차량 충돌 시 인명 피해 가장 큼 |
| `pedestrian` | 0.8 | 자전거-보행자 충돌 빈발 |
| `obstacle` | 0.7 | 공사 구역, 라바콘 등 낙차 위험 |
| `bicycle` | 0.4 | 동류 교통수단, 상대적 위험 낮음 |

---

### 9.4 Docker Compose 통합 아키텍처

```yaml
# docker-compose.yml 최종 구성
services:
  db:
    image: postgis/postgis:15-3.3
    ports: ["5432:5432"]

  backend:
    build: ./kride-project
    ports: ["8080:8080"]
    depends_on: [db]
    environment:
      ML_SERVER_URL: "http://ml-server:8001"

  ml-server:
    build: ./kride-project/ml-server
    ports: ["8001:8001"]
    volumes:
      - ./kride-project/ml-server/runs:/app/runs
    # GPU 사용 시 추가:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

**서비스 간 통신 흐름:**

```text
사용자 앱 → Spring Boot(:8080) → ML서버(:8001) → YOLOv8 추론
               ↕                        ↕
           PostgreSQL(:5432)    runs/weights/best.pt
```

---

### 9.5 성능 벤치마크 기준

| 지표 | 목표 | 측정 방법 |
| ---- | ---- | --------- |
| mAP50 | > 0.50 | `yolo val` 명령 |
| mAP50-95 | > 0.30 | 파인튜닝 후 검증 |
| `/predict` 응답시간 | < 500ms (CPU) | `timeit` 또는 Swagger |
| 도커 이미지 크기 | < 3GB | `docker images` |
| 메모리 사용량 | < 4GB | `docker stats` |

---

### 9.6 미결 사항 (딥러닝 서빙)

- [ ] **[P0]** AI Hub 자전거 주행 영상 신청 및 승인 — 승인 후 즉시 다운로드
- [ ] **[P0]** Roboflow 계정 생성 + bicycle 데이터셋 확보 (프로토타입용)
- [ ] **[P0]** YOLOv8n 파인튜닝 (mAP50 > 0.50 목표)
- [ ] **[P0]** FastAPI `/predict` 구현 및 로컬 테스트
- [ ] **[P1]** Docker Compose에 ml-server 서비스 추가
- [ ] **[P1]** Spring Boot ↔ ml-server 연동 (RestTemplate/WebClient)
- [ ] **[P2]** GPU 미사용 환경 최적화 (배치 크기, imgsz 조정)
- [ ] **[P2]** 모델 경량화 검토 (ONNX export 또는 TorchScript)
