# K-Ride ML 구현 계획

> 관련 문서: [Streamlit 포트폴리오 Next.js 배포 계획](../streamlit-portfolio/plan.md) | [배포 가이드](../streamlit-portfolio/guide.md)

## 목표

이번 주(3/31 ~ 4/4) 안에 ML 데이터 수집 및 분석 완료.
`project_human_plan.md` 기반으로 안전 경로 추천 모델의 1차 프로토타입 구현.

---

## 주차별 로드맵

### 1주차: ML 데이터 완성 (현재)

**목표:** 학습 가능한 정제 데이터셋 완성 + 기본 모델 구현

#### Day 1–2 (3/31 ~ 4/1): 데이터 수집 및 Spatial Join ✅ 완료

- [x] `road_clean.csv` 생성 완료 — 서울+경기 5,319행, 10컬럼
- [x] `facility_clean.csv` 생성 완료 — 3,368행, 6컬럼
- [x] `tour_poi.csv` 생성 완료 — 2,529건 (서울+경기, contentTypeId 12/14/28)
- [x] **Spatial Join 완료** (`road_features.csv` 생성)
  - 관광지 1km 반경: tourist_count 평균 1.63, cultural_count 0.74, leisure_count 0.10
  - 편의시설 500m 반경: facility_count 평균 1.09
  - 최종 shape: (1,647행, 14컬럼) — 위경도 결측으로 5,319 → 1,647행 축소
  - 파일 크기: 171,398 bytes
- [x] 회귀 모델 성능 확인 → LinearRegression R²=0.0096, RandomForest R²=0.1890
  - **회귀 한계 확인: 타겟(safety_index)이 단순 공식 기반 → 회귀 적합도 낮음**
  - **전략 전환: 분류(Classification) + 점수 기반 추천으로 방향 변경**

#### Day 3 (4/2): 모델 학습 및 pkl 생성 ✅ 완료

**모델 1: 안전등급 분류 모델** → `safety_classifier.pkl` + `safety_scaler.pkl`

- [x] `road_features.csv` 로드 후 타겟 생성
  - `safety_index_v2` 3분위로 구간화 → `danger_level` (0=안전, 1=보통, 2=위험)
  - q33=0.4661, q66=0.4787
- [x] 피처: `width_m`, `length_km`, `district_danger`, `road_attr_score`
- [x] MinMaxScaler 적용 → `safety_scaler.pkl` 저장
- [x] RandomForestClassifier 학습 → `safety_classifier.pkl` 저장
  - R²=0.9539 (회귀), F1-macro=0.9864 (분류)
- [x] RandomForestRegressor → `safety_regressor.pkl` 저장
- [x] 메타 정보 → `safety_meta.pkl` 저장 (피처명, 등급 매핑, q33/q66)

**모델 2: 관광 경로 추천 스코어** → `tourism_scaler.pkl`

- [x] MinMaxScaler로 `tourism_raw` 정규화 → `tourism_scaler.pkl` 저장
- [x] `tourism_score`, `safety_score`, `final_score` 컬럼 생성
- [x] `road_scored.csv` 저장 (1,647행, 19컬럼)
  - safety_score 평균: 0.506 / tourism_score 평균: 0.088 / final_score 평균: 0.338

**실제 저장 위치:**
```
kride-project/models/
├── safety_classifier.pkl   ✅ RandomForest 분류 모델
├── safety_regressor.pkl    ✅ RandomForest 회귀 모델
├── safety_scaler.pkl       ✅ MinMaxScaler (분류 입력 전처리)
├── safety_meta.pkl         ✅ 피처명·등급 매핑·임계값 메타
└── tourism_scaler.pkl      ✅ MinMaxScaler (관광 점수 정규화)
```

#### Day 4 (4/3): Streamlit 앱 통합 ✅ 완료

- [x] `kride-project/streamlit_kride.py` 작성
  - **사이드바**: 사용자 모드 선택 (🛡️ 안전 우선 / ⚖️ 균형 / 🗺️ 관광 우선)
  - **탭 1 - 안전등급 예측**: 슬라이더 입력 → `safety_classifier` 예측 → 🟢🟡🔴 등급 + 신뢰도 확률 바
  - **탭 2 - 경로 추천 Top-10**: 모드별 가중치 적용 → 상위 10개 경로 테이블 + 점수 분포 차트
  - **탭 3 - 데이터 탐색**: 히스토그램, 산점도(추천점수 컬러맵), 기술통계
- [ ] Streamlit Cloud 배포

**모드별 가중치:**

| 모드 | 안전(safety_score) | 관광(tourism_score) |
|------|-------------------|---------------------|
| 안전 우선 | 70% | 30% |
| 균형 | 50% | 50% |
| 관광 우선 | 30% | 70% |

**실행 명령:**
```bash
streamlit run kride-project/streamlit_kride.py
```

#### Day 5 (4/4): 검토 및 문서화

- [ ] 전체 파이프라인 end-to-end 실행 확인
- [ ] `kride-project/report/model_comparison.md` 작성
- [ ] Streamlit Cloud 배포 URL 확정
- [ ] Next.js 포트폴리오에 K-Ride 앱 카드 추가

---

### 2주차: 딥러닝 (YOLOv8) + 서빙 연동

**목표:** 객체 탐지 모델 학습 + FastAPI 서빙 프로토타입

- [ ] AI Hub 자전거 주행 영상 데이터 다운로드 및 라벨 변환
- [ ] YOLOv8n 파인튜닝 (보행자, 장애물, 이륜차)
- [ ] FastAPI `/predict` 엔드포인트 구현
- [ ] Docker Compose로 DB + ML 서버 통합 실행

---

## 폴더 구조 (신규 생성 파일 포함)

```text
kride-project/
├── data/
│   └── raw_ml/
│       ├── road_clean.csv              ← ✅ 완료 (5,319행, 10컬럼, 332KB)
│       ├── facility_clean.csv          ← ✅ 완료 (3,368행, 6컬럼)
│       ├── tour_poi.csv                ← ✅ 완료 (2,529건, 335KB)
│       ├── road_features.csv           ← ✅ 완료 (1,647행, 14컬럼, 171KB)
│       └── road_scored.csv             ← ✅ 완료 (1,647행, 19컬럼) safety/tourism/final_score 포함
├── models/
│   ├── safety_classifier.pkl           ← ✅ 완료 RandomForest 분류 (F1=0.9864)
│   ├── safety_regressor.pkl            ← ✅ 완료 RandomForest 회귀 (R²=0.9539)
│   ├── safety_scaler.pkl               ← ✅ 완료 MinMaxScaler
│   ├── safety_meta.pkl                 ← ✅ 완료 피처·등급 메타
│   └── tourism_scaler.pkl              ← ✅ 완료 관광 점수 스케일러
├── report/
│   ├── sido_distribution.png           ← ✅ 완료
│   └── model_comparison.md             ← ⏳ Day 5
└── streamlit_kride.py                  ← ✅ 완료 (Day 4)
```

---

## 우선순위 매트릭스

| 우선순위 | 작업 | 완료 기준 |
| -------- | ---- | --------- |
| P0 | safety_grade 타겟 생성 | ✅ 완료 (q33/q66 3분위) |
| P0 | RandomForest 분류 학습 | ✅ 완료 F1=0.9864 |
| P0 | safety_classifier.pkl 저장 | ✅ 완료 |
| P0 | tourist_scorer.pkl 저장 | ✅ 완료 (tourism_scaler.pkl) |
| P1 | streamlit_kride.py 작성 | ✅ 완료 (로컬 실행 가능) |
| P1 | Streamlit Cloud 배포 | ⏳ Day 5 |
| P2 | TAAS Spatial Join | ⏳ 2주차 이월 |
| P2 | YOLOv8 학습 | ⏳ 2주차 이월 |

---

## 의존성 및 제약사항

- TAAS Spatial Join은 P2 — 시간 부족 시 2주차로 이월
- `road_features.csv` 위경도 결측(3,624행) 있으나 분류 모델에는 위경도 미사용이므로 무관
- GPU 없을 경우 YOLOv8n (nano) 모델 사용 (CPU 학습 가능)

---

## 알려진 이슈 및 트러블슈팅

| 날짜 | 이슈 | 원인 | 해결 |
| ---- | ---- | ---- | ---- |
| 4/1 | `KeyError: 'x'` | `facility_clean.csv` 컬럼명이 `x 좌표`, `y 좌표` (공백 포함) | `["x", "y"]` → `["x 좌표", "y 좌표"]` 로 수정 |
| 4/1 | `NameError: 'pts' is not defined` | `count_poi_in_buffer` 함수가 두 셀로 분리되어 함수 내부 지역변수가 참조 불가 | 함수 전체를 하나의 셀에 합치기 |
| 4/1 | Spatial Join 속도 느림 | 4회 반복 버퍼 생성 | 1km/500m 버퍼를 미리 한 번만 생성 후 재사용 |
| 4/1 | 회귀 R² 낮음 (0.19) | safety_index가 단순 공식 기반 타겟 → 회귀 부적합 | **분류로 전략 전환 (3등급)** |
