# K-Ride ML 구현 계획

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
- [ ] TAAS 데이터 위경도 정제 및 Spatial Join (100m 반경 사고 집계)
  - 결과 저장: `data/raw_ml/route_safety_features.csv`

#### Day 3 (4/2): 피처 엔지니어링 + 모델 학습

- [ ] `kride-project/ml.ipynb` — TAAS 병합 후 최종 피처셋 구성
  - **⚠️ 현재 모델 성능 낮음** (위경도만 있을 때): LinearRegression R²=0.0096, RandomForest R²=0.1890
  - TAAS 사고 데이터 병합 후 재학습 필요
  - 피처 중요도 현황: 기점위도(0.50) > 기점경도(0.39) > 도로폭(0.11)
  - 목표 모델: RandomForest (현재 최고), XGBoost 추가 비교
  - 평가 지표: RMSE, R², F1-score
- [ ] 최적 모델 저장: `kride-project/ml-server/models/safety_model.pkl`

#### Day 4 (4/3): 추천 엔진 구현

- [ ] 코사인 유사도 기반 경로 추천 함수 작성
  - 입력: 사용자 선호도 벡터 (안전 우선 / 거리 우선 / 관광 우선)
  - 출력: Top-5 추천 경로 리스트
- [ ] folium으로 추천 경로 지도 시각화
- [ ] 결과 리포트 저장: `kride-project/report/recommendation_eval.md`

#### Day 5 (4/4): 검토 및 문서화

- [ ] 전체 파이프라인 end-to-end 실행 확인
- [ ] `kride-project/report/` 에 분석 결과 정리
- [ ] 다음 주 YOLO 학습 준비 (데이터셋 경로 정리)

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
│       ├── route_safety_features.csv   ← ⏳ TAAS Spatial Join 결과 (미완)
│       └── recommendation_input.csv    ← ⏳ 추천 엔진 입력 (미완)
├── ml-server/
│   └── models/
│       ├── safety_model.pkl            ← ⏳ 안전지수 예측 모델 (미완)
│       └── yolov8_kride.pt             ← ⏳ YOLO 파인튜닝 모델 (2주차)
└── report/
    ├── sido_distribution.png           ← ✅ 완료 (시도별 분포 그래프)
    ├── model_comparison.md             ← ⏳ 모델 비교 결과 (미완)
    └── recommendation_eval.md         ← ⏳ 추천 엔진 평가 (미완)
```

---

## 우선순위 매트릭스

| 우선순위 | 작업 | 완료 기준 |
| -------- | ---- | --------- |
| P0 | TAAS Spatial Join | CSV 파일 생성 + null 0% |
| P0 | 안전지수 예측 모델 | RMSE < 10, R² > 0.7 |
| P1 | 경로 추천 엔진 | Top-5 결과 출력 확인 |
| P1 | folium 시각화 | HTML 지도 파일 생성 |
| P2 | YOLOv8 학습 | mAP@0.5 > 0.6 |
| P2 | FastAPI 서빙 | `/predict` 200 응답 확인 |

---

## 의존성 및 제약사항

- PostGIS Docker 컨테이너 실행 필요 (`docker-compose up -d db`)
- AI Hub 데이터 다운로드는 회원가입 후 수동 진행 필요
- GPU 없을 경우 YOLOv8n (nano) 모델 사용 (CPU 학습 가능)

---

## 알려진 이슈 및 트러블슈팅

| 날짜 | 이슈 | 원인 | 해결 |
| ---- | ---- | ---- | ---- |
| 4/1 | `KeyError: 'x'` | `facility_clean.csv` 컬럼명이 `x 좌표`, `y 좌표` (공백 포함) | `["x", "y"]` → `["x 좌표", "y 좌표"]` 로 수정 |
| 4/1 | `NameError: 'pts' is not defined` | `count_poi_in_buffer` 함수가 두 셀로 분리되어 함수 내부 지역변수가 참조 불가 | 함수 전체를 하나의 셀에 합치기 |
| 4/1 | Spatial Join 속도 느림 | 4회 반복 버퍼 생성 (LineString 1,647개 × 버퍼 연산 × 4회) | 1km/500m 버퍼를 미리 한 번만 생성 후 재사용 권장 |
