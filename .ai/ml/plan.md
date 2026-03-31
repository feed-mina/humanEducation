# K-Ride ML 구현 계획

## 목표

이번 주(3/31 ~ 4/4) 안에 ML 데이터 수집 및 분석 완료.
`project_human_plan.md` 기반으로 안전 경로 추천 모델의 1차 프로토타입 구현.

---

## 주차별 로드맵

### 1주차: ML 데이터 완성 (현재)

**목표:** 학습 가능한 정제 데이터셋 완성 + 기본 모델 구현

#### Day 1–2 (3/31 ~ 4/1): TAAS 데이터 전처리 완료

- [ ] `0331_taas_data.ipynb` — 사고 데이터 정제 마무리
  - 결측치 처리 전략 확정 (median imputation or drop)
  - 안전지수 산출 공식 최종화: `score = 사망×50 + 중상×20 + 경상×10`
  - 위경도 컬럼 추출 및 유효성 검증 (범위: 위도 33~38, 경도 125~130)
- [ ] 자전거도로 공공데이터 + TAAS Spatial Join 구현
  - 반경 100m 이내 사고 건수를 경로별 집계
  - 결과 저장: `kride-project/data/raw_ml/route_safety_features.csv`

#### Day 3 (4/2): 피처 엔지니어링 + 모델 학습

- [ ] `kride-project/ml.ipynb` — 피처 정의 및 모델 비교
  - 입력 피처: 도로폭, 사고건수, 사고유형비율, 시간대별 위험도
  - 타겟: 안전지수 (회귀) / 위험등급 (분류: 상/중/하)
  - 모델 비교: LinearRegression, RandomForest, XGBoost
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
│       ├── route_safety_features.csv   ← Spatial Join 결과 (신규)
│       └── recommendation_input.csv    ← 추천 엔진 입력 (신규)
├── ml-server/
│   └── models/
│       ├── safety_model.pkl            ← 안전지수 예측 모델 (신규)
│       └── yolov8_kride.pt             ← YOLO 파인튜닝 모델 (2주차)
└── report/
    ├── model_comparison.md             ← 모델 비교 결과 (신규)
    └── recommendation_eval.md         ← 추천 엔진 평가 (신규)
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
