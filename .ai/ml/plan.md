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

**목표:** 객체 탐지 모델 학습 + FastAPI 서빙 프로토타입 + Docker 통합

#### Day 1 (4/7): 데이터 준비 및 환경 구성

- [ ] **데이터 확보** (아래 우선순위 순)
  - 1순위: AI Hub `자전거 주행 영상` 다운로드 (로그인 필요, 무료)
    - URL: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=189
    - 라벨 포맷: JSON → YOLO txt 변환 필요
  - 2순위: Roboflow `bicycle-safety` 공개 데이터셋 (즉시 다운로드 가능)
    - URL: https://universe.roboflow.com/search?q=bicycle+obstacle
    - 라벨 포맷: YOLO 형식 직접 export 가능
  - 3순위: Open Images v7 subset (`bicycle`, `person`, `vehicle` 클래스)
    - 도구: `fiftyone` 라이브러리 또는 OIDv4_ToolKit
- [ ] YOLOv8 환경 설치
  ```bash
  pip install ultralytics
  yolo check  # GPU/CPU 확인
  ```
- [ ] 데이터 디렉토리 구성
  ```text
  kride-project/ml-server/
  ├── data/
  │   ├── images/train/
  │   ├── images/val/
  │   ├── labels/train/
  │   └── labels/val/
  └── dataset.yaml
  ```

#### Day 2 (4/8): YOLO 라벨 변환 + 파인튜닝

- [ ] AI Hub JSON → YOLO txt 라벨 변환 스크립트 작성
  ```python
  # convert_labels.py
  # {"bbox": [x,y,w,h]} → "class cx cy nw nh" (정규화)
  ```
- [ ] `dataset.yaml` 작성
  ```yaml
  path: kride-project/ml-server/data
  train: images/train
  val:   images/val
  nc: 4
  names: [pedestrian, bicycle, vehicle, obstacle]
  ```
- [ ] YOLOv8n 파인튜닝 실행
  ```bash
  yolo detect train \
    model=yolov8n.pt \
    data=dataset.yaml \
    epochs=30 imgsz=640 batch=16 \
    project=kride-project/ml-server/runs
  ```
- [ ] 학습 결과 확인: `runs/detect/train/results.csv`, `mAP50` 목표 > 0.5

#### Day 3 (4/9): FastAPI 서빙 구현

- [ ] `ml-server/main.py` 작성
  ```python
  # POST /predict — 이미지 업로드 → 탐지 결과 JSON 반환
  # GET  /health  — 서버 상태 확인
  # GET  /model-info — 모델 메타 (클래스명, 학습 날짜)
  ```
- [ ] 응답 스키마 (`schemas.py`)
  ```python
  class Detection(BaseModel):
      class_name: str
      confidence: float
      bbox: list[float]  # [x1,y1,x2,y2]
      danger_weight: float

  class PredictResponse(BaseModel):
      danger_score: float  # 탐지 결과 기반 위험도
      detections: list[Detection]
      processing_time_ms: float
  ```
- [ ] `danger_score` 계산 로직
  ```python
  DANGER_WEIGHTS = {"pedestrian": 0.8, "vehicle": 0.9, "obstacle": 0.7, "bicycle": 0.4}
  danger_score = sum(det.confidence * DANGER_WEIGHTS[det.class_name] for det in detections) / max(len(detections), 1)
  ```
- [ ] 로컬 테스트: `uvicorn main:app --reload --port 8001`
- [ ] Swagger UI (`/docs`) 에서 `/predict` 동작 확인

#### Day 4 (4/10): Docker Compose 통합

- [ ] `ml-server/Dockerfile` 작성
  ```dockerfile
  FROM python:3.11-slim
  RUN pip install ultralytics fastapi uvicorn python-multipart
  COPY . /app
  WORKDIR /app
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
  ```
- [ ] `docker-compose.yml` 업데이트
  ```yaml
  services:
    db:       # PostgreSQL + PostGIS
    backend:  # Spring Boot (기존)
    ml-server:  # FastAPI YOLOv8 서버 (신규)
      build: ./kride-project/ml-server
      ports: ["8001:8001"]
      volumes:
        - ./kride-project/ml-server/runs:/app/runs
  ```
- [ ] `docker compose up --build` 실행 후 통합 테스트

#### Day 5 (4/11): Spring Boot ↔ ML 서버 연동 + 문서화

- [ ] Spring Boot → ML 서버 호출 (`RestTemplate` 또는 `WebClient`)
  ```java
  // POST http://ml-server:8001/predict
  // MultipartFile → ByteArrayResource 변환 후 전송
  ```
- [ ] ML 서버 응답의 `danger_score` → 경로 안전등급 반영
- [ ] `kride-project/report/dl_serving_architecture.md` 작성
- [ ] plan.md / research.md 업데이트

**모델 저장 위치:**
```text
kride-project/ml-server/
├── main.py                    ← FastAPI 앱
├── schemas.py                 ← 요청/응답 스키마
├── Dockerfile
├── requirements.txt
├── data/                      ← 학습 데이터
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── dataset.yaml
└── runs/detect/train/
    └── weights/
        ├── best.pt            ← 최종 사용 모델
        └── last.pt
```

**평가 목표:**

| 지표 | 목표 | 비고 |
|------|------|------|
| mAP50 | > 0.50 | YOLO 파인튜닝 기준 |
| `/predict` 응답시간 | < 500ms | CPU 기준 |
| Docker Compose 통합 | ✅ | ml-server + db + backend |

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
