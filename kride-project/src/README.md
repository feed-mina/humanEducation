# K-Ride 소스코드 구조

> 이 `src/` 폴더는 프로젝트에 실제로 사용되는 코드만 모은 정리 폴더입니다.
> 루트에 있는 원본 파일과 동일하며, 역할별로 분류되어 있습니다.

---

## 실행 순서 (파이프라인)

```
[1단계: 데이터 수집]
  data_collect/fetch_weather_data.py      ← ASOS 기상 데이터 수집
  data_collect/step3_tour_collect_v2.py   ← 전국 관광 POI 수집 (TourAPI)
  data_collect/step3_food_collect.py      ← 카카오 맛집/카페 POI 수집
  data_collect/step4_naver_trend.py       ← 네이버 DataLab 검색 트렌드

[2단계: 전처리]
  preprocessing/preprocess_road.py        ← 자전거도로 전처리 → road_clean_v2.csv
  preprocessing/step1_facility_clean.py   ← 편의시설 전처리 → facility_clean.csv
  preprocessing/step2_road_clean.py       ← 도로 정제 보조
  preprocessing/step4_spatial_join.py     ← Spatial Join → road_features.csv

[3단계: ML 모델 학습]
  ml/build_safety_model.py                ← RF 안전 예측 (R²=0.9539)
  ml/build_tourism_model.py               ← 관광 점수 (규칙 기반)
  ml/build_tourism_score_v2.py            ← 관광 점수 v2 (attraction 반영)
  ml/build_route_graph.py                 ← OSM 경로 그래프 (172,656 노드)
  ml/build_poi_recommender_v2.py          ← Co-occurrence POI 추천 (Recall@5=0.1372)

[4단계: DL 모델 학습]
  dl/build_weather_lstm.py                ← WeatherLSTM (Acc=73.28%)
  dl/build_consume_model_v2.py            ← 소비 TabNet v2 (R²=0.1277)
  dl/build_attraction_model.py            ← POI 매력도 TabNet (R²=0.0662)
  dl/build_event_ner.py                   ← 이벤트 NER (zero-shot, 데이터 대기)

[5단계: 서비스 실행]
  app/streamlit_kride.py                  ← Streamlit 앱 (Tab 1~5)
  api/fastapi_server.py                   ← FastAPI ML 서버 (배포 미완)

[보고서]
  report/generate_report.py               ← PDF 보고서 생성
  report/report_step1_weather_lstm.py     ← WeatherLSTM 보고서
  report/report_step2_poi_tabnet.py       ← POI TabNet 보고서
  report/inject_slides.py                 ← 슬라이드 주입
```

---

## 폴더 구조

```
src/
├── preprocessing/          # 데이터 전처리
│   ├── preprocess_road.py
│   ├── step1_facility_clean.py
│   ├── step2_road_clean.py
│   └── step4_spatial_join.py
│
├── data_collect/           # 외부 데이터 수집 (API)
│   ├── fetch_weather_data.py    ← ASOS 기상청 API
│   ├── weather_kma.py           ← KMA 단기예보 API (실시간)
│   ├── step3_tour_collect_v2.py ← 한국관광공사 TourAPI 전국
│   ├── step3_food_collect.py    ← 카카오 로컬 REST API
│   └── step4_naver_trend.py     ← 네이버 DataLab 트렌드
│
├── ml/                     # 머신러닝 모델 학습
│   ├── build_safety_model.py       ← RF 안전 예측 ✅
│   ├── build_tourism_model.py      ← 관광 점수 (규칙 기반) ✅
│   ├── build_tourism_score_v2.py   ← 관광 점수 v2 ✅
│   ├── build_route_graph.py        ← OSM 경로 그래프 ✅
│   └── build_poi_recommender_v2.py ← Co-occurrence POI 추천 ✅
│
├── dl/                     # 딥러닝 모델 학습
│   ├── build_weather_lstm.py      ← WeatherLSTM ✅ (Acc=73.28%)
│   ├── build_consume_model_v2.py  ← 소비 TabNet v2 ✅ (R²=0.1277)
│   ├── build_attraction_model.py  ← POI 매력도 TabNet ✅
│   └── build_event_ner.py         ← 이벤트 NER (데이터 대기)
│
├── api/                    # 서버
│   ├── fastapi_server.py   ← FastAPI ML 서버 (배포 미완)
│   ├── db_schema.sql        ← PostgreSQL 스키마
│   ├── docker-compose.yml   ← Docker 구성
│   └── upload_to_hf.py      ← HuggingFace Hub 업로드
│
├── app/                    # Streamlit 앱
│   ├── streamlit_kride.py  ← 메인 앱 (Tab 1~5)
│   └── .streamlit/         ← Streamlit 설정
│
├── report/                 # 보고서 생성
│   ├── generate_report.py
│   ├── report_step1_weather_lstm.py
│   ├── report_step2_poi_tabnet.py
│   └── inject_slides.py
│
└── utils/
    └── requirements.txt    ← 의존성 패키지 목록
```

---

## 실행 방법

```bash
# Streamlit 앱 실행
streamlit run src/app/streamlit_kride.py

# FastAPI 서버 실행
uvicorn src.api.fastapi_server:app --reload --port 8001

# 파이프라인 전체 실행 예시
python src/preprocessing/preprocess_road.py
python src/ml/build_safety_model.py
python src/ml/build_tourism_score_v2.py
python src/ml/build_route_graph.py
python src/dl/build_weather_lstm.py
python src/dl/build_consume_model_v2.py
```

---

## 모델 성능 현황

| 모델 | 파일 | 성능 |
|------|------|------|
| RF 안전 예측 | models/safety_regressor.pkl | R²=0.9539, F1=0.9864 |
| WeatherLSTM | models/dl/weather_lstm.pt | Acc=73.28% |
| 소비 TabNet v2 | models/consume_regressor_v2.zip | R²=0.1277 |
| POI 매력도 TabNet | models/attraction_regressor.zip | R²=0.0662 |
| Co-occurrence POI | models/poi_cooccurrence_v2.pkl | Recall@5=0.1372 |
| 경로 그래프 | models/route_graph.pkl | 172,656 노드 (재실행 필요) |
