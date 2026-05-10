# 📜 K-Ride 2.0 프로젝트 문서화 산출물 리스트 (Artifacts List)

본 문서는 K-Ride 2.0 프로젝트에서 생성된 모든 성능 평가 지표, 시각화 차트, 데이터 테이블 및 모델 결과물을 카테고리별로 정리한 리스트입니다. 모든 파일은 `report/` 폴더 내에 위치합니다.

---

## 1. ⚠️ 안전 지수 모델 (Safety Regressor & Classifier)
자전거 주행 안전성을 평가하고 사고 위험을 예측하는 모델 관련 산출물입니다.

| 유형 | 산출물 명칭 | 파일 경로 | 주요 내용 |
| :--- | :--- | :--- | :--- |
| **시각화** | 특성 중요도 (Regressor) | `figures/safety_regressor_feature_importance.png` | 안전 점수에 영향을 주는 주요 변수 분석 |
| **시각화** | 예측 vs 실제값 분포 | `figures/safety_regressor_pred_vs_actual.png` | 회귀 모델의 예측 정확도 산점도 |
| **시각화** | 혼동 행렬 (Classifier) | `figures/safety_classifier_confusion_matrix.png` | 위험 등급 분류 모델의 성능 검증 |
| **시각화** | 특성 중요도 (Classifier) | `figures/safety_classifier_feature_importance.png` | 위험 분류 시 주요 결정 요인 |
| **차트** | 안전 점수 히스토그램 | `charts/safety_hist.png` | 전국 시군구별 안전 점수 분포 현황 |

---

## 2. 🌤️ 날씨 예측 모델 (Weather LSTM)
과거 기상 데이터를 바탕으로 여행지의 날씨를 예측하고 안전 페널티를 계산하는 모델입니다.

| 유형 | 산출물 명칭 | 파일 경로 | 주요 내용 |
| :--- | :--- | :--- | :--- |
| **시각화** | 클래스별 데이터 분포 | `figures/weather_lstm_class_distribution.png` | 학습 데이터의 날씨 상태별 균형 확인 |
| **시각화** | 분류 성능 지표 | `figures/weather_lstm_class_metrics.png` | Precision, Recall, F1-Score 등 |
| **시각화** | 예측 혼동 행렬 | `figures/weather_lstm_confusion_matrix.png` | 날씨 예측 모델의 오답 패턴 분석 |
| **시각화** | 날씨별 안전 페널티 | `charts/05_safety_penalty.png` | 기상 조건에 따른 안전 가중치 보정값 |
| **리포트** | 날씨 모델 상세 보고서 | `report_step1_weather_lstm.md` | LSTM 아키텍처 및 성능 분석 결과 |

---

## 3. 💵 소비 예측 모델 (Consumption Prediction v3)
국민여행조사 데이터를 기반으로 여행 코스별 예상 소비 금액을 예측하는 모델입니다.

| 유형 | 산출물 명칭 | 파일 경로 | 주요 내용 |
| :--- | :--- | :--- | :--- |
| **시각화** | v3 특성 중요도 | `figures/consume_v3_feature_importance.png` | 전국 모델의 주요 소비 결정 요인 |
| **시각화** | 지역별 MAE 분석 | `figures/consume_v3_sido_mae.png` | 17개 시도별 모델 예측 오차 비교 |
| **시각화** | 여행 유형별 소비 | `figures/consume_v3_trip_type_dist.png` | 개별/패키지 등 유형별 지출 규모 |
| **시각화** | v2 vs v3 성능 비교 | `charts/21_consume_v2_vs_v3.png` | 수도권 한정 모델 대비 전국 모델 개선도 |
| **테이블** | 모델 성능 지표 (R2/MAE) | `tables/consume_v3_metrics.csv` | 전국 모델의 핵심 성능 수치 |
| **테이블** | 시도별 상세 오차 | `tables/consume_v3_sido_performance.csv` | 지역별 정밀 성능 평가 테이블 |
| **리포트** | 소비 모델 통합 보고서 | `consume_v3_detail_report.md` | 모델 신뢰도 및 향후 고도화 계획 |

---

## 4. 🎤 K-Culture & GraphRAG
K-Drama 촬영지 데이터와 이를 연결하는 지식 그래프 네트워크 산출물입니다.

| 유형 | 산출물 명칭 | 파일 경로 | 주요 내용 |
| :--- | :--- | :--- | :--- |
| **시각화** | 드라마별 촬영지 TOP 10 | `figures/kculture_top_dramas.png` | 인기 드라마별 촬영지 보유 현황 |
| **시각화** | 지역별 촬영지 분포 | `figures/kculture_region_dist.png` | 시도별 K-Culture 성지 분포도 |
| **시각화** | 커뮤니티 노드 분포 | `figures/kculture_community_pie.png` | GraphRAG로 분류된 엔티티 그룹 비율 |
| **시각화** | 카테고리 가중치 효과 | `figures/poi_rec_category_boost.png` | 취향 반영 시 추천 결과의 변화 시각화 |
| **테이블** | 지역별 촬영지 통계 | `tables/kculture_region_stats.csv` | K-Culture POI 지역별 집계 데이터 |
| **데이터** | 그래프 노드 샘플 | `tables/graph_nodes_sample.csv` | 지식 그래프를 구성하는 엔티티 예시 |
| **모델** | GraphRAG 구조 (JSON) | `../models/kride_graph.json` | LLM 연동을 위한 그래프 데이터셋 |

---

> **참고**: 모든 시각화 자료는 보고서 작성 시 바로 삽입하여 사용할 수 있도록 고해상도 PNG 파일로 저장되어 있습니다.
