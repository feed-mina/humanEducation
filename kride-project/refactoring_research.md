# K-Ride ML/DL 향상 연구 문서

> 작성일: 2026-04-16
> 목적: 기존 연구(research.md)에서 미해결된 ML/DL 성능 문제를 분석하고,
>       전국 확대 및 SpringBoot + FastAPI + React 배포를 고려한 개선 방향을 정리한다.

---

## 1. 현재 모델 성능 종합 현황

### 1-1. 전체 모델 성능 대시보드
// [메모] 관광점수에서 관광지나 소비예측에서 여기어때, 야놀자, 카카오나 네이버 api 등을 사용할 수 있는지 여부를 확인하고 가능하다면 api연계로 db를 먼저 테이블로 저장하거나 api를 사용해서 실시간으로 데이터를 가져와서 모델을 학습시키는것도 생각하고 있습니다

 // [메모] 크롤링 (k컬쳐/아이돌 브이로그 방문지) 등은 우선 나중으로 생각합니다 
 // [메모] 또간집(https://share.google/jFIwVJYCwmqY7FXVT) 이나 망고플레이트(https://share.google/jFIwVJYCwmqY7FXVT) , 블루리본 등 맛집 리스트 추가도 생각이 있습니다. 어디에 어떻게 반영하면 좋을지 아이디어 부탁합니다 

| 모델 | 파일 | 현재 성능 | 목표 성능 | 상태 | 우선순위 |
|------|------|-----------|-----------|------|---------|
| **안전 RF 회귀** | safety_regressor.pkl | R²=0.9539 | R²≥0.97 | 양호 (개선 여지) | 3순위 |
| **안전 RF 분류** | safety_classifier.pkl | F1=0.9864 | F1≥0.99 | 양호 | 4순위 |
| **관광 점수** | road_scored.csv | 규칙 기반 (평균 0.088) | 실데이터 기반 ≥0.15 | 구조적 한계 | 2순위 |
| **소비 TabNet** | consume_regressor_v2.zip | MAE=129,653원, R²=0.1277 (v2 test) | R²≥0.25 | v2 완료 (0.0053→0.1277) — 전국 데이터 확보 후 v3 목표 | **1순위** |
| **POI 매력도 TabNet** | attraction_regressor.zip | MAE=0.6558, R²=0.0662 | R²≥0.15 | Selection bias 구조 한계 | 3순위 |
| **날씨 LSTM** | weather_lstm.pt | Test Acc=73.28% | Acc≥80% | 개선 가능 | 2순위 |
| **GRU 시퀀스** | visit_seq_gru.pt | top5=0.14 (랜덤 이하) | — | **폐기 → Co-occurrence 대체** | 완료 |
| **Co-occurrence POI 추천** | poi_cooccurrence_v2.pkl | Recall@5=0.1372 / Recall@10=0.1811 (v2 test, 베이스라인 3.7배) | Recall@5≥0.20 | v2 완료 (2026-04-16) — 전국 POI 15,851개 카테고리 연동 | 3순위 |
| **경로 그래프** | route_graph.pkl | 172,656 노드, 238,962 엣지 | 전국 확대 | 서울만 완료 | 2순위 |

---

## 2. 소비 예측 모델 (TabNet) — 긴급 개선 계획

> **v1**: R²=0.0053 (사실상 예측 불가) → **v2 실행 결과 (2026-04-16)**: MAE=129,653원, R²=0.1277 (test) / R²=0.1155 (val)  
> Step A~E 전부 적용 완료. 타겟이 활동소비 → 전체소비(4개 테이블 합산)로 변경되어 MAE 절대값은 증가했으나 설명력(R²)은 24배 향상.  
> **다음 목표**: 전국 AI Hub 데이터 확보(Step F) → R²≥0.25

### 2-1. 원인 상세 분석

| 원인 | 현황 수치 | 영향도 | 해결 난이도 |
|------|-----------|--------|------------|
| **소비 분산 극단** | min=1,000원 ~ max=15,071,450원, std=366,662원 | 매우 높음 | 낮음 (이상치 제거) |
| **타겟 범위 협소** | 활동 소비만 집계 (이동/숙박 제외) | 높음 | 중간 (테이블 머지) |
| **피처 부족** | 7개 피처만 사용 (소득, 목적, 성별 없음) | 높음 | 중간 |
| **계절 불균형** | 여름 82.8% (2,094/2,530행) | 중간 | 낮음 |
| **지역 인코딩 조악** | sgg_code → LabelEncoder (순서 없는 정수) | 중간 | 낮음 |
| **데이터 양 부족** | 2,530행 (TabNet 권장: 10,000행+) | 높음 | 높음 (전국 확대로 해결) |

### 2-2. 개선 로드맵 (우선순위 순)

#### Step A. 이상치 제거 + 로그 변환 (난이도: 낮음 / 예상 R² 개선: +0.05~0.10)

```python
# build_consume_model.py 수정
# Step 1: 이상치 제거 (상위 1% 컷)
q99 = df["consume_amt"].quantile(0.99)
q01 = df["consume_amt"].quantile(0.01)
df = df[(df["consume_amt"] >= q01) & (df["consume_amt"] <= q99)]

# Step 2: 로그 변환 (분산 안정화)
import numpy as np
df["log_consume"] = np.log1p(df["consume_amt"])
# 예측 후 역변환: np.expm1(pred)
```

#### Step B. 타겟 재정의 — 4개 소비 테이블 전체 합산 (난이도: 중간 / 예상 R² 개선: +0.10~0.20)

| 현재 타겟 | 개선 타겟 | 데이터 소스 | 비고 |
|---------|---------|-----------|------|
| 활동 소비만 | 활동 + 이동 + 숙박 + 사전 소비 합산 | tn_activity_consume + tn_mvmn_consume + tn_lodge_consume | 파일 존재 확인 필요 |

```python
def load_total_consume():
    activity = pd.read_csv("tn_activity_consume_his*.csv")
    # mvmn(이동) / lodge(숙박) 파일 있으면 머지
    # TRAVEL_ID 기준 합산
    total = activity.groupby("TRAVEL_ID")["CONSUME_AMT"].sum()
    return total
```

#### Step C. 피처 추가 — tn_traveller_master 연동 (난이도: 중간 / 예상 R² 개선: +0.10~0.15)
// [메모] 소득분위, 가구소득에서 연속형으로 말고 구간으로 나누는게 더 좋을 것 같습니다. 짠순이/보통/호캉스 등 3단계로 나눠 진행합니다 

| 추가 피처 | 출처 컬럼 | 타입 | 기대 효과 |
|---------|---------|------|---------|
| 소득 분위 | INCOME (1~8분위) | 연속형 | 소비 수준 직접 반영 (가장 중요) |
| 가구 소득 | HOUSE_INCOME | 연속형 | 소득 보완 |
| 성별 | GENDER | 범주형 | 소비 패턴 차이 |
| 연령대 | AGE_GRP | 범주형 | 세대별 소비 차이 |
| 여행 목적 | TRAVEL_PURPOSE | 범주형 | 레저/업무/가족 소비 패턴 상이 |
| 여행 스타일 | TRAVEL_STYL_1~8 (평균) | 연속형 | 활동 선호도 → 소비 연관 |

#### Step D. 지역 인코딩 고도화 — Target Encoding (난이도: 낮음 / 예상 R² 개선: +0.03~0.05)

```python
from category_encoders import TargetEncoder

te = TargetEncoder(cols=["sgg_code"])
X_train["sgg_enc"] = te.fit_transform(X_train["sgg_code"], y_train)
X_test["sgg_enc"]  = te.transform(X_test["sgg_code"])
# 시군구별 평균 소비금액을 피처로 사용 → LabelEncoding보다 의미 있는 지역 표현
```

#### Step E. 계절 불균형 해소 (난이도: 낮음 / 예상 R² 개선: +0.02~0.03)

```python
# 여름(season=2) 샘플 다운샘플링 또는 가중치 부여
summer_mask = X_train["season"] == 2
season_weights = np.where(summer_mask, 0.5, 1.5)  # 여름 절반 가중치
regressor.fit(X_train, y_train, weights=season_weights)
```

#### Step F. 전국 데이터로 샘플 확장 (난이도: 중간 / 예상 R² 개선: +0.10~0.20)

현재 수도권(2,530행) → 전국 AI Hub 데이터 추가 시 5,000~20,000행 예상.
데이터 양 증가가 TabNet에서 가장 큰 R² 개선 효과.

### 2-3. 개선 후 예상 성능
// [메모] 소득분위에서 연속형으로 말고 구간으로 나누는게 더 좋을 것 같습니다. 짠순이/보통/호캉스 등 3단계로 나눠 진행합니다 

| 단계 | 적용 개선 | 예상 R² | 예상 MAE | 비고 |
| --- | --- | --- | --- | --- |
| v1 (기준) | 활동소비 단독, 7 피처 | 0.0053 | 125,302원 | 활동 소비만 집계 |
| Step A 후 | 이상치 제거 + 로그 변환 | ~0.05 | ~60,000원 | 예상치 |
| Step B 후 | 타겟 재정의 (전체 소비) | ~0.15 | ~45,000원 | 예상치 |
| Step C 후 | 소득/목적/성별 피처 추가 | ~0.25 | ~35,000원 | 예상치 |
| Step D+E 후 | 인코딩 + 계절 보정 | ~0.28 | ~30,000원 | 예상치 |
| **v2 실제 (A~E 전부)** | **전체 소비, 12 피처, income_tier** | **0.1277 (test)** | **129,653원 (test)** | **✅ 완료 — 타겟이 전체소비로 변경되어 MAE 절대값 증가** |
| Step F 후 | 전국 데이터 (10,000+행) | ~0.25~0.35 | ~80,000원 | AI Hub 전국 여행로그 신청 후 |

### 2-4. 새 모델 피처 설계 (최종)
// [메모] 소득분위에서 연속형으로 말고 구간으로 나누는게 더 좋을 것 같습니다. 짠순이/보통/호캉스 등 3단계로 나눠 진행합니다 

| 피처명 | 원본 컬럼 | 전처리 | 타입 |
|--------|---------|--------|------|
| sgg_enc | SGG_CD | TargetEncoding (시군구별 평균 소비) | float |
| travel_duration_h | TRAVEL_START/END | 시간 차이 계산 | float |
| distance_km | 경로 좌표 | Haversine 합산 | float |
| companion_cnt | COMPANION_CNT | 그대로 | int |
| season | 날짜 | month → 계절 매핑 | int (1~4) |
| day_of_week | 날짜 | 요일 | int (0~6) |
| has_lodging | LODGE 테이블 | 숙박 여부 | bool |
| income | INCOME | 소득 분위 (1~8) | int |
| age_grp | AGE_GRP | 연령대 코드 | int |
| gender | GENDER | 성별 (M/F → 0/1) | int |
| travel_purpose | TRAVEL_PURPOSE | LabelEncoding | int |
| travel_styl_avg | TRAVEL_STYL_1~8 | 평균값 | float |
| **log_total_consume** | 4개 소비 테이블 합산 | log1p 변환 | float (타겟) |

---

## 3. 날씨 LSTM — 성능 향상 계획

> **현재**: Test Acc=73.28% (3-class, 랜덤 기준 33%)
> **목표**: Test Acc ≥ 80%

### 3-1. 현재 한계 분석

| 한계 | 수치 | 개선 방향 |
|------|------|---------|
| 흐림(클래스 1) 데이터 7%뿐 | 826/10,960행 | SMOTE 합성 또는 데이터 추가 수집 |
| 입력 피처 8개 | [월, 일, 요일, 기온, 강수량, 풍속, 습도, 관측소코드] | 이슬점, 기압, 일조시간 추가 |
| 시퀀스 길이 14일 고정 | — | 7/14/21일 비교 실험 |
| 단일 LSTM 구조 | hidden=64, layers=2 | Bi-LSTM or Transformer 비교 |
| 5개 관측소만 사용 | 서울, 수원, 인천, 양평, 이천 | 전국 확대 시 관측소 확대 |

### 3-2. 개선 방안

#### 방안 A. 피처 추가 (ASOS API 확장)

| 추가 피처 | ASOS 컬럼명 | 예상 효과 |
|---------|-----------|---------|
| 이슬점 온도 | DP_TEMP | 체감 온도 → 자전거 쾌적도 연관 |
| 기압 (해면) | SEA_LEVEL_PRS | 기압 하강 = 날씨 악화 예측 |
| 일조 시간 | SS_HR | 맑음 클래스 정밀도 향상 |
| 최고/최저 기온 | MAX_TEMP, MIN_TEMP | 기온 변동폭 반영 |

#### 방안 B. 클래스 불균형 해소

```python
# 흐림(826행) → SMOTE로 증강
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy={1: 2000}, random_state=42)
X_res, y_res = sm.fit_resample(X_flat, y)

# 또는 Focal Loss 사용 (흐림 클래스 가중 증가)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        ...
```

#### 방안 C. 모델 구조 개선

| 모델 | 파라미터 수 | 예상 Acc | 학습 시간 |
|------|-----------|---------|---------|
| 현재 LSTM (hidden=64, layers=2) | ~50K | 73.28% | 기준 |
| Bi-LSTM (hidden=64, layers=2) | ~100K | ~76% | 1.5배 |
| LSTM + Attention | ~60K | ~77% | 1.3배 |
| Transformer (d_model=64, heads=4) | ~80K | ~78% | 2배 |

**권장**: 먼저 피처 추가(A)와 클래스 불균형(B)을 적용 후 재학습.
모델 구조 변경(C)은 A+B 후에도 미달 시 시도.

---

## 4. 관광 점수 — 구조적 개선 계획

> **현재 문제**: 65.2% 세그먼트 tourism_score=0, 평균 0.088
> **근본 원인**: POI 밀도가 낮고 경기도 레저스포츠 83개뿐

### 4-1. 단기 개선 (데이터 보완)

| 개선 항목 | 방법 | 결과/예상 효과 |
| --- | --- | --- |
| ✅ 경기도 레저스포츠 POI 재수집 | step3_tour_collect_v2.py --sido all | 경기 845건 수집 완료 (v1: 83건 타임아웃 → 해결) |
| ✅ 전국 POI 데이터 수집 | --sido all --include_lodging | **15,905건** (관광지 8,931 / 레저 3,243 / 숙박 2,080 / 문화 1,651), 오류 없음 |
| 관광지 반경 확대 | 1km → 2km Spatial Join | tourism_score>0 세그먼트 65% → 80%+ (미적용, Step 4 전처리 때 반영 예정) |

### 4-2. 중기 개선 — 네이버 DataLab 검색 트렌드 연동

> **현황 (2026-04-16)**: 네이버 DataLab API 활성화 ✅ | 네이버 Cloud Maps Geocoding 등록 ✅ | 카카오 로컬 REST API 사용 가능 ✅ (비즈니스 등록 불필요, FD6/CE7, 300,000콜/일) | 카카오 지도 SDK → Naver Cloud Maps Dynamic Map으로 대체

#### DataLab API 제약 및 배치 처리 전략

| 항목 | 내용 |
| --- | --- |
| 일일 한도 | 1,000콜/일 |
| 요청당 최대 키워드 | 5개 그룹 |
| 처리 가능 POI/일 | 5,000개 (1,000 × 5) |
| 전국 POI 15,905건 처리 | 약 4일 (배치 분산) |
| 활용 용도 | 장소별 평균 검색 비율(0~100) → `sns_mention_norm` |

#### 관광 점수 공식 (현재 적용 가능 범위)

```
tourism_score_final =
    0.5 × poi_density_score     ← TourAPI 15,905건 수집 완료 ✅
  + 0.2 × attraction_score      ← TabNet POI 매력도 (기존 모델)
  + 0.2 × food_poi_density      ← 네이버 지역 검색 API (추가 신청 후 수집 예정)
  + 0.1 × sns_mention_norm      ← 네이버 DataLab 검색 트렌드 (API 발급 완료 ✅)
```

SNS 크롤링(Phase 8) 도입 전까지 DataLab 검색 트렌드를 `sns_mention_norm` 대리 지표로 사용.

### 4-3. 장기 개선 (감성 분석 KLUE-BERT)

리뷰 텍스트 수집 후 적용:

| 모델 | 입력 | 출력 | 데이터 요건 |
|------|------|------|-----------|
| snunlp/KR-FinBert-SC (zero-shot) | 리뷰 텍스트 | 긍정 확률 (0~1) | 수집만 하면 즉시 가능 |
| KLUE-RoBERTa fine-tuned | 리뷰 텍스트 | 긍정/부정 분류 | 라벨 데이터 필요 |

---

## 5. 안전 모델 — 고도화 방향

> **현재 성능**: R²=0.9539, F1=0.9864 (이미 우수)
> **한계**: 1,647개 서울 세그먼트만 학습, 실제 사고 데이터 미반영

### 5-1. 현재 한계

| 한계 | 현황 | 개선 방법 |
|------|------|---------|
| 학습 데이터 1,647행 | 서울 일부만 커버 | 전국 확대 시 10,000+ |
| safety_index 타겟이 규칙 기반 | 실사고 데이터 미반영 | TAAS 사고 다발지 데이터 연동 |
| 위경도 피처만 사용 | 도로 구조 정보 미반영 | OSM 도로 속성 (경사, 교차로 수) 추가 |
| 서울 지역 district_danger만 | 경기도·전국 없음 | 전국 사고 데이터 수집 |

### 5-2. TAAS 사고 데이터 연동 계획

```
데이터: 경찰청 교통사고분석시스템 (TAAS)
URL: https://taas.koroad.or.kr/
내용: 자전거 사고 발생 좌표 + 사망/부상/사고 유형
비용: 무료 (회원가입 후 다운로드)

연동 방법:
  TAAS 사고 좌표 → road_scored.csv 세그먼트 Spatial Join (반경 300m)
  → accident_count 피처 추가 → safety_index 타겟 재설계
  → safety_index_v3 = 0.5×(1-district_danger) + 0.3×road_attr + 0.2×(1-accident_rate)
```

### 5-3. TabNet Safety — Phase 5 도입 조건

| 조건 | 현재 | 충족 시점 |
|------|------|---------|
| 학습 데이터 3,000행 이상 | 1,647행 | 전국 확대 후 |
| TAAS 사고 데이터 연동 | 미완 | TAAS 다운로드 후 |
| RF 앙상블 필요성 검토 | RF R²=0.9539 충분 | 전국 데이터 학습 후 재평가 |

---

## 6. Co-occurrence POI 추천 — Recall 향상

> **v2 실행 결과 (2026-04-16)**: Recall@5=0.1372 / Recall@10=0.1811 (test, 부스트 O) — 베이스라인(0.0370) 대비 **3.7배** 향상  
> 전국 POI 15,905건 수집 → 15,851개 카테고리 연동, 7,748장소 좌표 보유, vocab 1,646개 (train 1,775여행)  
> **목표**: Recall@5≥0.20 (전국 AI Hub 여행로그 확보 후 vocab 확대로 달성 예정)

### 6-1. 향상 방법

| 방법 | 예상 Recall@5 개선 | 난이도 |
|------|------------------|--------|
| **지리 필터 적용** — 현재 위치 반경 내 POI만 추천 | +0.03~0.05 | 낮음 |
| **BM25 텍스트 유사도 결합** — POI 설명 텍스트 활용 | +0.02~0.04 | 중간 |
| **카테고리 가중치** — 같은 카테고리 POI 부스트 | +0.01~0.02 | 낮음 |
| **전국 데이터** — 2,560 → 10,000+ 여행 | +0.05~0.08 | 높음 (데이터 확보 필요) |
| **Matrix Factorization (ALS)** — 암묵적 피드백 기반 | +0.04~0.07 | 중간 |

### 6-2. 지리 필터 개선 코드

```python
def recommend_with_geo(seeds, current_lat, current_lon, radius_km=20, top_n=10):
    candidates = _compute_jaccard_scores(seeds)
    # 좌표 있는 POI만 필터
    poi_df = tour_poi_df[tour_poi_df["mapx"].notna()]
    # 현재 위치 반경 필터
    poi_df["dist_km"] = poi_df.apply(
        lambda r: haversine((current_lat, current_lon), (r.mapy, r.mapx)), axis=1
    )
    local_pois = poi_df[poi_df["dist_km"] <= radius_km]["title"].tolist()
    # 반경 내 POI만 추천 대상으로 제한
    candidates = {k: v for k, v in candidates.items() if k in local_pois}
    return sorted(candidates, key=candidates.get, reverse=True)[:top_n]
```

---

## 7. 전국 확대 계획

> **현재**: 서울 자전거도로 1,647 세그먼트, 서울/경기 AI Hub 데이터
> **목표**: 전국 광역시도 자전거도로 + 전국 AI Hub 여행로그

### 7-1. 데이터 수집 현황 및 계획

#### 자전거도로 데이터

| 광역시도 | 현재 상태 | 수집 방법 | 예상 세그먼트 수 |
|---------|---------|---------|--------------|
| 서울 | ✅ 완료 | 공공데이터포털 | 1,647개 |
| 경기 | ⚠️ 일부 (road_clean.csv 포함) | 공공데이터포털 | ~3,000개 |
| 인천 | ❌ 미수집 | 공공데이터포털 | ~800개 |
| 부산 | ❌ 미수집 | 공공데이터포털 | ~600개 |
| 대구 | ❌ 미수집 | 공공데이터포털 | ~400개 |
| 광주 | ❌ 미수집 | 공공데이터포털 | ~300개 |
| 대전 | ❌ 미수집 | 공공데이터포털 | ~350개 |
| 전국 | — | 공공데이터포털 통합 | ~15,000개+ |

#### OSM 경로 그래프 (전국 확대)

```python
# 현재: 서울 bbox만
# 확대: 전국 시도별 또는 전국 한번에

# 방법 1: 시도별 bbox 분할 (메모리 절약)
SIDO_BBOXES = {
    "서울": (37.413, 37.715, 126.764, 127.185),
    "경기": (37.000, 38.300, 126.500, 127.900),
    "부산": (35.000, 35.400, 128.800, 129.300),
    # ...
}

# 방법 2: 전국 한번에 (RAM 8GB+ 필요)
G = ox.graph_from_place("South Korea", network_type="bike")
```

#### AI Hub 여행로그 전국 확대

| 데이터셋 | 현재 | 확대 방법 | 예상 행 수 |
|---------|------|---------|---------|
| 수도권 여행로그 2023 | ✅ 2,560 여행 | — | 2,560 여행 |
| 전국 여행로그 (별도 신청) | ❌ 미보유 | AI Hub 추가 신청 | ~15,000 여행 |
| 한국관광공사 TourAPI | ✅ 완료 (2026-04-16) | step3_tour_collect_v2.py --sido all --include_lodging | 전국 15,905건 (관광지+문화+레저+숙박) |

### 7-2. 전국 확대 시 모델별 예상 효과

| 모델 | 현재 (서울) | 전국 확대 후 |
|------|-----------|-----------|
| 소비 TabNet | R²=0.0053 | R²≥0.25 (데이터 10배 증가) |
| 안전 RF | R²=0.9539 (1,647행) | R²≥0.97 (15,000행) |
| Co-occurrence | Recall@5=0.1372 (v2, 2,560 여행) | Recall@5≥0.20 (전국 여행로그 확보 후) |
| 날씨 LSTM | Acc=73.28% (5 관측소) | Acc≥80% (전국 관측소) |

### 7-3. 전국 데이터 수집 우선순위

| 순위 | 데이터 | 출처 | 비용 | 소요 시간 |
|-----|------|------|------|---------|
| 1 | 전국 자전거도로 (공공데이터포털) | data.go.kr | 무료 | 1일 |
| 2 | 전국 AI Hub 여행로그 | aihub.or.kr | 무료 | 1~3일 승인 |
| 3 | 한국관광공사 TourAPI (전국) | data.visitkorea.or.kr | 무료 | 즉시 |
| 4 | TAAS 전국 자전거 사고 | taas.koroad.or.kr | 무료 | 즉시 |
| 5 | 전국 기상 관측소 (ASOS) | data.kma.go.kr | 무료 | 즉시 |

---

## 8. 데이터베이스 설계 — 배포 대비 스키마

> 배포 환경: SpringBoot + PostgreSQL + PostGIS (공간 쿼리)
> 현재 CSV 기반 → RDBMS 마이그레이션 계획

### 8-1. 테이블 분리 계획 (현재 CSV → DB 스키마)

#### 핵심 테이블 목록

| 테이블명 | 원본 CSV | 행 수 | 핵심 컬럼 | 비고 |
|---------|---------|-------|---------|------|
| `road_segment` | road_scored_v2.csv | 1,647 → 전국 15,000+ | id, geom, safety_score, tourism_score, final_score | PostGIS GEOMETRY |
| `poi` | tour_poi.csv | 2,529 → 전국 25만 | id, name, category, geom, attraction_score | PostGIS POINT |
| `facility` | facility_clean.csv | 3,368 | id, type, name, geom | PostGIS POINT |
| `travel_log` | tn_travel 등 4개 테이블 | 2,560 → 전국 15,000+ | id, travel_id, sido, duration, total_consume | 머지 결과 |
| `visit_area` | tn_visit_area_info | 21,384 → 전국 | travel_id, poi_name, geom, satisfaction, residence_min | — |
| `road_accident` | TAAS 사고 데이터 (미수집) | 전국 | id, geom, accident_type, year | PostGIS POINT |
| `weather_daily` | weather_asos_daily.csv | 5,480 | station_id, date, temp, rain, wind, label | 관측소별 |
| `user_profile` | (신규) | — | id, age_grp, gender, bike_type, home_sgg | 서비스 사용자 |

#### road_segment 테이블 DDL (PostgreSQL + PostGIS)

// [메모] tourism_score와 tourism_score2 둘다 사용 해야 하나요? 하나로 사용할 수 있다면 하나로 사용하고 두개 가 필요하면 칼럼 이름을 변경할 필요가 있습니다
```sql
CREATE TABLE road_segment (
    id              SERIAL PRIMARY KEY,
    sido_nm         VARCHAR(20),          -- 시도명
    sgg_nm          VARCHAR(30),          -- 시군구명
    road_name       VARCHAR(100),         -- 노선명
    road_type       VARCHAR(30),          -- 도로 유형
    length_km       DECIMAL(8,3),
    width_m         DECIMAL(6,2),
    geom            GEOMETRY(LINESTRING, 4326),  -- PostGIS 공간 컬럼
    safety_score         DECIMAL(5,4),
    tourism_score        DECIMAL(5,4),         -- POI 밀도 기반 (규칙 기반 v1, 마이그레이션 완료 후 제거 예정)
    tourism_score_final  DECIMAL(5,4),         -- 통합 점수 (v2: POI+attraction+SNS 가중 합산) — 서비스 사용 컬럼
    final_score          DECIMAL(5,4),
    final_score_v2       DECIMAL(5,4),
    district_danger DECIMAL(5,4),
    updated_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_road_geom ON road_segment USING GIST(geom);
CREATE INDEX idx_road_sido ON road_segment(sido_nm);
```

#### poi 테이블 DDL

```sql
CREATE TABLE poi (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200),
    category        VARCHAR(50),          -- 관광지/문화/레저/편의시설
    sido_nm         VARCHAR(20),
    sgg_nm          VARCHAR(30),
    address         TEXT,
    geom            GEOMETRY(POINT, 4326),
    attraction_score DECIMAL(5,4),        -- TabNet 예측값
    visit_count     INTEGER DEFAULT 0,    -- Co-occurrence 빈도
    updated_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_poi_geom ON poi USING GIST(geom);
CREATE INDEX idx_poi_category ON poi(category);
```

#### consume_prediction 테이블 DDL

```sql
CREATE TABLE consume_prediction (
    id              SERIAL PRIMARY KEY,
    sgg_code        VARCHAR(10),
    duration_h      DECIMAL(5,2),
    distance_km     DECIMAL(8,3),
    companion_cnt   INTEGER,
    season          SMALLINT,             -- 1=봄, 2=여름, 3=가을, 4=겨울
    has_lodging     BOOLEAN,
    income_tier     SMALLINT,            -- 소비 성향 (0=짠순이 1~3분위, 1=보통 4~6분위, 2=호캉스 7~8분위)
    age_grp         VARCHAR(10),
    gender          CHAR(1),
    predicted_amt   INTEGER,             -- TabNet 예측 소비금액 (원)
    log_predicted   DECIMAL(8,4),        -- 로그 예측값
    model_version   VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW()
);
```

### 8-2. PostGIS 공간 쿼리 활용

```sql
-- 특정 좌표 반경 2km 내 안전한 경로 조회
SELECT id, road_name, safety_score, tourism_score
FROM road_segment
WHERE ST_DWithin(
    geom::geography,
    ST_SetSRID(ST_MakePoint(126.9780, 37.5665), 4326)::geography,
    2000  -- 2000m
)
ORDER BY final_score_v2 DESC
LIMIT 20;

-- 경로 선상 500m 내 POI 조회 (경로 추천 결과 활용)
SELECT p.name, p.category, ST_Distance(p.geom::geography, r.geom::geography) as dist_m
FROM poi p, road_segment r
WHERE r.id = :segment_id
  AND ST_DWithin(p.geom::geography, r.geom::geography, 500)
ORDER BY dist_m;
```

---

## 9. 모델 향상 우선순위 최종 정리

### 9-1. 즉시 실행 가능한 것 (데이터 있음)

| 순위 | 작업 | 예상 개선 | 소요 시간 |
|-----|------|---------|---------|
| 1 | 소비 모델 이상치 제거 + 로그 변환 | R² 0.005 → 0.05 | 1시간 |
| 2 | 소비 모델 피처 추가 (traveller_master 머지) | R² → 0.20+ | 2~3시간 |
| 3 | 소비 모델 타겟 재정의 (4개 테이블 합산) | R² → 0.15 | 2시간 |
| 4 | Co-occurrence 지리 필터 적용 | Recall@5 → 0.18 | 1시간 |
| 5 | 경기도 레저스포츠 POI 재수집 | tourism_score 분포 개선 | 2시간 |

### 9-2. 데이터 수집 후 실행 가능한 것

| 순위 | 작업 | 선행 조건 | 예상 개선 |
|-----|------|---------|---------|
| 1 | 전국 자전거도로 데이터 (data.go.kr) | 다운로드 | 전국 서비스 기반 |
| 2 | TAAS 사고 데이터 연동 | 회원가입 + 다운로드 | safety 타겟 품질 향상 |
| 3 | AI Hub 전국 여행로그 신청 | 승인 (1~3일) | 소비 모델 R² 0.35+ |
| 4 | KLUE-BERT 감성 분석 | 리뷰 크롤링 | tourism_score 정교화 |
| 5 | AI Hub 자전거도로 이미지 신청 | 승인 (1~3일) | CNN 도로 상태 분류 |

### 9-3. 폐기/보류 항목

| 항목 | 이유 | 대안 |
|------|------|------|
| GRU 방문지 시퀀스 | 구조적 한계 (vocab 희소성) | Co-occurrence로 대체 완료 |
| 네이버 로드뷰 CNN | 약관 금지 | AI Hub 자전거도로 이미지 |
| Neural CF 개인화 | 데이터 부족 + 복잡도 높음 | Co-occurrence로 MVP, 추후 ALS |
| TabNet Safety (Phase 5) | RF R²=0.9539 충분 | 전국 데이터 확보 후 재검토 |
