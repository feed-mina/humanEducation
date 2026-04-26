# K-Ride 2.0 Research Log

> **최종 업데이트**: 2026-04-25
> **목적**: 기존 K-Ride 자산 분석 + 신규 데이터 소스 조사 + 모델링 전략 연구

---

## 프로젝트 기획
```
* 프로젝트 계획은 바꾸려고 합니다. 

기획의도는 한국인/외국인 기반으로 새로운 관광 서비스 제공하는 서비스를 제공하려고 합니다. 
대상 : 한국인, K한류/Kpop을 좋아하는 외국인
목적 : (1)한국 여행을 하면서 맛집정보 (유튜브 또간집, 블루리본, 방송에서 나온 맛집 등) 와 (2-1)자신이 좋아하는 아이돌/가수/배우 등이 뮤직비디오, 영화, 드라마 촬영지나 SNS에 올린 장소나 팬덤이 방문하고 싶은 장소 정보 (2-2) 요즘 떠오르고 있는 케데헌/한국 문화를 알고싶어하는 사람들을 위한 국립중앙박물관 (서울, 경주, 기타 등등) 과 유명 관광지 (2-3) 자연을 좋아하는 사람이라면 자연치화적이면 아마 전국에서 제공하는 지역마다 스탬프를 찍을 수 있는 둘레길이나 관광경로 정보 (4) 당일치기, 1박2일, 2박3일, 3박 4일 정도로 추천해주는 관광지 경로로 이동할때 들어가는 예측 소비,  (5) 이동수단은 걷기, 자전거, 오토바이/자동차, 기차여행 등이 있습니다. 

(6)
공공데이터나 사용가능한 api / 크롤링도 한가지 방법이 될수있음 을 통해 학습데이터를 찾고 로컬에서 ml,dl,llm 등 모델링 
(7) ollma를 통한 모델 
(8) groq, huggingface, roboflow, kaggle 에서도 필요한 데이터(원천 또는 api )를 받아서 사용할수 있음 
(9) rag, graphrag와 mcp까지 확대 예정 
(10) 저비용을 위한 원천학습  모델링 생성> 오픈소스 모델링 사용 > 배포를 위해 최대한 경량화  (허깅페이스, 로보플로우, 캐글, groq등 외부에 올릴 수 있다면 올려서 사용)
(11) 구글 드라이브에서 gpu 사용이나 클라우드 배포까지 

현재 학습하고 모델링 되어있는 부분에서 전국으로 범위를 확대하고 지도 기반의 새로운 비즈니스 를 만들어보려고 합니다. 이동수단에 필요한 학습 부분은 시간이 오래 걸릴꺼 같아 1차 MVP 이후에 2차나 3차 MVP 에 붙이고 싶습니다 

기획의도에 맞춰서 현재 상태와 plan, research를 전부 읽고 새로운 new_plan, new_research 문서를 만들어주세요. 


```

## 1. 기존 자산 성능 현황 (2026-04-25 기준)

### 1-1. 모델 파일 상태 진단

| 모델 | 파일 | 크기 | 상태 | 재활용 |
|------|------|------|------|--------|
| 안전 RF 회귀 | safety_regressor.pkl | 135B | ❌ Git LFS 포인터 (빈 파일) | 재학습 필요 |
| 안전 RF 분류 | safety_classifier.pkl | 134B | ❌ Git LFS 포인터 | 재학습 필요 |
| 소비 TabNet v2 | consume_regressor_v2.zip | 392KB | ✅ 정상 | ✅ 재활용 |
| POI 매력도 TabNet | attraction_regressor.zip | 96KB | ✅ 정상 | ✅ 재활용 |
| Co-occurrence POI | poi_cooccurrence_v2.pkl | 21MB | ✅ 정상 | ✅ 재활용 |
| WeatherLSTM | weather_lstm.pt | 213KB | ✅ 정상 | ✅ 재활용 |
| 경로 그래프 | route_graph.pkl | 136B | ❌ Git LFS 포인터 | 재빌드 필요 |

### 1-2. 모델 성능 대시보드

| 모델 | 성능 | 학습 데이터 범위 | 전국 확대 필요 |
|------|------|----------------|:---:|
| 안전 RF | R²=0.9539, F1=0.9864 | 수도권 (4,998행) | ✅ |
| 소비 TabNet v2 | MAE=₩129,653, R²=0.1277 | 수도권 (2,508행) | ✅ |
| POI 매력도 | MAE=0.6558, R²=0.0662 | 수도권 (14,364행) | ✅ |
| Co-occurrence | Recall@5=0.1372 | 수도권 (2,536 여행) | ✅ |
| WeatherLSTM | Acc=73.28% | 수도권 5개 관측소 | ✅ |
| GRU 시퀀스 | top5=0.14 | 수도권 | ❌ 폐기 |

### 1-3. 데이터 파일 현황

| 파일 | 행수 | 범위 | 신규 프로젝트 활용 |
|------|------|------|:--:|
| tour_poi.csv | 15,905 | **전국 17개 시도** | ✅ 핵심 |
| road_clean_v2.csv | 4,998 | 수도권+일부 | ⚠️ 전국 확대 |
| road_features.csv | 1,647 | 서울만 | ⚠️ 전국 확대 |
| facility_clean.csv | 3,368 | 서울 | ⚠️ 전국 확대 |
| weather_asos_daily.csv | 5,480 | 수도권 5개소 | ⚠️ 관측소 추가 |
| poi_attraction.csv | 8,454 | 수도권 | ⚠️ 전국 확대 |
| AI Hub 여행로그 4개 | 2,560여행 | 수도권 | ⚠️ 전국 신청 |

---

## 2. 데이터 소스 조사

### 2-1. 맛집 데이터

#### 카카오 로컬 REST API (FD6/CE7)
```
상태: ✅ API 키 보유, 스크립트 보유 (step3_food_collect.py)
쿼터: 300,000콜/일
방법: 전국 시군구 중심 좌표 기준 반경 검색
예상: ~50,000건 (중복 제거 후)
장점: 정확한 좌표, 카테고리, 전화번호, 리뷰수
단점: 별점/리뷰 텍스트 미제공
```

#### 블루리본 서베이
```
상태: [ ] 조사 필요
URL: bluer.co.kr
robots.txt: 확인 필요
방법: requests + BeautifulSoup (지역별 맛집 목록)
예상: ~3,000건 (블루리본 인증점)
장점: 큐레이션된 고품질 맛집
단점: 크롤링 약관 확인 필요
```

#### 유튜브 또간집/백종원
```
상태: [ ] 조사 필요
방법 A: YouTube Data API v3 (영상 검색 + 설명 텍스트 파싱)
방법 B: 나무위키/위키 "또간집 목록" 페이지 파싱
예상: ~500건
장점: 대중적 인지도 높은 맛집
단점: 가게명 → 좌표 변환 필요 (카카오 Geocoding)
```

#### 공공데이터 맛집
```
소스: data.go.kr "모범음식점" / "착한가격업소"
상태: [ ] 확인 필요
예상: 지역별 수백~수천건
장점: 공식 인증, API 제공
단점: 트렌디한 맛집과 차이 있음
```

### 2-2. K-Culture 성지순례

#### 한국관광공사 TourAPI 촬영지
```
상태: ✅ API 키 보유, 스크립트 보유
방법: contentTypeId=25 (여행코스) 중 "촬영지" 키워드 필터
     또는 cat3="A02060300" (촬영지)
예상: ~2,000건 (전국)
장점: 공식 API, 좌표 포함
```

#### K-Drama/K-Pop 위키
```
소스 후보:
  - 나무위키: "OO (드라마)/촬영지" 문서
  - 한국영화데이터베이스 (KMDb): kobis.or.kr
  - 위버스/팬카페: 팬덤 성지 목록 (수동 큐레이션)
방법: requests + BeautifulSoup
예상: ~1,000건
주의: 좌표 없음 → 장소명 기반 Geocoding 필요
```

#### 아이돌/배우 SNS 장소
```
방법: Instagram 해시태그 (#BTSspots, #블랙핑크촬영지 등)
     → SerpAPI 또는 직접 크롤링
예상: ~500건
주의: robots.txt + 약관 확인, 변경 빈번
대안: 팬 커뮤니티 위키 기반 수동 큐레이션
```

### 2-3. 문화·관광지

#### 한국관광공사 TourAPI
```
상태: ✅ 15,905건 보유 (tour_poi.csv)
분포: 경기 2,460 / 강원 2,074 / 경북 1,594 / 경남 1,442 / 전남 1,376 ...
카테고리: 관광지 8,931 / 레저 3,243 / 숙박 2,080 / 문화 1,651
활용: 문화·관광지 카테고리 기본 DB로 그대로 사용
```

#### 국립중앙박물관 API
```
소스: museum.go.kr 또는 data.go.kr
내용: 서울/경주/나주/김해/춘천 등 분관 정보
상태: [ ] API 명세 확인 필요
```

### 2-4. 둘레길·자연

#### 두루누비 (durunubi.kr)
```
소스: 한국관광공사 도보여행 포털
내용: 전국 둘레길/트레킹 코스 GPS 경로
방법: 공공데이터 API 또는 웹 크롤링
예상: ~800코스
핵심: 스탬프 시스템 연동 가능
```

#### 산림청 숲길
```
소스: data.go.kr "산림청_전국자연휴양림"
내용: 전국 자연휴양림, 숲길, 둘레길
상태: [ ] API 명세 확인 필요
```

---

## 3. LLM 전략 연구

### 3-1. Ollama 로컬 LLM

| 모델 | 크기 | VRAM | 한국어 성능 | 권장 용도 |
|------|------|------|-----------|----------|
| llama3.1:8b | 4.7GB | 6GB | 중 | 코스 생성, 대화 |
| gemma2:9b | 5.5GB | 8GB | 중상 | 한국어 질의 처리 |
| EEVE-Korean-10.8B | 6.5GB | 8GB | 상 | 한국어 특화 |
| phi3:mini | 2.3GB | 4GB | 하 | 경량 테스트 |

```bash
# 설치 및 실행
ollama pull llama3.1:8b
ollama serve  # localhost:11434

# Python 연동
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434")
```

### 3-2. Groq API (배포 환경)

```
무료 tier: 30 requests/min, 14,400 requests/day
모델: llama-3.1-70b-versatile (가장 강력)
지연: ~0.5초 (매우 빠름)
장점: 로컬 GPU 불필요, 배포 환경 적합
단점: rate limit, 외부 의존성
```

### 3-3. RAG 파이프라인 설계

```python
# ChromaDB 벡터 스토어 구성
from chromadb import Client
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("intfloat/multilingual-e5-small")  # 한영 다국어

collection = chroma.create_collection("poi_db")

# POI 데이터 임베딩
for poi in poi_list:
    text = f"{poi.name} {poi.category} {poi.address} {poi.description}"
    embedding = embedder.encode(text)
    collection.add(
        documents=[text],
        embeddings=[embedding.tolist()],
        metadatas=[{"id": poi.id, "category": poi.category, "lat": poi.lat, "lon": poi.lon}],
        ids=[str(poi.id)]
    )

# 검색
results = collection.query(
    query_texts=["BTS 뮤비 촬영지 근처 맛집"],
    n_results=10
)
```

### 3-4. GraphRAG 확대 (2차 MVP)

```
노드: POI (맛집/관광지/K-Culture/둘레길)
엣지:
  - NEAR (거리 < 2km)
  - SAME_CATEGORY (같은 카테고리)
  - SAME_TRIP (같은 여행에서 함께 방문)
  - FILMING_LOCATION (같은 작품 촬영지)

질의 예시:
  "BTS MV 촬영지에서 걸어서 갈 수 있는 맛집"
  → BTS 노드 → FILMING_LOCATION → 촬영지 노드 → NEAR → 맛집 노드
```

### 3-5. MCP (3차 MVP)

```
Model Context Protocol 도구:
  - weather_tool: KMA API 실시간 날씨
  - map_tool: 네이버/카카오 지도 경로 검색
  - transit_tool: TAGO 대중교통 API
  - cost_tool: TabNet 소비 예측 모델

LLM이 도구를 직접 호출:
  "내일 서울 1박2일 여행 계획 세워줘"
  → weather_tool("서울", "내일") → 맑음 22°C
  → map_tool("경복궁→홍대") → 4.5km, 도보 50분
  → cost_tool(duration=2, companion=2) → ₩185,000
```

---

## 4. 다국어 지원 전략

### 4-1. POI 데이터 다국어화

| 방법 | 비용 | 품질 |
|------|------|------|
| TourAPI 영문 데이터 | 무료 | 높음 (공식 번역) |
| Google Translate API | 유료 ($20/100만자) | 중 |
| Ollama 번역 | 무료 | 중하 |
| 수동 번역 (핵심 POI만) | 인건비 | 최상 |

```python
# TourAPI 영문 조회
# MobileOS=ETC&MobileApp=KRide&_type=json&arrange=A&contentTypeId=12
# → title, addr1 등이 영문으로 반환됨
```

### 4-2. 검색 다국어 대응

```
한국어: "BTS 촬영지 근처 맛집"
영어:   "restaurants near BTS filming locations"
  ↓ multilingual-e5-small 임베딩
  ↓ 같은 벡터 공간에서 검색
  → 동일한 POI 결과 반환
```

---

## 5. 소비 예측 모델 확장 계획

### 5-1. 현재 모델 (TabNet v2)

```
입력: sgg_enc, travel_duration_h, distance_km, companion_cnt,
      season, day_of_week, has_lodging, income_tier, age_grp_enc,
      gender_enc, travel_purpose_enc, travel_styl_avg
출력: log_total_consume → 실제 금액 (역변환)
성능: MAE=₩129,653 / R²=0.1277
한계: 수도권 2,508행, 계절 편향 (여름 82.7%)
```

### 5-2. 일정별 예측으로 확장

```
기존: 여행 1건 → 총 소비 예측
확장: 일정(당일/1박2일/2박3일/3박4일) + 카테고리별 예측

입력 추가:
  - duration_days (1~4)
  - n_restaurants (방문 맛집 수)
  - n_attractions (방문 관광지 수)
  - accommodation_type (없음/게스트하우스/호텔)
  - transport_type (walk/bike/public/car)

출력:
  - 식비 예측
  - 숙박비 예측
  - 교통비 예측
  - 입장료 예측
  - 총 예상 비용
```

---

## 6. 데이터 수집 우선순위

| 순위 | 데이터 | 소스 | 난이도 | 예상 소요 |
|------|--------|------|:---:|---------|
| 1 | 전국 맛집 POI | 카카오 로컬 API | 낮 | 2~3시간 (스크립트 보유) |
| 2 | K-Culture 촬영지 | TourAPI + 위키 크롤링 | 중 | 1~2일 |
| 3 | 둘레길 코스 | 두루누비 공공데이터 | 중 | 1일 |
| 4 | 블루리본 맛집 | 크롤링 | 중 | 반나절 |
| 5 | 안전 모델 재학습 | 로컬 실행 | 낮 | 5분 (openpyxl 설치 후) |
| 6 | 전국 AI Hub 여행로그 | aihub.or.kr 신청 | — | 승인 1~3일 |
| 7 | 아이돌 SNS 장소 | 수동 큐레이션 | 높 | 지속적 |

---

## 7. 외부 플랫폼 활용 계획

| 플랫폼 | 용도 | 상태 |
|--------|------|------|
| **HuggingFace** | ML 모델 호스팅 (pkl/pt/zip) + Spaces 데모 | [ ] 레포 생성 |
| **Roboflow** | 이미지 분류 모델 (관광지 사진 등) | [ ] 필요 시 |
| **Kaggle** | 학습 데이터셋 공유 + GPU 학습 | [ ] Notebook 작성 |
| **Groq** | 배포 환경 LLM 추론 (무료 tier) | [ ] API 키 발급 |
| **Google Colab** | GPU 학습 (A100/T4) | [ ] Drive 연동 |

---

## 8. 기존 research.md에서 계승하는 핵심 인사이트

### 유지할 연구 결과
- **safety_index_v2 공식**: `(1-district_danger)×0.6 + road_attr_score×0.4` — 재학습에 그대로 사용
- **소비 모델 income_tier 구간화**: 짠순이(1~3)/보통(4~6)/호캉스(7~8) — 유지
- **Co-occurrence Jaccard + 지리 필터**: Recall@5=0.1372 달성 방법론 유지
- **날씨 보정값**: 맑음 0.0 / 흐림 -0.05 / 비눈 -0.20 — 유지
- **데이터 모수 문제**: 좌표 결측 68% → 전국 확대 시 Geocoding 보완 전략

### 폐기/대체할 연구
- GRU 방문지 시퀀스 → **LLM 기반 코스 생성으로 대체**
- 자전거 전용 프로파일 (BMI/자전거종류) → **2차 MVP로 이동**
- Phase 7 셀카 패션 추천 → **프로젝트 범위 외**
- TabNet Safety Phase 5 → RF R²=0.9539 충분, 전국 확대 후 재검토
