# K-Ride 2.0 — 한국 관광 AI 플랫폼 마스터 플랜

> **최종 업데이트**: 2026-04-25
> **프로젝트명**: K-Ride 2.0 (Korea Ride & Discover)
> **슬로건**: "한국을 발견하다 — AI가 만드는 나만의 여행 코스"

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

## 1. 기획 의도

### 대상
- **한국인**: 국내 여행·맛집·둘레길·문화 관광 수요
- **외국인**: K-Pop/K-Drama/K-Culture 팬덤, 한국 여행 계획자

### 서비스 핵심 기능 (5개 카테고리)

| # | 카테고리 | 내용 | 데이터 소스 |
|---|---------|------|-----------|
| 1 | 🍽️ **맛집 정보** | 유튜브 또간집·블루리본·방송 맛집 | 카카오 로컬 API, 크롤링, 공공데이터 |
| 2-1 | 🎤 **K-Culture 성지순례** | MV/드라마/영화 촬영지, SNS 핫플, 팬덤 방문지 | 크롤링, 커뮤니티, 위키 |
| 2-2 | 🏛️ **문화·관광지** | 국립중앙박물관(서울·경주 등), 유명 관광지 | TourAPI(15,905건 보유), 공공데이터 |
| 2-3 | 🌿 **자연·둘레길** | 전국 둘레길, 스탬프 코스, 자연 관광 경로 | 공공데이터(두루누비), 산림청 |
| 3 | 💰 **일정별 소비 예측** | 당일~3박4일 코스별 예상 비용 | AI Hub 여행로그(보유), TabNet v2 |

### 이동수단
- **1차 MVP**: 걷기 + 자전거 (기존 자산 활용)
- **2차 MVP**: 대중교통 (지하철/버스 API)
- **3차 MVP**: 자동차/기차 (네이버 길찾기 API, 코레일 API)

---

## 2. 기존 자산 재활용 현황

### ✅ 그대로 재활용 가능

| 자산 | 파일 | 활용 방법 |
|------|------|----------|
| 전국 관광 POI 15,905건 | `tour_poi.csv` | 문화·관광지 카테고리 기본 DB |
| 소비 예측 TabNet v2 | `consume_regressor_v2.zip` | 일정별 소비 예측 (R²=0.1277) |
| AI Hub 수도권 여행로그 | `data/ai-hub/` 4개 테이블 | 소비 모델·POI 추천 학습 데이터 |
| Co-occurrence POI 추천 | `poi_cooccurrence_v2.pkl` | 관광지 추천 (Recall@5=0.1372) |
| POI 매력도 TabNet | `attraction_regressor.zip` | 관광지 점수 (R²=0.0662) |
| 카카오 로컬 API 스크립트 | `step3_food_collect.py` | 맛집/카페 POI 수집 |
| 네이버 DataLab 스크립트 | `step4_naver_trend.py` | SNS 인기도 수집 |
| 날씨 LSTM + KMA API | `weather_lstm.pt` + `weather_kma.py` | 날씨 예보 (Acc=73%) |
| FastAPI 서버 코드 | `fastapi_server.py` (27KB) | API 서버 뼈대 |

### ⚠️ 수정 후 재활용

| 자산 | 변경 내용 |
|------|----------|
| 안전 RF 모델 (LFS 포인터) | `build_safety_model.py` 재실행 필요 |
| 경로 그래프 (LFS 포인터) | 전국 확대 + 도보 그래프 추가 |
| Streamlit 앱 | UI 전면 개편 (카테고리별 탭 재구성) |

### ❌ 불필요 / 보류

| 자산 | 이유 |
|------|------|
| GRU 방문지 시퀀스 | 이미 폐기 결정 |
| Phase 7 셀카 패션 추천 | 프로젝트 방향 불일치 |
| 자전거 전용 프로파일 (BMI 등) | 2차 MVP로 이동 |

---

## 3. MVP 로드맵

### 1차 MVP — "지도 기반 관광 추천" (4주)

```
[Week 1] 데이터 수집 + 전처리
  - 전국 맛집 POI 수집 (카카오 로컬 API)
  - K-Culture 촬영지/성지 데이터 수집 (크롤링 + 수동 큐레이션)
  - 둘레길 데이터 수집 (두루누비 공공데이터)
  - 전국 자전거도로 데이터 재수집 (data.go.kr)

[Week 2] 모델링 + 백엔드
  - 안전 모델 재학습 (전국 확대)
  - 소비 예측 모델 전국 데이터 보강
  - Ollama 로컬 LLM 연동 (여행 코스 생성)
  - FastAPI 엔드포인트 재설계

[Week 3] 프론트엔드 + 지도
  - React + 네이버 지도 (or Leaflet) 연동
  - 카테고리별 필터 UI (맛집/K-Culture/관광지/둘레길)
  - 일정별 코스 생성 UI (당일~3박4일)
  - 다국어 지원 (한/영)

[Week 4] 통합 + 배포
  - RAG 파이프라인 구축 (POI 정보 + LLM)
  - Docker Compose 구성
  - Vercel(React) + Railway(FastAPI) 배포
  - 모델 경량화 + HuggingFace 업로드
```

### 2차 MVP — "이동수단 확장 + 개인화" (이후)
- 대중교통 경로 (TAGO API)
- 자동차/기차 경로 (네이버 길찾기, 코레일)
- 사용자 프로파일 기반 개인화
- GraphRAG 확대

### 3차 MVP — "소셜 + 실시간" (이후)
- 사용자 리뷰/평점 시스템
- 실시간 혼잡도 연동
- MCP (Model Context Protocol) 통합
- 다국어 챗봇 (일본어/중국어)

---

## 4. 시스템 아키텍처

```
[사용자] ← HTTPS →
[React Frontend — Vercel]
  - 네이버 지도 / Leaflet 지도 시각화
  - 카테고리별 POI 필터 (맛집/K-Culture/관광/둘레길)
  - 일정 생성기 (당일~3박4일)
  - AI 챗봇 (여행 비서)
  - 다국어 (한/영)
       ↓ REST API
[FastAPI ML Server — Railway]
  - /poi/search         ← POI 검색 (카테고리+지역+반경)
  - /poi/recommend      ← Co-occurrence 추천
  - /course/generate    ← 일정별 코스 생성 (LLM 오케스트레이터)
  - /predict/consume    ← 소비 예측 (TabNet v2)
  - /weather            ← 날씨 예보 (LSTM + KMA)
  - /chat               ← RAG 챗봇 (Ollama + LangChain)
       ↓
[Ollama LLM — 로컬 or Groq API]
  - 코스 생성 오케스트레이션
  - 자연어 질의 → POI 검색 변환
  - RAG: POI DB + 벡터 임베딩
       ↓
[PostgreSQL + PostGIS — Neon]
  - poi (전국 POI 통합 DB)
  - course_template (일정별 코스 템플릿)
  - road_segment (자전거도로/둘레길)
  - user_profile (2차 MVP)
```

---

## 5. 데이터 수집 계획

### 5-1. 맛집 데이터 (카테고리 1)

| 소스 | 방법 | 예상 건수 | 상태 |
|------|------|----------|------|
| 카카오 로컬 API (FD6/CE7) | REST API (300,000콜/일) | ~50,000건 | ✅ 스크립트 보유 |
| 블루리본 서베이 | 크롤링 (robots.txt 확인) | ~3,000건 | [ ] |
| 유튜브 또간집 | 영상 설명 파싱 + 가게명 추출 | ~500건 | [ ] |
| 공공데이터 맛집 인증 | data.go.kr | 지역별 상이 | [ ] |
| 방송 맛집 (백종원 등) | 위키/나무위키 파싱 | ~1,000건 | [ ] |

### 5-2. K-Culture 성지순례 (카테고리 2-1)

| 소스 | 방법 | 예상 건수 |
|------|------|----------|
| 한국관광공사 촬영지 DB | TourAPI `contentTypeId=25(여행코스)` | ~2,000건 |
| K-Drama 촬영지 위키 | 크롤링 (드라마별 촬영지 목록) | ~1,000건 |
| K-Pop MV 촬영지 | 팬위키/나무위키 파싱 | ~300건 |
| 팬카페/위버스 인기 장소 | 수동 큐레이션 + 크롤링 | ~200건 |

### 5-3. 문화·관광지 (카테고리 2-2)

| 소스 | 방법 | 상태 |
|------|------|------|
| 한국관광공사 TourAPI | 전국 15,905건 | ✅ 보유 (`tour_poi.csv`) |
| 국립중앙박물관 API | 공공데이터 | [ ] |
| 문화재청 국가문화유산 | 공공데이터 | [ ] |

### 5-4. 자연·둘레길 (카테고리 2-3)

| 소스 | 방법 | 예상 건수 |
|------|------|----------|
| 두루누비 (durunubi.kr) | 공공데이터 API / 크롤링 | ~800코스 |
| 산림청 숲길 | 공공데이터 | ~500코스 |
| 한국관광공사 도보코스 | TourAPI | 보유 데이터에서 필터 |
| 전국 자전거도로 | data.go.kr (20,262행 보유) | ✅ 보유 |

### 5-5. 소비·여행 데이터 (카테고리 3)

| 소스 | 상태 |
|------|------|
| AI Hub 수도권 여행로그 2023 | ✅ 보유 (2,560 여행, 11,739 소비) |
| AI Hub 전국 여행로그 | [ ] 신청 필요 |

---

## 6. 모델링 계획

### 6-1. 기존 모델 유지 (재학습)

| 모델 | 현재 성능 | 변경 사항 |
|------|-----------|----------|
| 안전 RF | R²=0.9539 (LFS 포인터) | **재학습 필요** (`build_safety_model.py` 실행) |
| 소비 TabNet v2 | R²=0.1277 | 전국 AI Hub 데이터 확보 후 v3 |
| 날씨 LSTM | Acc=73.28% | 유지 (전국 관측소 확대 예정) |
| Co-occurrence POI | Recall@5=0.1372 | 전국 데이터로 v3 |

### 6-2. 신규 모델

| 모델 | 목적 | 기술 | 우선순위 |
|------|------|------|---------|
| **코스 생성 LLM** | 일정별 최적 코스 생성 | Ollama (llama3/gemma2) + RAG | 1순위 |
| **POI 카테고리 분류** | 맛집/K-Culture/관광/자연 분류 | BERT or 규칙 기반 | 1순위 |
| **다국어 임베딩** | 한/영 POI 검색 | multilingual-e5-small | 2순위 |
| **리뷰 감성분석** | POI 품질 점수 | KR-FinBERT (zero-shot) | 2순위 |
| **코스 소비 예측** | 일정별 총 예상 비용 | TabNet v2 확장 | 1순위 |

### 6-3. LLM 전략 (저비용 원칙)

```
[우선순위 1] Ollama 로컬 실행 (무료)
  - llama3.1:8b 또는 gemma2:9b
  - RAG: ChromaDB + POI 벡터 임베딩
  - 코스 생성, 자연어 질의 처리

[우선순위 2] Groq API (무료 tier)
  - 배포 환경에서 로컬 GPU 없을 때
  - llama-3.1-70b-versatile (무료 30 req/min)

[우선순위 3] HuggingFace Inference API
  - 경량 모델 배포 (Spaces)
  - 임베딩 모델 호스팅

[우선순위 4] OpenAI API
  - Groq 장애 시 fallback
  - gpt-4o-mini (저비용)
```

### 6-4. RAG → GraphRAG → MCP 확대 로드맵

```
[1차 MVP] RAG
  - ChromaDB 벡터 스토어
  - POI 정보 + 리뷰 텍스트 임베딩
  - LLM이 검색 결과 기반 코스 생성

[2차 MVP] GraphRAG
  - POI 간 관계 그래프 (거리, 카테고리, 동선)
  - Neo4j or NetworkX 기반
  - "이 관광지 근처 맛집" 같은 관계 질의

[3차 MVP] MCP (Model Context Protocol)
  - 외부 도구 연결 (날씨 API, 지도 API, 교통 API)
  - LLM이 도구를 직접 호출하여 실시간 정보 활용
```

---

## 7. 프론트엔드 설계

### 메인 페이지 구성

```
┌──────────────────────────────────────────┐
│  🇰🇷 K-Ride 2.0  [한국어|English]  [로그인] │
├──────────────────────────────────────────┤
│  [🍽️맛집] [🎤K-Culture] [🏛️관광] [🌿둘레길]  │  ← 카테고리 탭
├────────────┬─────────────────────────────┤
│ 필터 사이드바 │          지도 영역           │
│ - 지역 선택   │  (네이버 지도 or Leaflet)   │
│ - 일정 선택   │   POI 마커 + 경로 오버레이  │
│ - 이동수단    │                             │
│ - AI 추천     │                             │
├────────────┴─────────────────────────────┤
│  📋 추천 코스 카드  |  💰 예상 비용  |  🌤️ 날씨 │
└──────────────────────────────────────────┘
```

### AI 챗봇 패널

```
💬 "BTS 뮤비 촬영지 돌면서 맛집도 가고 싶어요. 1박2일로!"

🤖 추천 코스:
  Day 1: 용산 하이브 → 성수동 MV 촬영지 → 을지로 맛집 → 홍대
  Day 2: 경복궁 → 북촌 한옥마을 → 광장시장

  💰 예상 비용: ₩185,000 (숙박 포함)
  🌤️ 내일 날씨: 맑음 22°C
```

---

## 8. DB 스키마 (PostgreSQL + PostGIS)

```sql
-- 통합 POI 테이블
CREATE TABLE poi (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    name_en         VARCHAR(200),          -- 영문명 (다국어)
    category        VARCHAR(30) NOT NULL,  -- food/kculture/tourism/nature/trail
    sub_category    VARCHAR(50),           -- MV촬영지/드라마촬영지/블루리본 등
    sido            VARCHAR(20),
    sigungu         VARCHAR(30),
    address         VARCHAR(200),
    geom            GEOMETRY(POINT, 4326),
    description     TEXT,
    description_en  TEXT,
    source          VARCHAR(50),           -- kakao/tourapi/crawl/manual
    score           DECIMAL(3,2),          -- 0.00~5.00
    visit_count     INTEGER DEFAULT 0,
    image_url       VARCHAR(500),
    tags            TEXT[],                -- ['BTS','MV','성수동']
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_poi_geom ON poi USING GIST(geom);
CREATE INDEX idx_poi_category ON poi(category);
CREATE INDEX idx_poi_tags ON poi USING GIN(tags);

-- 코스 템플릿
CREATE TABLE course_template (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(200),
    title_en        VARCHAR(200),
    duration_days   SMALLINT,              -- 1=당일, 2=1박2일 ...
    category        VARCHAR(30),
    transport       VARCHAR(20),           -- walk/bike/public/car
    estimated_cost  INTEGER,               -- 원
    poi_ids         INTEGER[],             -- 방문 POI 순서
    route_geom      GEOMETRY(LINESTRING, 4326),
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 둘레길/자전거도로
CREATE TABLE trail (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200),
    trail_type      VARCHAR(20),           -- dullegil/bike_road/forest
    sido            VARCHAR(20),
    length_km       DECIMAL(8,3),
    difficulty      SMALLINT,              -- 1=쉬움, 2=보통, 3=어려움
    has_stamp       BOOLEAN DEFAULT FALSE,
    geom            GEOMETRY(LINESTRING, 4326),
    source          VARCHAR(50)
);
CREATE INDEX idx_trail_geom ON trail USING GIST(geom);
```

---

## 9. 배포 전략 (저비용)

| 서비스 | 기술 | 배포처 | 비용 |
|--------|------|--------|------|
| Frontend | React + Vite | Vercel | 무료 |
| ML API | FastAPI | Railway (500h/월) | 무료 |
| LLM | Ollama | 로컬 or Groq API | 무료 |
| DB | PostgreSQL + PostGIS | Neon (500MB) | 무료 |
| 벡터 DB | ChromaDB | FastAPI 내장 | 무료 |
| ML 모델 파일 | pkl/pt/zip | HuggingFace Hub | 무료 |
| GPU 학습 | Colab / Kaggle | Google Drive 연동 | 무료 |
| 도메인 | — | Vercel 기본 도메인 | 무료 |

### 모델 경량화 전략

```
[로컬 학습] 원천 모델 (full precision)
  ↓ 양자화 / ONNX 변환
[HuggingFace / Roboflow / Kaggle 업로드] 경량 모델
  ↓ 런타임 다운로드
[Railway / Render] FastAPI 서버에서 추론
```

---

## 10. 기술 스택 요약

| 레이어 | 기술 |
|--------|------|
| Frontend | React, Vite, TypeScript, Leaflet or Naver Maps |
| Backend | FastAPI, Python 3.11+ |
| ML/DL | scikit-learn, PyTorch, pytorch-tabnet |
| LLM | Ollama (llama3/gemma2), LangChain, Groq API |
| RAG | ChromaDB, sentence-transformers |
| DB | PostgreSQL + PostGIS (Neon) |
| 배포 | Docker Compose, Vercel, Railway |
| 데이터 | 공공데이터포털, TourAPI, 카카오 로컬, AI Hub |
| 버전관리 | Git + Git LFS, HuggingFace Hub |

---

## 11. 환경 변수

```env
# API Keys
KMA_API_KEY=...
KAKAO_REST_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
TOUR_API_KEY=...

# LLM
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=...
OPENAI_API_KEY=...          # fallback

# DB
DATABASE_URL=postgresql://...@...neon.tech/kride

# Deployment
HF_REPO_ID=your-name/kride-models
VERCEL_URL=https://kride.vercel.app
```
