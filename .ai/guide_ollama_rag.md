# K-Ride 초보자 가이드: Ollama + RAG 완전 입문

> **대상**: Ollama, RAG, 벡터DB를 처음 접하는 분
> **목표**: K-Ride 관광지 추천 챗봇을 직접 만들어보기
> **전제**: Python 기본 문법은 알고 있음

---

## 0. 핵심 개념을 먼저 이해하자

### Ollama가 뭔가요?

> **한 줄 요약**: 내 PC에서 ChatGPT 같은 AI를 무료로 실행하는 프로그램

ChatGPT는 OpenAI 서버에 요청을 보내고 돈을 냅니다.
Ollama는 AI 모델을 내 컴퓨터에 설치해서 오프라인으로 무료 실행합니다.

```
ChatGPT 방식:  내 PC → 인터넷 → OpenAI 서버 → 응답  (유료, 느림)
Ollama 방식:   내 PC → 내 PC 안의 AI → 응답          (무료, 빠름)
```

### RAG가 뭔가요?

> **한 줄 요약**: AI가 내 데이터를 참고해서 답변하게 만드는 기술

일반 AI(ChatGPT)의 문제:
- K-Ride의 96만건 POI 데이터를 모름
- 경복궁 근처 맛집을 물으면 엉뚱한 답을 할 수 있음

RAG를 쓰면:
- AI가 답변하기 전에 우리 DB에서 관련 자료를 먼저 검색
- 검색된 자료를 참고해서 정확한 답변 생성

```
[RAG 없이]
사용자: "BTS 뮤비 촬영지 근처 맛집 알려줘"
AI: "음... 아마도 홍대에 있는 [가상의 식당]이요" ← 틀릴 수 있음

[RAG 있으면]
사용자: "BTS 뮤비 촬영지 근처 맛집 알려줘"
         ↓ 먼저 DB 검색
DB에서 찾음: [연남동 한식당A, 서교동 카페B, ...]
AI: "검색 결과를 보면 연남동 한식당A가 가장 가깝습니다" ← 정확함
```

### 전체 흐름 한눈에 보기

```
[K-Ride RAG 시스템]

1. 준비 단계 (한 번만)
   POI DB (96만건) → 임베딩 모델(bge-m3) → 벡터DB(ChromaDB) 저장

2. 서비스 단계 (매 질문마다)
   사용자 질문
      ↓
   임베딩 모델이 질문을 숫자로 변환
      ↓
   벡터DB에서 유사한 POI 검색 (Top 10~50개)
      ↓
   Ollama LLM이 검색 결과를 보고 자연어 답변 생성
      ↓
   사용자에게 추천 결과 + 이미지 URL 반환
```

---

## 1. Ollama 설치 및 첫 실행

### 1-1. 설치

```bash
# Windows: 아래 URL에서 설치 파일 다운로드
# https://ollama.com/download

# 설치 확인
ollama --version
```

### 1-2. 첫 번째 모델 받기

```bash
# 한/영/일 지원, 경량 (4.5GB)
ollama pull qwen2.5:7b

# 다운로드 완료 후 바로 대화 가능
ollama run qwen2.5:7b
>>> 서울 1박2일 여행 코스 추천해줘
```

### 1-3. Ollama는 백그라운드 서버로 동작

```bash
# Ollama를 실행하면 내 PC에 서버가 열림
ollama serve
# → http://localhost:11434 로 접근 가능

# Python에서 이 서버에 요청을 보내는 방식으로 사용
```

### 1-4. Python에서 Ollama 연결

```bash
pip install ollama
```

```python
import ollama

# 가장 기본적인 사용법
response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "서울 1박2일 여행 코스 추천해줘"}
    ]
)

print(response["message"]["content"])
# → "서울 1박2일 코스를 추천해드릴게요. 첫날은..."
```

---

## 2. 임베딩이란? (벡터DB의 핵심 원리)

### 2-1. 텍스트를 숫자로 바꾸는 이유

컴퓨터는 "경복궁"과 "고궁"이 비슷한 말인지 모릅니다.
임베딩은 텍스트를 숫자 배열(벡터)로 바꿔서 의미가 비슷하면 숫자도 비슷하게 만듭니다.

```
"경복궁" → [0.23, -0.11, 0.87, 0.45, ...]  ← 1024개 숫자
"고궁"   → [0.21, -0.09, 0.84, 0.43, ...]  ← 비슷한 숫자!
"맛집"   → [-0.55, 0.72, -0.13, 0.91, ...] ← 완전히 다른 숫자
```

이 숫자들의 거리를 계산하면 어떤 POI가 질문과 가장 관련있는지 찾을 수 있습니다.

### 2-2. 임베딩 모델 설치

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# bge-m3: 한/영/일 100개 언어 지원, 570MB
model = SentenceTransformer("BAAI/bge-m3")

# 텍스트를 벡터로 변환
vector = model.encode("경복궁 근처 한식 맛집")
print(vector.shape)  # (1024,) ← 1024개 숫자로 표현됨

# 두 문장의 유사도 계산
from sentence_transformers import util

vec1 = model.encode("경복궁")
vec2 = model.encode("조선시대 궁궐")
vec3 = model.encode("삼겹살 맛집")

print(util.cos_sim(vec1, vec2))  # 0.85 ← 비슷함
print(util.cos_sim(vec1, vec3))  # 0.12 ← 관련 없음
```

---

## 3. ChromaDB 설치 및 POI 저장

### 3-1. ChromaDB란?

> 벡터(숫자 배열)를 저장하고 빠르게 검색하는 데이터베이스

일반 DB (PostgreSQL): `WHERE name LIKE '%경복궁%'` → 글자 정확히 일치해야 함
ChromaDB: "고궁 추천" → 경복궁, 창덕궁, 덕수궁 모두 찾아줌 (의미 기반)

```bash
pip install chromadb
```

### 3-2. POI 데이터를 ChromaDB에 저장하기

```python
import chromadb
from sentence_transformers import SentenceTransformer

# 모델 로드
embedder = SentenceTransformer("BAAI/bge-m3")

# ChromaDB 초기화 (로컬 파일로 저장됨)
client = chromadb.PersistentClient(path="./chroma_db")

# 컬렉션 = 하나의 검색 테이블 개념
collection = client.get_or_create_collection(
    name="kride_poi",
    metadata={"hnsw:space": "cosine"}  # 코사인 유사도로 검색
)

# --- 예시: POI 10개를 저장하는 경우 ---
sample_pois = [
    {
        "id": "1",
        "name": "경복궁",
        "category": "kculture",
        "sido": "서울",
        "address": "서울시 종로구 사직로 161",
        "description": "조선시대 정궁, 역사적 건축물",
        "image_url": "https://example.com/gyeongbokgung.jpg"
    },
    {
        "id": "2",
        "name": "광장시장 육회비빔밥",
        "category": "food",
        "sido": "서울",
        "address": "서울시 종로구 창경궁로 88",
        "description": "전통시장 먹거리, 육회비빔밥 유명",
        "image_url": "https://example.com/gwangjang.jpg"
    },
    # ... 나머지 POI들
]

# 텍스트 생성 + 임베딩 + 저장
texts = []
ids = []
metadatas = []

for poi in sample_pois:
    # 검색에 쓸 텍스트: 이름 + 카테고리 + 지역 + 설명을 합침
    text = f"{poi['name']} {poi['category']} {poi['sido']} {poi['description']}"
    texts.append(text)
    ids.append(poi["id"])
    metadatas.append({
        "name": poi["name"],
        "category": poi["category"],
        "sido": poi["sido"],
        "address": poi["address"],
        "image_url": poi["image_url"]
    })

# 한 번에 임베딩 + 저장
embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"저장 완료: {collection.count()}개 POI")
```

### 3-3. 실제 DB에서 POI 가져와서 저장하기

```python
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def load_poi_from_db_and_index():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    # DB에서 POI 조회 (1000개씩 나눠서 처리 — 메모리 절약)
    cur.execute("""
        SELECT id, name, category, sub_category, sido, address, image_url
        FROM poi
        WHERE image_url IS NOT NULL
        LIMIT 1000
    """)
    rows = cur.fetchall()

    texts, ids, metadatas = [], [], []

    for row in rows:
        id_, name, cat, subcat, sido, addr, img_url = row
        text = f"{name} {cat} {subcat or ''} {sido} {addr or ''}"

        texts.append(text)
        ids.append(str(id_))
        metadatas.append({
            "name": name,
            "category": cat,
            "sido": sido,
            "address": addr or "",
            "image_url": img_url or ""
        })

    # 배치 임베딩 (한 번에 처리)
    embeddings = embedder.encode(texts, batch_size=64, normalize_embeddings=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"인덱싱 완료: {len(rows)}개")
    cur.close()
    conn.close()

load_poi_from_db_and_index()
```

---

## 4. 검색 (Retrieval) 구현

### 4-1. 기본 벡터 검색

```python
def search_poi(query: str, top_k: int = 5) -> list[dict]:
    # 질문을 벡터로 변환
    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()

    # ChromaDB에서 유사한 POI 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 결과 정리
    pois = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        pois.append({
            "name": meta["name"],
            "category": meta["category"],
            "sido": meta["sido"],
            "address": meta["address"],
            "image_url": meta["image_url"],
            "similarity": round(1 - dist, 3)  # 거리 → 유사도로 변환
        })

    return pois

# 테스트
results = search_poi("BTS 뮤비 촬영지 근처 맛집")
for r in results:
    print(f"{r['name']} ({r['sido']}) - 유사도: {r['similarity']}")
    print(f"  이미지: {r['image_url']}")
```

---

## 5. RAG 완성: 검색 결과를 LLM에 넣기

이제 검색된 POI를 Ollama LLM에 전달해서 자연스러운 답변을 만듭니다.

```python
import ollama

def kride_recommend(user_query: str, lang: str = "ko") -> dict:

    # Step 1: POI 검색
    pois = search_poi(user_query, top_k=5)

    # Step 2: 검색 결과를 텍스트로 정리 (LLM에 전달할 컨텍스트)
    context = "\n".join([
        f"- {p['name']} ({p['sido']}, {p['category']}): {p['address']}"
        for p in pois
    ])

    # Step 3: 언어별 프롬프트
    prompts = {
        "ko": f"""당신은 한국 여행 전문 AI 가이드입니다.
아래 검색된 장소 목록을 참고해서 사용자 질문에 답해주세요.
검색 목록에 없는 장소는 절대 만들어내지 마세요.

[검색된 장소]
{context}

[사용자 질문]
{user_query}

답변을 짧고 친절하게 해주세요.""",

        "en": f"""You are a Korean travel AI guide.
Answer based only on the search results below. Do not invent places.

[Search Results]
{context}

[User Question]
{user_query}

Keep your answer short and friendly.""",

        "ja": f"""あなたは韓国旅行のAIガイドです。
以下の検索結果だけを参考に回答してください。存在しない場所を作り出さないでください。

[検索結果]
{context}

[ユーザーの質問]
{user_query}

短く親切に答えてください。"""
    }

    # Step 4: Ollama LLM 호출
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompts[lang]}]
    )

    # Step 5: 결과 반환 (텍스트 답변 + POI 목록 + 이미지 URL 포함)
    return {
        "answer": response["message"]["content"],
        "pois": pois,  # 이미지 URL이 포함된 POI 목록
        "query": user_query
    }


# --- 실행 테스트 ---
result = kride_recommend("서울에서 K-Culture 체험할 수 있는 곳 알려줘")

print("=== AI 답변 ===")
print(result["answer"])

print("\n=== 추천 장소 + 이미지 ===")
for poi in result["pois"]:
    print(f"📍 {poi['name']} ({poi['sido']})")
    print(f"   주소: {poi['address']}")
    print(f"   이미지: {poi['image_url']}")
```

---

## 6. 이미지 추가 (Phase 1: URL 저장 방식)

### 6-1. DB에 image_url 컬럼 추가

```python
import psycopg2, os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

# image_url 컬럼 추가 (없으면)
cur.execute("""
    ALTER TABLE poi
    ADD COLUMN IF NOT EXISTS image_url VARCHAR(500),
    ADD COLUMN IF NOT EXISTS image_url_thumb VARCHAR(500);
""")
conn.commit()
cur.close()
conn.close()
print("컬럼 추가 완료")
```

### 6-2. TourAPI 수집 시 이미지 URL 저장

```python
import requests

def collect_tourapi_with_image(area_code: str):
    url = "https://apis.data.go.kr/B551011/KorService1/areaBasedList1"
    params = {
        "serviceKey": os.getenv("TOUR_API_KEY"),
        "MobileOS": "ETC",
        "MobileApp": "KRide",
        "_type": "json",
        "areaCode": area_code,
        "numOfRows": 100,
    }
    res = requests.get(url, params=params)
    items = res.json()["response"]["body"]["items"]["item"]

    for item in items:
        poi = {
            "name": item.get("title", ""),
            "address": item.get("addr1", ""),
            "image_url": item.get("firstimage", ""),        # ← 원본 이미지
            "image_url_thumb": item.get("firstimage2", ""), # ← 썸네일
            "lat": item.get("mapy"),
            "lon": item.get("mapx"),
        }
        # DB에 저장 ...
```

---

## 7. 전체 파일 구조

```
kride-project/
├── chroma_db/                  ← ChromaDB 벡터 저장소 (자동 생성됨)
│
└── src/
    └── rag/
        ├── indexer.py          ← POI DB → ChromaDB 임베딩 저장
        ├── retriever.py        ← ChromaDB 검색 함수
        ├── recommender.py      ← RAG 완성 (검색 + LLM 답변)
        └── test_rag.py         ← 테스트 실행
```

---

## 8. 실행 순서 요약 (처음 시작할 때)

```bash
# 1. 패키지 설치
pip install ollama chromadb sentence-transformers

# 2. Ollama 모델 다운로드 (한 번만)
ollama pull qwen2.5:7b

# 3. Ollama 서버 실행 (터미널 1개 켜두기)
ollama serve

# 4. POI 인덱싱 (한 번만 — DB 전체를 ChromaDB에 저장)
python src/rag/indexer.py

# 5. 추천 테스트
python src/rag/test_rag.py
```

---

## 9. 자주 하는 실수와 해결법

| 실수 | 원인 | 해결 |
|------|------|------|
| `Connection refused` 에러 | Ollama 서버가 꺼져 있음 | `ollama serve` 먼저 실행 |
| 검색 결과가 엉뚱함 | 인덱싱할 때와 다른 임베딩 모델 사용 | 같은 모델 `bge-m3` 고정 |
| 메모리 부족 | 96만건을 한 번에 임베딩 | `batch_size=64`로 나눠서 처리 |
| LLM이 없는 장소 만들어냄 | 프롬프트에 "만들지 말라"는 지시 없음 | 프롬프트에 명시적 제한 추가 |
| ChromaDB 중복 저장 | 같은 ID를 두 번 add | `collection.upsert()` 사용 |

---

## 10. 다음 단계

```
[지금 할 수 있는 것]
  ✅ Phase 1: Ollama 설치 + 기본 대화
  ✅ Phase 2: bge-m3 임베딩 + ChromaDB POI 저장
  ✅ Phase 3: RAG 파이프라인 완성 (검색 + LLM 답변 + 이미지 URL)

[이후 단계]
  [ ] Reranker 추가 (bge-reranker-v2-m3) — 검색 정확도 향상
  [ ] bge-visualized-m3 — 이미지로 유사 장소 검색
  [ ] Qwen2.5-VL:7b — 이미지 보고 설명 자동 생성
  [ ] GraphRAG — BTS 촬영지→근처 맛집 그래프 탐색
  [ ] Streamlit 웹 UI 연결
```
