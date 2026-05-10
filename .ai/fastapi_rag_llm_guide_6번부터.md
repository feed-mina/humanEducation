# FastAPI + RAG + LLM 연동 가이드

K-Ride SDUI 프론트엔드의 온보딩 결과(아티스트/지역/목적/예산)를
실제 AI 추천 결과로 변환하는 백엔드 파이프라인 구현 가이드.

---

## 0. MSA 폴더 구조 (중요)

FastAPI는 Next.js(`kride/`)와 **완전히 다른 언어/런타임**이므로 반드시 분리된 위치에 있어야 합니다.

```
kride-project/                        ← 프로젝트 루트
│
├── subproject/
│   └── SDUI/
│       ├── SDUI-server/              ← Spring Boot (Java, port 8080) — 건드리지 않음
│       └── kride/                    ← Next.js 14 (TypeScript, port 3000) — 건드리지 않음
│           └── src/                  ← TS/TSX 파일만 있어야 함 ← .py 파일 두면 안 됨
│
├── src/
│   └── api/                          ← ★ FastAPI Python 서비스 위치 (port 8000)
│       ├── fastapi_server.py         ← 기존 파일 (6 엔드포인트) + 신규 4개 추가
│       ├── neo4j_client.py           ← 신규 생성 위치 (여기!)
│       ├── rag_client.py             ← 신규 생성 위치 (여기!)
│       └── supabase_client.py        ← 신규 생성 위치 (여기!)
│
├── chroma_db/                        ← ChromaDB 벡터 DB (자동 생성)
├── models/                           ← ML 모델 pkl/pt 파일
└── .env                              ← 환경변수 (모든 서비스 공유)
```

### 각 서비스 역할 요약

| 서비스 | 언어 | 포트 | 역할 |
|--------|------|------|------|
| `subproject/SDUI/SDUI-server/` | Java (Spring Boot) | 8080 | SDUI UI 메타데이터 API |
| `subproject/SDUI/kride/` | TypeScript (Next.js) | 3000 | 프론트엔드 렌더링 |
| `src/api/` | Python (FastAPI) | 8000 | AI 추천 / 경로 / 날씨 |

> **혼동 주의:** `SDUI/kride/src/api/`는 Next.js의 라우트 핸들러 폴더가 아닙니다.
> `kride/src/` 전체가 TypeScript 소스입니다. Python 파일을 여기 두면 Next.js 빌드와 충돌합니다.

### DB 역할 분리 (혼동 금지)

| DB | 위치 | 접근 주체 | 테이블/노드 |
|----|------|----------|------------|
| PostgreSQL | AWS EC2 | Spring Boot 전용 | users, ui_metadata, query_master |
| Neo4j Aura | Cloud | FastAPI 전용 | Region, POI, Artist, 관계 |
| Supabase | Cloud | FastAPI 전용 | artist, poi, artist_poi, 이미지 |
| ChromaDB | FastAPI 서버 로컬 | FastAPI 전용 | POI 임베딩 4컬렉션 |

> Next.js(Vercel)는 DB에 직접 접근하지 않는다.
> Spring Boot가 PostgreSQL을 읽어 ui_metadata를 내려주고,
> FastAPI가 Neo4j/Supabase/ChromaDB를 읽어 AI 추천을 내려준다.

---

## 1. 전체 통신 아키텍처

```
SDUI Frontend (Next.js 14)  ← subproject/SDUI/kride/  port 3000
  /movies    → GET  http://localhost:8000/api/artists
  /latest    → GET  http://localhost:8000/api/regions
  /my-list   → POST http://localhost:8000/api/recommend/ai
  /focus     → POST http://localhost:8000/api/recommend/itinerary
       │
       ▼
FastAPI Server  ← kride-project/src/api/fastapi_server.py  port 8000
  ├─ neo4j_client.py    ← Artist-POI 그래프 (FILMING_AT, IN_REGION)
  ├─ rag_client.py      ← ChromaDB 벡터 검색 + Groq LLM
  ├─ supabase_client.py ← artist/poi 테이블 + 이미지 URL
  └─ PostgreSQL         ← POI 메타데이터, 예산 필터링
       │
       ├─ Neo4j Aura       neo4j+s://e6e5a79c.databases.neo4j.io
       ├─ ChromaDB          ./chroma_db  (multilingual-e5-small 384dim)
       ├─ Groq API          openai/gpt-oss-120b
       └─ Supabase          artist / poi / artist_poi / course_template 테이블
```

---

## 2. 환경변수 추가 (`.env`)

기존 `.env`에 아래 항목을 추가한다.
`GROQ_API_KEY`는 이미 존재하므로 중복 추가 불필요.

```dotenv
# Neo4j Aura  (.ai/Neo4j-e6e5a79c-Created-2026-05-05_kride.txt 참조)
NEO4J_URI=neo4j+s://e6e5a79c.databases.neo4j.io
NEO4J_USERNAME=e6e5a79c
NEO4J_PASSWORD=S31NVMejWZ5HjPVfg8eFdbe5a20oA2CVHF9ssTkjkVU
NEO4J_DATABASE=e6e5a79c

# Supabase (대시보드 → Settings → API)
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<anon-public-key>

# HuggingFace 캐시 (sentence-transformers 모델 저장 경로)
HF_HOME=D:/hf_cache
TRANSFORMERS_CACHE=D:/hf_cache/hub
```

---

## 3. 필수 패키지 설치

```bash
pip install neo4j chromadb sentence-transformers groq supabase
```

| 패키지 | 버전 기준 | 용도 |
|--------|----------|------|
| `neo4j` | ≥5.x | Neo4j Python 드라이버 |
| `chromadb` | ≥0.4 | 벡터 DB 클라이언트 |
| `sentence-transformers` | ≥2.x | multilingual-e5-small 임베딩 |
| `groq` | ≥0.9 | Groq Python SDK |
| `supabase` | ≥2.x | Supabase 클라이언트 |

> `python-dotenv`, `fastapi`, `uvicorn`은 이미 설치되어 있음.

---

## 4. 신규 모듈 구조

```
src/api/
├── fastapi_server.py      ← 기존 6 엔드포인트 + 신규 4 엔드포인트 추가
├── neo4j_client.py        ← (신규) Neo4j 드라이버 + Cypher 헬퍼
├── rag_client.py          ← (신규) ChromaDB + Groq RAG 파이프라인
└── supabase_client.py     ← (신규) artist/poi 테이블 + 이미지 URL
```

---

## 5. `src/api/neo4j_client.py` — 완성 코드

```python
"""neo4j_client.py — Neo4j Aura 드라이버 + Cypher 헬퍼"""
from __future__ import annotations
import os
from neo4j import GraphDatabase

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def get_artist_pois(artist_ids: list[str], limit: int = 20) -> list[dict]:
    """선택 아티스트의 FILMING_AT POI 조회"""
    query = """
    MATCH (a:Artist)-[:FILMING_AT]->(p:POI)
    WHERE a.id IN $artist_ids
    RETURN DISTINCT
        p.id        AS poi_id,
        p.name      AS name,
        p.lat       AS lat,
        p.lon       AS lon,
        p.category  AS category,
        p.sido      AS sido,
        p.address   AS address,
        p.image_url AS image_url,
        collect(a.name) AS artists
    LIMIT $limit
    """
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    with get_driver().session(database=db) as session:
        result = session.run(query, artist_ids=artist_ids, limit=limit)
        return [dict(r) for r in result]


def get_region_pois(region_names: list[str], limit: int = 20) -> list[dict]:
    """선택 지역의 POI 조회 (IN_REGION 관계)"""
    query = """
    MATCH (r:Region)<-[:IN_REGION]-(p:POI)
    WHERE r.name IN $region_names
    RETURN DISTINCT
        p.id        AS poi_id,
        p.name      AS name,
        p.lat       AS lat,
        p.lon       AS lon,
        p.category  AS category,
        p.sido      AS sido,
        p.address   AS address,
        p.image_url AS image_url,
        r.name      AS region
    LIMIT $limit
    """
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    with get_driver().session(database=db) as session:
        result = session.run(query, region_names=region_names, limit=limit)
        return [dict(r) for r in result]


def get_regions(limit: int = 20) -> list[dict]:
    """Region 노드 전체 조회 (안전점수 내림차순)"""
    query = """
    MATCH (r:Region)
    RETURN r.id AS id, r.name AS name, r.image_url AS image_url,
           r.safety_score AS safety_score
    ORDER BY r.safety_score DESC
    LIMIT $limit
    """
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    with get_driver().session(database=db) as session:
        result = session.run(query, limit=limit)
        return [dict(r) for r in result]


def get_region_profile(region_name: str) -> dict:
    """Region 안전점수 + 날씨/소비 프로파일 조회"""
    query = """
    MATCH (r:Region {name: $region_name})
    OPTIONAL MATCH (r)-[:HAS_WEATHER]->(w:WeatherProfile)
    OPTIONAL MATCH (r)-[:HAS_SPEND]->(s:SpendProfile)
    RETURN r.safety_score AS safety_score,
           w.avg_temp     AS avg_temp,
           w.rainy_days   AS rainy_days,
           s.avg_spend    AS avg_spend,
           s.budget_tier  AS budget_tier
    """
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    with get_driver().session(database=db) as session:
        result = session.run(query, region_name=region_name)
        record = result.single()
        return dict(record) if record else {}
```

---

## 6. `src/api/supabase_client.py` — 완성 코드

```python
"""supabase_client.py — Supabase Full DB 클라이언트"""
from __future__ import annotations
import os
from supabase import create_client, Client

_client: Client | None = None

def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
    return _client


def get_all_artists() -> list[dict]:
    """artist 테이블 전체 조회 → {id, name, imageUrl}"""
    resp = get_client().table("artist").select("id, name, image_url").execute()
    return [
        {"id": row["id"], "name": row["name"], "imageUrl": row["image_url"]}
        for row in (resp.data or [])
    ]


def get_poi_details(poi_ids: list[str]) -> list[dict]:
    """poi 테이블에서 상세 정보 + 이미지 URL 조회"""
    resp = (
        get_client()
        .table("poi")
        .select("id, name, address, lat, lon, category, image_url, avg_cost")
        .in_("id", poi_ids)
        .execute()
    )
    return resp.data or []


def get_artist_poi_map(artist_ids: list[str]) -> list[dict]:
    """artist_poi 조인 테이블 — 아티스트별 촬영지 목록"""
    resp = (
        get_client()
        .table("artist_poi")
        .select("artist_id, poi_id")
        .in_("artist_id", artist_ids)
        .execute()
    )
    return resp.data or []
```

---

## 7. `src/api/rag_client.py` — 완성 코드

```python
"""rag_client.py — ChromaDB 벡터 검색 + Groq LLM"""
from __future__ import annotations
import os
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_MODEL = "intfloat/multilingual-e5-small"
GROQ_MODEL  = "openai/gpt-oss-120b"
CHROMA_PATH = "./chroma_db"

# ── 싱글턴 초기화 ──────────────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_chroma: chromadb.ClientAPI | None = None
_groq: Groq | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq


# ── ChromaDB 컬렉션 이름 (기존 4개) ────────────────────────────────────────
# kride_poi_kculture / kride_poi_food / kride_poi_nature / kride_poi_history
COLLECTION_MAP = {
    "kculture":  "kride_poi_kculture",
    "food":      "kride_poi_food",
    "nature":    "kride_poi_nature",
    "history":   "kride_poi_history",
    "shopping":  "kride_poi_kculture",   # fallback
    "rest":      "kride_poi_nature",     # fallback
}


def search_pois_by_purpose(
    purposes: list[str],
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """purposes 기반 ChromaDB 벡터 검색"""
    embedder = get_embedder()
    chroma   = get_chroma()
    query_vec = embedder.encode(query_text, normalize_embeddings=True).tolist()

    results: list[dict] = []
    seen_ids: set[str] = set()

    for purpose in purposes:
        collection_name = COLLECTION_MAP.get(purpose, "kride_poi_kculture")
        try:
            col = chroma.get_collection(collection_name)
        except Exception:
            continue

        res = col.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            poi_id = meta.get("id", meta.get("name", ""))
            if poi_id not in seen_ids:
                seen_ids.add(poi_id)
                results.append({**meta, "similarity": round(1 - dist, 3), "purpose": purpose})

    return results


def generate_recommendation_text(
    pois: list[dict],
    artists: list[str],
    regions: list[str],
    purposes: list[str],
    lang: str = "ko",
) -> str:
    """RAG 기반 추천 이유 텍스트 생성"""
    context = "\n".join([
        f"- {p.get('name', '')} ({p.get('sido', '')}): {p.get('address', '')}"
        for p in pois[:8]
    ])
    prompt = f"""아래 POI 목록만 참고해서 여행 추천 이유를 3~4문장으로 작성하세요.
목록에 없는 장소는 절대 언급하지 마세요.

아티스트: {', '.join(artists)}
지역: {', '.join(regions)}
여행목적: {', '.join(purposes)}

[POI 목록]
{context}

한국어로 친절하게 작성하세요."""

    resp = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "당신은 한국 여행 전문 AI 가이드입니다."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content


def generate_itinerary(
    duration: str,
    artists: list[str],
    regions: list[str],
    purposes: list[str],
    budget: dict,
    pois: list[dict],
) -> dict:
    """Groq LLM → 일정 JSON 생성

    duration: "당일치기" | "1박2일" | "2박3일"
    반환: {"itinerary": [{"day": 1, "morning": {"places": [...]}, "afternoon": {"places": [...]}}]}
    """
    day_count = {"당일치기": 1, "1박2일": 2, "2박3일": 3}.get(duration, 1)

    context = "\n".join([
        f"{i+1}. {p.get('name','?')} ({p.get('sido','?')}, {p.get('category','?')}) — {p.get('address','')}"
        for i, p in enumerate(pois[:15])
    ])

    system_prompt = (
        "당신은 한국 여행 일정 전문가입니다. "
        "반드시 제공된 POI 목록에서만 장소를 선택하세요. "
        "응답은 순수 JSON만 출력하고 다른 텍스트는 포함하지 마세요."
    )

    user_prompt = f"""아래 조건에 맞는 {day_count}일 여행 일정을 JSON으로 생성하세요.

여행기간: {duration} ({day_count}일)
선택아티스트: {', '.join(artists)}
선택지역: {', '.join(regions)}
여행목적: {', '.join(purposes)}
예산: {budget.get('min', 0):,}원 ~ {budget.get('max', 2000000):,}원

[사용 가능한 POI 목록]
{context}

출력 형식 (JSON만, 설명 없이):
{{
  "itinerary": [
    {{
      "day": 1,
      "morning": {{
        "places": [
          {{"name": "장소명", "address": "주소", "tip": "한줄 팁"}}
        ]
      }},
      "afternoon": {{
        "places": [
          {{"name": "장소명", "address": "주소", "tip": "한줄 팁"}}
        ]
      }}
    }}
  ]
}}
"""

    resp = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=1500,
    )
    import json
    raw = resp.choices[0].message.content.strip()
    # JSON 블록만 추출 (```json ... ``` 래핑 대응)
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"itinerary": [], "raw": raw, "error": "JSON 파싱 실패"}
```

---

## 8. `fastapi_server.py` 수정 사항

기존 `src/api/fastapi_server.py` 상단과 하단에 아래 내용을 추가한다.

### 8-1. import 추가 (파일 상단 `from __future__ import annotations` 바로 아래)

```python
# ── 신규 클라이언트 모듈 ─────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

try:
    from src.api.neo4j_client import get_artist_pois, get_region_pois, get_regions
    from src.api.rag_client import search_pois_by_purpose, generate_recommendation_text, generate_itinerary
    from src.api.supabase_client import get_all_artists, get_poi_details
    HAS_AI = True
except ImportError as _e:
    print(f"[K-Ride] AI 모듈 로드 실패 (pip install neo4j chromadb groq supabase): {_e}")
    HAS_AI = False
```

> 서버를 `src/api/` 디렉터리에서 직접 실행하는 경우 import 경로를 아래처럼 변경:
> ```python
> from neo4j_client import get_artist_pois, ...
> from rag_client import search_pois_by_purpose, ...
> from supabase_client import get_all_artists, ...
> ```

### 8-2. Pydantic 스키마 추가

> **`CourseRequest`가 뭔가요?**
> 기존 `fastapi_server.py`에는 자전거 경로·코스용 스키마 3개가 이미 있습니다:
> - `RecommendRequest` — 반경 내 세그먼트 추천 (lat/lon/radius)
> - `RouteRequest` — A→B 최적 경로 (start_lat/end_lat)
> - `CourseRequest` — 시작점 기반 순환 코스 (start_lat/distance_km) ← 이 클래스 **바로 아래**에 신규 스키마를 추가하라는 의미였습니다.
>
> 신규 `fastapi_server.py`를 새로 만들었다면 기존 3개 클래스 없이 아래 스키마만 추가하면 됩니다.

#### 추가할 Pydantic 스키마 (기존 클래스들 아래 또는 파일 끝에 추가)

```python
class BudgetSchema(BaseModel):
    min: int = 30000
    max: int = 2000000

class RecommendAIRequest(BaseModel):
    artists:  list[str] = []
    regions:  list[str] = []
    purposes: list[str] = []
    budget:   BudgetSchema = BudgetSchema()

class ItineraryRequest(BaseModel):
    duration: str = "당일치기"   # 당일치기 | 1박2일 | 2박3일
    artists:  list[str] = []
    regions:  list[str] = []
    purposes: list[str] = []
    budget:   BudgetSchema = BudgetSchema()
```

### 8-3. 신규 엔드포인트 4개 추가 (파일 맨 아래)

```python
# ══════════════════════════════════════════════════════════════════════════════
# 신규 엔드포인트 — AI 추천 (Neo4j + ChromaDB + Groq)
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# GET /api/artists
# ─────────────────────────────────────────────
@app.get("/api/artists")
def list_artists():
    """아티스트 목록 반환 (Supabase artist 테이블)"""
    if not HAS_AI:
        raise HTTPException(status_code=503, detail="AI 모듈 미설치 (pip install supabase)")
    try:
        return {"artists": get_all_artists()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ─────────────────────────────────────────────
# GET /api/regions
# ─────────────────────────────────────────────
@app.get("/api/regions")
def list_regions():
    """지역 목록 반환 (Neo4j Region 노드, 안전점수 내림차순)"""
    if not HAS_AI:
        raise HTTPException(status_code=503, detail="AI 모듈 미설치 (pip install neo4j)")
    try:
        regions = get_regions(limit=20)
        # Neo4j에 Region 노드가 없으면 하드코딩 fallback
        if not regions:
            regions = [
                {"id": str(i), "name": n, "imageUrl": None, "safety_score": None}
                for i, n in enumerate([
                    "서울", "경기", "인천", "강원", "충북", "충남",
                    "전북", "전남", "경북", "경남", "부산", "대구",
                    "광주", "대전", "울산", "세종", "제주",
                ], 1)
            ]
        return {"regions": regions}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ─────────────────────────────────────────────
# POST /api/recommend/ai
# ─────────────────────────────────────────────
@app.post("/api/recommend/ai")
def recommend_ai(req: RecommendAIRequest):
    """
    온보딩 기반 POI 추천
    파이프라인: Neo4j(아티스트 촬영지) → ChromaDB(목적 유사 POI) → Groq(추천 이유)
    """
    if not HAS_AI:
        raise HTTPException(status_code=503, detail="AI 모듈 미설치")

    # 1. Neo4j — 아티스트 촬영지 POI
    neo4j_pois = []
    if req.artists:
        try:
            neo4j_pois = get_artist_pois(req.artists, limit=15)
        except Exception:
            pass

    # 2. ChromaDB — 목적 기반 유사 POI
    chroma_pois = []
    if req.purposes:
        query_text = " ".join(req.purposes + req.regions)
        try:
            chroma_pois = search_pois_by_purpose(req.purposes, query_text, top_k=5)
        except Exception:
            pass

    # 3. 합산 + 중복 제거
    merged: dict[str, dict] = {}
    for p in neo4j_pois + chroma_pois:
        key = p.get("poi_id") or p.get("name", "")
        if key not in merged:
            merged[key] = p

    pois = list(merged.values())

    # 4. 예산 필터링 (avg_cost 필드 있을 때만)
    if req.budget:
        pois = [
            p for p in pois
            if p.get("avg_cost") is None
            or req.budget.min <= p.get("avg_cost", 0) <= req.budget.max
        ]

    # 5. Groq — 추천 이유 텍스트
    rec_text = ""
    if pois:
        try:
            rec_text = generate_recommendation_text(
                pois, req.artists, req.regions, req.purposes
            )
        except Exception as e:
            rec_text = f"추천 텍스트 생성 실패: {e}"

    return {
        "pois": pois[:10],
        "recommendation_text": rec_text,
        "count": len(pois),
    }


# ─────────────────────────────────────────────
# POST /api/recommend/itinerary
# ─────────────────────────────────────────────
@app.post("/api/recommend/itinerary")
def recommend_itinerary(req: ItineraryRequest):
    """
    AI 일정 생성
    파이프라인: Neo4j(아티스트+지역 POI) → ChromaDB(목적 유사 POI) → Groq(일정 JSON)
    """
    if not HAS_AI:
        raise HTTPException(status_code=503, detail="AI 모듈 미설치")

    # 1. Neo4j — 아티스트 촬영지
    artist_pois = []
    if req.artists:
        try:
            artist_pois = get_artist_pois(req.artists, limit=10)
        except Exception:
            pass

    # 2. Neo4j — 지역 POI
    region_pois = []
    if req.regions:
        try:
            region_pois = get_region_pois(req.regions, limit=10)
        except Exception:
            pass

    # 3. ChromaDB — 목적 기반 POI
    chroma_pois = []
    if req.purposes:
        query_text = " ".join(req.purposes + req.regions)
        try:
            chroma_pois = search_pois_by_purpose(req.purposes, query_text, top_k=5)
        except Exception:
            pass

    # 4. 합산 + 중복 제거
    merged: dict[str, dict] = {}
    for p in artist_pois + region_pois + chroma_pois:
        key = p.get("poi_id") or p.get("name", "")
        if key not in merged:
            merged[key] = p

    all_pois = list(merged.values())

    # 5. Groq — 일정 생성
    try:
        itinerary_result = generate_itinerary(
            duration=req.duration,
            artists=req.artists,
            regions=req.regions,
            purposes=req.purposes,
            budget=req.budget.dict(),
            pois=all_pois,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"일정 생성 실패: {e}")

    # 6. mapData 마커 생성 (좌표 있는 POI만)
    markers = [
        {"name": p.get("name", ""), "lat": p["lat"], "lon": p["lon"]}
        for p in all_pois
        if p.get("lat") and p.get("lon")
    ]

    return {
        **itinerary_result,
        "mapData": {"markers": markers},
        "source_pois": all_pois[:15],
    }
```

---

## 9. 엔드포인트 입출력 요약

### `GET /api/artists`
```json
// Response 200
{
  "artists": [
    {"id": "1", "name": "BTS", "imageUrl": "https://...supabase.co/storage/bts.jpg"},
    {"id": "2", "name": "아이유", "imageUrl": "https://..."}
  ]
}
```

### `GET /api/regions`
```json
// Response 200
{
  "regions": [
    {"id": "1", "name": "서울", "imageUrl": null, "safety_score": 0.87},
    {"id": "2", "name": "부산", "imageUrl": null, "safety_score": 0.82}
  ]
}
```

### `POST /api/recommend/ai`
```json
// Request
{
  "artists": ["BTS", "아이유"],
  "regions": ["서울", "경기"],
  "purposes": ["kculture", "food"],
  "budget": {"min": 50000, "max": 300000}
}

// Response 200
{
  "pois": [
    {"name": "경복궁", "sido": "서울", "address": "...", "lat": 37.58, "lon": 126.97, "image_url": "..."}
  ],
  "recommendation_text": "BTS 팬이라면 용산구 방면 성지순례 코스를...",
  "count": 8
}
```

### `POST /api/recommend/itinerary`
```json
// Request
{
  "duration": "1박2일",
  "artists": ["BTS"],
  "regions": ["서울"],
  "purposes": ["kculture", "food"],
  "budget": {"min": 50000, "max": 500000}
}

// Response 200
{
  "itinerary": [
    {
      "day": 1,
      "morning":   {"places": [{"name": "경복궁", "address": "...", "tip": "개장 직후 방문 추천"}]},
      "afternoon": {"places": [{"name": "광장시장", "address": "...", "tip": "육회비빔밥 필수"}]}
    },
    {
      "day": 2,
      "morning":   {"places": [{"name": "홍대 클럽거리", "address": "...", "tip": "낮에도 볼거리 많음"}]},
      "afternoon": {"places": [{"name": "한강공원 여의도", "address": "...", "tip": "자전거 대여 가능"}]}
    }
  ],
  "mapData": {
    "markers": [
      {"name": "경복궁", "lat": 37.5796, "lon": 126.9770}
    ]
  }
}
```

---

## 10. SDUI 프론트엔드 연동 포인트

### `SDUI/kride/src/app/(afterLogin)/movies/page.tsx`
```typescript
// 하드코딩 ARTIST_LIST 대신 API 호출로 교체
const { data } = useQuery({
  queryKey: ['artists'],
  queryFn: () => fetch(`${process.env.NEXT_PUBLIC_KRIDE_API_BASE}/api/artists`)
                   .then(r => r.json()),
  staleTime: 1000 * 60 * 60,  // 1시간 캐시
});
const artistList = data?.artists ?? ARTIST_LIST;
```

### `SDUI/kride/src/app/(afterLogin)/latest/page.tsx`
```typescript
const { data } = useQuery({
  queryKey: ['regions'],
  queryFn: () => fetch(`${process.env.NEXT_PUBLIC_KRIDE_API_BASE}/api/regions`)
                   .then(r => r.json()),
  staleTime: 1000 * 60 * 60,
});
const regionList = data?.regions ?? REGION_LIST;
```

### `SDUI/kride/src/app/(afterLogin)/focus/page.tsx`
```typescript
// MOCK_ITINERARY 대신 실제 API 호출로 교체
const store = useOnboardingStore();
const { data, isLoading } = useQuery({
  queryKey: ['itinerary', store.duration, store.artists, store.regions],
  queryFn: () => fetch(`${process.env.NEXT_PUBLIC_KRIDE_API_BASE}/api/recommend/itinerary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      duration: store.duration,
      artists:  store.artists,
      regions:  store.regions,
      purposes: store.purposes,
      budget:   store.budget,
    }),
  }).then(r => r.json()),
  enabled: !!store.duration,
});
```

### `.env.local` (SDUI/kride/)
```dotenv
```

---

## 11. 실행 순서 및 테스트

```bash
# 1. 패키지 설치
pip install neo4j chromadb sentence-transformers groq supabase

# 2. .env에 Neo4j + Supabase 환경변수 추가 (섹션 2 참조)

# 3. FastAPI 서버 실행 (kride-project 루트에서)
uvicorn src.api.fastapi_server:app --reload --port 8000

# 4. Swagger UI 테스트
# 브라우저: http://localhost:8000/docs

# 5. 엔드포인트 테스트 (curl)
curl http://localhost:8000/api/artists
curl http://localhost:8000/api/regions
curl -X POST http://localhost:8000/api/recommend/ai \
  -H "Content-Type: application/json" \
  -d '{"artists":["BTS"],"regions":["서울"],"purposes":["kculture"],"budget":{"min":0,"max":500000}}'
curl -X POST http://localhost:8000/api/recommend/itinerary \
  -H "Content-Type: application/json" \
  -d '{"duration":"당일치기","artists":["BTS"],"regions":["서울"],"purposes":["kculture","food"],"budget":{"min":0,"max":200000}}'
```

---

## 12. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `503 AI 모듈 미설치` | import 실패 | `pip install neo4j chromadb groq supabase` |
| Neo4j `ServiceUnavailable` | URI/인증 오류 | `.env`의 `NEO4J_*` 변수 재확인 |
| ChromaDB `Collection not found` | 컬렉션 이름 불일치 | `chroma_db/` 폴더 컬렉션명 확인 후 `COLLECTION_MAP` 수정 |
| Groq `AuthenticationError` | API 키 오류 | `.env`의 `GROQ_API_KEY=gsk_...` 확인 |
| Groq JSON 파싱 실패 | LLM이 마크다운 코드블록 포함 | `rag_client.generate_itinerary()` 내 ` ``` ` 파싱 로직 이미 처리됨 |
| sentence-transformers 모델 다운로드 느림 | 최초 1회 HF Hub 다운로드 | `HF_HOME=D:/hf_cache` 설정으로 캐시 재사용 |
| CORS 오류 (프론트엔드 → FastAPI) | 기본 `allow_origins=["*"]` | 배포 시 Vercel URL로 교체 (`fastapi_server.py` L76) |

---

## 13. Neo4j 데이터 없을 때 fallback 전략

Neo4j Aura에 아직 Artist/POI/Region 노드가 없는 경우:
- `GET /api/artists` → Supabase `artist` 테이블로 대체
- `GET /api/regions` → 하드코딩 17개 광역시도 반환 (이미 내장)
- `POST /api/recommend/ai` → ChromaDB 검색 결과만 반환 (neo4j_pois=[] 허용)
- `POST /api/recommend/itinerary` → ChromaDB POI만으로 Groq 일정 생성

Neo4j 노드 적재는 `notebooks/neo4j_data_loader.ipynb` 참조.
