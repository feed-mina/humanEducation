"""
kaggle_server.py — K-Ride Slim FastAPI (Kaggle 전용)
=====================================================
TorchServe / Celery 없이 모델 직접 로딩.
ChromaDB PersistentClient (로컬), Neo4j·Supabase·Groq은 클라우드 접속.

[실행] (Kaggle 노트북 내)
  !uvicorn kaggle_server:app --host 0.0.0.0 --port 8000 &

[엔드포인트]
  GET  /api/health
  GET  /api/artists
  GET  /api/regions
  POST /api/recommend/ai
  POST /api/recommend/itinerary
  POST /api/chatbot
  POST /api/chatbot/reset
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import chromadb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

# ══════════════════════════════════════════════════════════════════════════════
# 환경변수
# ══════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "") or None

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Kaggle 데이터셋 경로 (노트북에서 심볼릭 링크 또는 직접 설정)
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/kaggle/input/kride-data/chroma_db")
MODELS_PATH = os.environ.get("MODELS_PATH", "/kaggle/input/kride-data/models")

# ══════════════════════════════════════════════════════════════════════════════
# 모델 직접 로딩 (싱글턴)
# ══════════════════════════════════════════════════════════════════════════════
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[K-Ride] SentenceTransformer 로딩 중...")
        _embedder = SentenceTransformer("intfloat/multilingual-e5-small")
        print("[K-Ride] SentenceTransformer 로딩 완료")
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("[K-Ride] CrossEncoder 로딩 중...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[K-Ride] CrossEncoder 로딩 완료")
    return _reranker


def embed_texts(texts: list[str]) -> list[list[float]]:
    """SentenceTransformer 임베딩 (직접 호출)"""
    vecs = get_embedder().encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def rerank_pairs(query: str, documents: list[str]) -> list[float]:
    """CrossEncoder 리랭킹 (직접 호출)"""
    pairs = [[query, doc] for doc in documents]
    scores = get_reranker().predict(pairs)
    return scores.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB (PersistentClient)
# ══════════════════════════════════════════════════════════════════════════════
_chroma: chromadb.ClientAPI | None = None


def get_chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"[K-Ride] ChromaDB 로드: {CHROMA_PATH}")
    return _chroma


COLLECTION_MAP = {
    "kculture": "kride_poi_kculture",
    "food": "kride_poi_food",
    "nature": "kride_poi_nature",
    "history": "kride_poi_history",
    "shopping": "kride_poi_kculture",
    "rest": "kride_poi_nature",
}

PDF_COLLECTION = "kride_pdf_knowledge"
POI_COLLECTIONS = [
    "kride_poi_kculture",
    "kride_poi_food",
    "kride_poi_nature",
    "kride_poi_history",
]

# ══════════════════════════════════════════════════════════════════════════════
# Groq LLM
# ══════════════════════════════════════════════════════════════════════════════
_groq: Groq | None = None


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j
# ══════════════════════════════════════════════════════════════════════════════
_neo4j_driver = None


def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
    return _neo4j_driver


def neo4j_get_artist_pois(artist_ids: list[str], limit: int = 20) -> list[dict]:
    query = """
    MATCH (p:POI)-[:FILMING_AT]->(a:Artist)
    WHERE a.name IN $artist_ids
    RETURN DISTINCT
        p.id AS poi_id, p.name AS name, p.lat AS lat, p.lon AS lon,
        p.category AS category, p.sido AS sido, p.address AS address,
        p.image_url AS image_url, collect(a.name) AS artists
    LIMIT $limit
    """
    with get_neo4j().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, artist_ids=artist_ids, limit=limit)
        return [dict(r) for r in result]


def neo4j_get_region_pois(region_names: list[str], limit: int = 20) -> list[dict]:
    query = """
    MATCH (p:POI)
    WHERE ANY(r IN $region_names WHERE p.address CONTAINS r)
    RETURN DISTINCT
        p.id AS poi_id, p.name AS name, p.lat AS lat, p.lon AS lon,
        p.category AS category, p.sido AS sido, p.address AS address,
        p.image_url AS image_url,
        [r IN $region_names WHERE p.address CONTAINS r][0] AS region
    LIMIT $limit
    """
    with get_neo4j().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, region_names=region_names, limit=limit)
        return [dict(r) for r in result]


def neo4j_get_regions(limit: int = 20) -> list[dict]:
    query = """
    MATCH (r:Region)
    RETURN r.id AS id, r.name AS name, r.image_url AS image_url,
           r.safety_score AS safety_score
    ORDER BY r.safety_score DESC
    LIMIT $limit
    """
    with get_neo4j().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, limit=limit)
        return [dict(r) for r in result]


# ══════════════════════════════════════════════════════════════════════════════
# Supabase
# ══════════════════════════════════════════════════════════════════════════════
_supabase = None


def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def supabase_get_all_artists() -> list[dict]:
    resp = get_supabase().table("nodes").select("id, metadata").like("id", "artist_%").execute()
    results = []
    for row in (resp.data or []):
        meta = row.get("metadata") or {}
        name = meta.get("name", "")
        if not name:
            continue
        results.append({"id": row["id"], "name": name, "imageUrl": meta.get("image_url", "")})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 앙상블 랭커 (pickle)
# ══════════════════════════════════════════════════════════════════════════════
_ensemble_model = None
_cooccurrence = None


def _load_ensemble():
    global _ensemble_model
    if _ensemble_model is None:
        path = os.path.join(MODELS_PATH, "ensemble_ranker.pkl")
        if os.path.exists(path):
            import pickle
            with open(path, "rb") as f:
                _ensemble_model = pickle.load(f)
            print(f"[K-Ride] 앙상블 모델 로드: {path}")
    return _ensemble_model


def _load_cooccurrence():
    global _cooccurrence
    if _cooccurrence is None:
        path = os.path.join(MODELS_PATH, "poi_cooccurrence_v2.pkl")
        if os.path.exists(path):
            import pickle
            with open(path, "rb") as f:
                _cooccurrence = pickle.load(f)
        else:
            _cooccurrence = {}
    return _cooccurrence


# feature engineering (인라인, 외부 의존성 제거)
PURPOSE_CATEGORY_MAP = {
    "kculture": {"관광지", "문화시설", "K-pop", "한류", "촬영지"},
    "food": {"음식점", "카페", "맛집", "식당"},
    "nature": {"자연", "공원", "산", "바다", "해변", "자연관광지"},
    "history": {"역사", "유적", "문화재", "박물관", "고궁"},
    "shopping": {"쇼핑", "시장", "면세점"},
    "rest": {"숙박", "리조트", "펜션", "호텔"},
}


def compute_features(poi, neo4j_poi_ids, neo4j_artist_counts, chroma_sims,
                     user_artists, user_regions, user_purposes, user_budget):
    pid = poi.get("poi_id") or poi.get("name", "")
    pname = poi.get("name", "")

    neo4j_hit = 1.0 if pid in neo4j_poi_ids or pname in neo4j_poi_ids else 0.0
    neo4j_ac = float(neo4j_artist_counts.get(pid, neo4j_artist_counts.get(pname, 0)))
    chroma_sim = chroma_sims.get(pid, chroma_sims.get(pname, 0.0))

    cooc = _load_cooccurrence()
    jaccard = 0.0
    if cooc and pname:
        scores = [cooc.get((pname, a), cooc.get((a, pname), 0.0)) for a in user_artists]
        jaccard = max(scores) if scores else 0.0

    cat_match = 0.0
    poi_cat = poi.get("category", "")
    for purpose in user_purposes:
        cats = PURPOSE_CATEGORY_MAP.get(purpose, set())
        if any(cat in poi_cat for cat in cats):
            cat_match = 1.0
            break

    region_match = 0.0
    poi_addr = poi.get("address", "") or poi.get("sido", "")
    for region in user_regions:
        if region in poi_addr:
            region_match = 1.0
            break

    budget_fit = 1.0
    avg_cost = poi.get("avg_cost")
    if avg_cost is not None and user_budget:
        bmin = user_budget.get("min", 0)
        bmax = user_budget.get("max", 2_000_000)
        budget_fit = 1.0 if bmin <= avg_cost <= bmax else 0.0

    return np.array([neo4j_hit, neo4j_ac, chroma_sim, jaccard,
                      cat_match, region_match, 0.0, budget_fit], dtype=np.float32)


def ensemble_rank_pois(neo4j_pois, chroma_pois, artists, regions, purposes, budget, top_k=15):
    model_data = _load_ensemble()

    merged = {}
    for p in neo4j_pois + chroma_pois:
        key = p.get("poi_id") or p.get("name", "")
        if key and key not in merged:
            merged[key] = p
    candidates = list(merged.values())
    if not candidates:
        return []
    if model_data is None:
        return candidates[:top_k]

    model = model_data["model"]
    neo4j_ids = set()
    neo4j_ac = {}
    for p in neo4j_pois:
        pid = p.get("poi_id") or p.get("name", "")
        neo4j_ids.add(pid)
        al = p.get("artists", [])
        neo4j_ac[pid] = len(al) if isinstance(al, list) else 1

    chroma_sims = {}
    for p in chroma_pois:
        pid = p.get("poi_id") or p.get("name", "")
        chroma_sims[pid] = p.get("similarity", 0.5)

    X = np.array([
        compute_features(p, neo4j_ids, neo4j_ac, chroma_sims,
                         artists, regions, purposes, budget)
        for p in candidates
    ])
    scores = model.predict(X)
    for poi, score in zip(candidates, scores):
        poi["ensemble_score"] = float(score)
    ranked = sorted(candidates, key=lambda x: x["ensemble_score"], reverse=True)
    return ranked[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# RAG 검색
# ══════════════════════════════════════════════════════════════════════════════
def search_pois_by_purpose(purposes: list[str], query_text: str, top_k: int = 5) -> list[dict]:
    chroma = get_chroma()
    query_vec = embed_texts([query_text])[0]
    results = []
    seen_ids = set()
    for purpose in purposes:
        col_name = COLLECTION_MAP.get(purpose, "kride_poi_kculture")
        try:
            col = chroma.get_collection(col_name)
        except Exception:
            continue
        res = col.query(query_embeddings=[query_vec], n_results=top_k,
                        include=["documents", "metadatas", "distances"])
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            poi_id = meta.get("id", meta.get("name", ""))
            if poi_id not in seen_ids:
                seen_ids.add(poi_id)
                results.append({**meta, "similarity": round(1 - dist, 3), "purpose": purpose})
    return results


def generate_recommendation_text(pois, artists, regions, purposes):
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
            {"role": "user", "content": prompt},
        ],
        temperature=0.7, max_tokens=512,
    )
    return resp.choices[0].message.content


def generate_itinerary(duration, artists, regions, purposes, budget, pois):
    day_count = {"당일치기": 1, "1박2일": 2, "2박3일": 3}.get(duration, 1)
    context = "\n".join([
        f"{i+1}. {p.get('name','?')} ({p.get('sido','?')}, {p.get('category','?')}) — {p.get('address','')}"
        for i, p in enumerate(pois[:15])
    ])
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
      "morning": {{"places": [{{"name": "장소명", "address": "주소", "tip": "한줄 팁"}}]}},
      "afternoon": {{"places": [{{"name": "장소명", "address": "주소", "tip": "한줄 팁"}}]}}
    }}
  ]
}}"""
    resp = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "당신은 한국 여행 일정 전문가입니다. 반드시 제공된 POI 목록에서만 장소를 선택하세요. 응답은 순수 JSON만 출력하세요."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5, max_tokens=1500,
    )
    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"itinerary": [], "raw": raw, "error": "JSON 파싱 실패"}


# ══════════════════════════════════════════════════════════════════════════════
# 챗봇 RAG 파이프라인
# ══════════════════════════════════════════════════════════════════════════════
CHATBOT_SYSTEM_PROMPT = """당신은 한국 여행 전문 AI 챗봇 'K-Ride 가이드'입니다.

규칙:
1. 반드시 제공된 [참고 자료] 내용을 기반으로 답변하세요.
2. 참고 자료에 없는 내용은 "해당 정보는 확인되지 않았습니다"라고 말하세요.
3. 답변은 친절하고 구체적으로, 한국어로 작성하세요.
4. POI(관광지) 정보가 있으면 이름, 주소, 특징을 포함하세요.
5. 출처(PDF 파일명 또는 컬렉션명)를 [출처: ...] 형식으로 답변 끝에 명시하세요."""

_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 10
RETRIEVE_TOP_K_PDF = 10
RETRIEVE_TOP_K_POI = 5
RERANK_TOP_K = 10
MULTI_QUERY_COUNT = 3


def _generate_query_variants(query: str) -> list[str]:
    if not GROQ_API_KEY:
        return [query]
    prompt = f"""아래 사용자 질문에 대해 {MULTI_QUERY_COUNT}개의 다른 표현으로 바꿔주세요.
검색 엔진에서 더 다양한 결과를 얻기 위한 목적입니다.
각 변형을 한 줄씩, 번호 없이 출력하세요.

원본 질문: {query}

변형:"""
    try:
        resp = get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "당신은 한국어 검색 쿼리 변환 전문가입니다. 변형 쿼리만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        variants = [line.strip().lstrip("0123456789.-) ") for line in raw.split("\n") if line.strip()]
        variants = [v for v in variants if v and v != query][:MULTI_QUERY_COUNT]
    except Exception:
        variants = []
    return [query] + variants


def _search_collection(col_name, query_vec, top_k, source_type):
    try:
        col = get_chroma().get_collection(col_name)
    except Exception:
        return []
    res = col.query(query_embeddings=[query_vec], n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    passages = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        passages.append({
            "text": doc, "metadata": meta, "source_type": source_type,
            "collection": col_name, "chroma_distance": float(dist),
        })
    return passages


def _chatbot_retrieve(queries: list[str]) -> list[dict]:
    all_passages = []
    seen_keys = set()
    all_vecs = embed_texts(queries)
    for query, vec in zip(queries, all_vecs):
        for p in _search_collection(PDF_COLLECTION, vec, RETRIEVE_TOP_K_PDF, "pdf"):
            key = p["metadata"].get("source_pdf", "") + str(p["metadata"].get("page", "")) + str(p["metadata"].get("chunk_index", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                all_passages.append(p)
        for col_name in POI_COLLECTIONS:
            for p in _search_collection(col_name, vec, RETRIEVE_TOP_K_POI, "poi"):
                poi_name = p["metadata"].get("name", p["text"][:50])
                if poi_name not in seen_keys:
                    seen_keys.add(poi_name)
                    all_passages.append(p)
    return all_passages


def _chatbot_build_context(passages):
    context_parts = []
    sources = set()
    pois = []
    for i, p in enumerate(passages, 1):
        text = p["text"][:500]
        context_parts.append(f"[{i}] {text}")
        if p["source_type"] == "pdf":
            sources.add(p["metadata"].get("source_pdf", "PDF"))
        else:
            sources.add(p.get("collection", "POI"))
            meta = p.get("metadata", {})
            if meta.get("name"):
                pois.append({
                    "name": meta.get("name", ""),
                    "address": meta.get("address", ""),
                    "category": meta.get("category", ""),
                    "lat": meta.get("lat"), "lon": meta.get("lon"),
                })
    return "\n\n".join(context_parts), list(sources), pois


def chatbot_chat(message: str, session_id: str) -> dict:
    queries = _generate_query_variants(message)
    passages = _chatbot_retrieve(queries)

    if passages:
        docs = [p.get("text", "") for p in passages]
        scores = rerank_pairs(message, docs)
        for passage, score in zip(passages, scores):
            passage["rerank_score"] = float(score)
        passages = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_TOP_K]

    context_str, sources, pois = _chatbot_build_context(passages)

    history = _sessions[session_id][-MAX_HISTORY_TURNS * 2:]
    messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"[참고 자료]\n{context_str}\n\n[사용자 질문]\n{message}"})

    try:
        resp = get_groq().chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.7, max_tokens=1024,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"응답 생성 중 오류가 발생했습니다: {e}"

    _sessions[session_id].append({"role": "user", "content": message})
    _sessions[session_id].append({"role": "assistant", "content": reply})
    if len(_sessions[session_id]) > MAX_HISTORY_TURNS * 2:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_TURNS * 2:]

    return {"reply": reply, "sources": sources, "pois": pois}


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 앱
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="K-Ride Slim API (Kaggle)", version="1.0.0-kaggle")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 요청 스키마 ──────────────────────────────────────────────────────────────

class BudgetSchema(BaseModel):
    min: int = 30000
    max: int = 2000000


class RecommendAIRequest(BaseModel):
    artists: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    purposes: list[str] = Field(default_factory=list)
    budget: BudgetSchema = Field(default_factory=BudgetSchema)


class ItineraryRequest(BaseModel):
    duration: str = "당일치기"
    artists: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    purposes: list[str] = Field(default_factory=list)
    budget: BudgetSchema = Field(default_factory=BudgetSchema)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ResetRequest(BaseModel):
    session_id: str = "default"


_REQUEST_MODEL_TYPES = {"BudgetSchema": BudgetSchema}
RecommendAIRequest.model_rebuild(_types_namespace=_REQUEST_MODEL_TYPES)
ItineraryRequest.model_rebuild(_types_namespace=_REQUEST_MODEL_TYPES)


# ── 아티스트 fallback / 이름 매핑 ────────────────────────────────────────────
FALLBACK_ARTISTS = [
    {"id": "bts", "name": "BTS", "name_ko": "방탄소년단"},
    {"id": "blackpink", "name": "BLACKPINK", "name_ko": "블랙핑크"},
    {"id": "seventeen", "name": "SEVENTEEN", "name_ko": "세븐틴"},
    {"id": "twice", "name": "TWICE", "name_ko": "트와이스"},
    {"id": "exo", "name": "EXO", "name_ko": "엑소"},
    {"id": "iu", "name": "IU", "name_ko": "아이유"},
    {"id": "straykids", "name": "Stray Kids", "name_ko": "스트레이키즈"},
]

ARTIST_NAME_MAP = {}
for _a in FALLBACK_ARTISTS:
    ARTIST_NAME_MAP[_a["name"]] = _a.get("name_ko") or _a["name"]
    ARTIST_NAME_MAP[_a["name"].upper()] = _a.get("name_ko") or _a["name"]
    ARTIST_NAME_MAP[_a["name"].lower()] = _a.get("name_ko") or _a["name"]

FALLBACK_REGIONS = [
    {"id": str(i), "name": name, "imageUrl": None, "safety_score": None}
    for i, name in enumerate(
        ["서울", "경기", "인천", "강원", "충북", "충남", "전북", "전남",
         "경북", "경남", "부산", "대구", "광주", "대전", "울산", "세종", "제주"], 1,
    )
]


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    chroma_ok = False
    try:
        get_chroma().heartbeat()
        chroma_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "runtime": "kaggle",
        "chroma": chroma_ok,
        "groq_key": bool(GROQ_API_KEY),
        "neo4j_uri": bool(NEO4J_URI),
        "supabase_url": bool(SUPABASE_URL),
    }


@app.get("/api/artists")
def list_artists():
    try:
        artists = supabase_get_all_artists()
        return {"artists": artists or FALLBACK_ARTISTS}
    except Exception as e:
        print(f"[K-Ride] Supabase artists fallback: {e}")
        return {"artists": FALLBACK_ARTISTS}


@app.get("/api/regions")
def list_regions():
    try:
        regions = neo4j_get_regions(limit=20)
        return {"regions": regions or FALLBACK_REGIONS}
    except Exception as e:
        print(f"[K-Ride] Neo4j regions fallback: {e}")
        return {"regions": FALLBACK_REGIONS}


@app.post("/api/recommend/ai")
def recommend_ai(req: RecommendAIRequest):
    neo4j_pois = []
    if req.artists:
        search_names = list(set(
            req.artists + [ARTIST_NAME_MAP.get(a, a) for a in req.artists]
        ))
        try:
            neo4j_pois = neo4j_get_artist_pois(search_names, limit=15)
        except Exception as e:
            print(f"[K-Ride] Neo4j artist_pois 실패: {e}")

    chroma_pois = []
    if req.purposes:
        query_text = " ".join(req.purposes + req.regions)
        try:
            chroma_pois = search_pois_by_purpose(req.purposes, query_text, top_k=5)
        except Exception as e:
            print(f"[K-Ride] ChromaDB 실패: {e}")

    merged = {}
    for p in neo4j_pois + chroma_pois:
        key = p.get("poi_id") or p.get("name", "")
        if key not in merged:
            merged[key] = p
    pois = list(merged.values())

    if req.budget:
        pois = [p for p in pois
                if p.get("avg_cost") is None
                or req.budget.min <= p.get("avg_cost", 0) <= req.budget.max]

    rec_text = ""
    if pois:
        try:
            rec_text = generate_recommendation_text(pois, req.artists, req.regions, req.purposes)
        except Exception as e:
            rec_text = f"추천 텍스트 생성 실패: {e}"

    return {"pois": pois[:10], "recommendation_text": rec_text, "count": len(pois)}


@app.post("/api/recommend/itinerary")
def recommend_itinerary(req: ItineraryRequest):
    resolved_artists = [ARTIST_NAME_MAP.get(a, a) for a in req.artists]
    search_artists = list(set(req.artists + resolved_artists))

    artist_pois = []
    if search_artists:
        try:
            artist_pois = neo4j_get_artist_pois(search_artists, limit=10)
        except Exception as e:
            print(f"[K-Ride] Neo4j artist_pois 실패: {e}")

    region_pois = []
    if req.regions:
        try:
            region_pois = neo4j_get_region_pois(req.regions, limit=10)
        except Exception as e:
            print(f"[K-Ride] Neo4j region_pois 실패: {e}")

    chroma_pois = []
    if req.purposes:
        query_text = " ".join(req.purposes + req.regions)
        try:
            chroma_pois = search_pois_by_purpose(req.purposes, query_text, top_k=5)
        except Exception as e:
            print(f"[K-Ride] ChromaDB 실패: {e}")

    neo4j_all = artist_pois + region_pois
    model_data = _load_ensemble()
    if model_data:
        try:
            all_pois = ensemble_rank_pois(
                neo4j_all, chroma_pois, req.artists, req.regions,
                req.purposes, req.budget.dict(), top_k=15,
            )
        except Exception:
            merged = {}
            for p in neo4j_all + chroma_pois:
                key = p.get("poi_id") or p.get("name", "")
                if key not in merged:
                    merged[key] = p
            all_pois = list(merged.values())
    else:
        merged = {}
        for p in neo4j_all + chroma_pois:
            key = p.get("poi_id") or p.get("name", "")
            if key not in merged:
                merged[key] = p
        all_pois = list(merged.values())

    try:
        itinerary_result = generate_itinerary(
            duration=req.duration, artists=req.artists, regions=req.regions,
            purposes=req.purposes, budget=req.budget.dict(), pois=all_pois,
        )
    except Exception as e:
        print(f"[K-Ride] itinerary 생성 실패: {e}")
        itinerary_result = {"itinerary": []}

    if isinstance(itinerary_result, str):
        try:
            itinerary_result = json.loads(itinerary_result)
        except json.JSONDecodeError:
            itinerary_result = {"itinerary": [], "raw": itinerary_result}
    if not isinstance(itinerary_result, dict):
        itinerary_result = {"itinerary": []}
    itinerary_result.setdefault("itinerary", [])

    markers = [
        {"name": p.get("name", ""), "lat": p["lat"], "lon": p["lon"]}
        for p in all_pois if p.get("lat") and p.get("lon")
    ]

    return {
        **itinerary_result,
        "mapData": {"markers": markers},
        "source_pois": all_pois[:15],
    }


@app.post("/api/chatbot")
def chatbot_endpoint(req: ChatRequest):
    result = chatbot_chat(req.message, req.session_id)
    return result


@app.post("/api/chatbot/reset")
def chatbot_reset(req: ResetRequest):
    _sessions.pop(req.session_id, None)
    return {"status": "ok", "session_id": req.session_id}


# ══════════════════════════════════════════════════════════════════════════════
# 직접 실행 시
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
