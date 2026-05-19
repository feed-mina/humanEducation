"""
chatbot_chain.py — 핵심 RAG 파이프라인 오케스트레이션
====================================================
파이프라인:
  1. Multi-Query (원본 + 3 변형)
  2. Multi-Source Retrieval (PDF + POI 컬렉션 4개)
  3. 중복 제거
  4. Rerank (BGE-reranker-v2-m3)
  5. Context 조립
  6. Groq LLM 응답 생성
"""
from __future__ import annotations

from collections import defaultdict

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from chatbot.config import (
    CHROMA_PATH,
    PDF_COLLECTION,
    POI_COLLECTIONS,
    EMBED_MODEL,
    GROQ_MODEL,
    GROQ_API_KEY,
    RETRIEVE_TOP_K_PDF,
    RETRIEVE_TOP_K_POI,
    RERANK_TOP_K,
    MAX_HISTORY_TURNS,
)
from chatbot.multi_query import generate_query_variants
from chatbot.reranker import Reranker

# ── 싱글턴 ────────────────────────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_chroma: chromadb.ClientAPI | None = None
_groq: Groq | None = None
_reranker: Reranker | None = None

# 세션별 대화 이력 (인메모리)
_sessions: dict[str, list[dict]] = defaultdict(list)


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma


def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# ── 검색 ──────────────────────────────────────────────────────────────────────
def _search_collection(
    collection_name: str,
    query_vec: list[float],
    top_k: int,
    source_type: str,
) -> list[dict]:
    """단일 ChromaDB 컬렉션 검색"""
    try:
        col = _get_chroma().get_collection(collection_name)
    except Exception:
        return []

    res = col.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        passages.append({
            "text": doc,
            "metadata": meta,
            "source_type": source_type,
            "collection": collection_name,
            "chroma_distance": float(dist),
        })
    return passages


def _multi_source_retrieve(queries: list[str]) -> list[dict]:
    """멀티쿼리 × 멀티소스 검색 → 통합"""
    embedder = _get_embedder()
    all_passages: list[dict] = []
    seen_keys: set[str] = set()

    for query in queries:
        vec = embedder.encode(query, normalize_embeddings=True).tolist()

        # PDF 컬렉션
        for p in _search_collection(PDF_COLLECTION, vec, RETRIEVE_TOP_K_PDF, "pdf"):
            key = p["metadata"].get("source_pdf", "") + str(p["metadata"].get("page", "")) + str(p["metadata"].get("chunk_index", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                all_passages.append(p)

        # POI 컬렉션 4개
        for col_name in POI_COLLECTIONS:
            for p in _search_collection(col_name, vec, RETRIEVE_TOP_K_POI, "poi"):
                poi_name = p["metadata"].get("name", p["text"][:50])
                if poi_name not in seen_keys:
                    seen_keys.add(poi_name)
                    all_passages.append(p)

    return all_passages


# ── 응답 생성 ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 한국 여행 전문 AI 챗봇 'K-Ride 가이드'입니다.

규칙:
1. 반드시 제공된 [참고 자료] 내용을 기반으로 답변하세요.
2. 참고 자료에 없는 내용은 "해당 정보는 확인되지 않았습니다"라고 말하세요.
3. 답변은 친절하고 구체적으로, 한국어로 작성하세요.
4. POI(관광지) 정보가 있으면 이름, 주소, 특징을 포함하세요.
5. 출처(PDF 파일명 또는 컬렉션명)를 [출처: ...] 형식으로 답변 끝에 명시하세요."""


def _build_context(passages: list[dict]) -> tuple[str, list[str], list[dict]]:
    """리랭킹된 패시지로 컨텍스트 조립 → (context_str, sources, pois)"""
    context_parts: list[str] = []
    sources: list[str] = set()
    pois: list[dict] = []

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
                    "lat": meta.get("lat"),
                    "lon": meta.get("lon"),
                })

    return "\n\n".join(context_parts), list(sources), pois


def chat(message: str, session_id: str, context: dict | None = None) -> dict:
    """
    챗봇 메인 파이프라인.

    Returns: {"reply": str, "sources": list, "pois": list}
    """
    # 1. Multi-Query
    queries = generate_query_variants(message)

    # 2. Multi-Source Retrieval
    passages = _multi_source_retrieve(queries)

    # 3. Rerank
    if passages:
        reranker = _get_reranker()
        passages = reranker.rerank(message, passages, text_key="text", top_k=RERANK_TOP_K)

    # 4. Context 조립
    context_str, sources, pois = _build_context(passages)

    # 5. 대화 이력 구성
    history = _sessions[session_id][-MAX_HISTORY_TURNS * 2:]  # 최근 N턴

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"""[참고 자료]
{context_str}

[사용자 질문]
{message}""",
    })

    # 6. Groq LLM 응답
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"응답 생성 중 오류가 발생했습니다: {e}"

    # 이력 저장
    _sessions[session_id].append({"role": "user", "content": message})
    _sessions[session_id].append({"role": "assistant", "content": reply})

    # 이력 트리밍
    if len(_sessions[session_id]) > MAX_HISTORY_TURNS * 2:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_TURNS * 2:]

    return {
        "reply": reply,
        "sources": sources,
        "pois": pois,
    }


def reset_session(session_id: str):
    """세션 초기화"""
    _sessions.pop(session_id, None)
