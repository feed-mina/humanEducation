"""
reranker_comparison.py — MiniLM vs BGE 리랭커 벤치마크
=====================================================
실행:
    cd subproject/NLP && python -m chatbot.reranker_comparison

결과:
    .ai/memo/reranker_comparison.md 에 비교 문서 생성
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from groq import Groq

from chatbot.config import (
    CHROMA_PATH,
    PDF_COLLECTION,
    EMBED_MODEL,
    GROQ_MODEL,
    GROQ_API_KEY,
)

# ── 비교 대상 모델 ────────────────────────────────────────────────────────────
MODELS = {
    "MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BGE-M3": "BAAI/bge-reranker-v2-m3",
}

# ── 테스트 쿼리 (한국어 관광) ──────────────────────────────────────────────────
TEST_QUERIES = [
    "제주도 자연 관광지 추천해주세요",
    "서울에서 K-pop 관련 여행지",
    "부산 해운대 근처 맛집",
    "강원도 겨울 여행 코스",
    "경주 역사 유적지 가볼 만한 곳",
    "전주 한옥마을 체험 프로그램",
    "인천 섬 여행 추천",
    "대구 근대골목 투어",
    "여수 밤바다 관광 코스",
    "속초 설악산 등산 코스",
    "가족 여행으로 좋은 농촌 체험",
    "반려동물과 함께 갈 수 있는 관광지",
    "워케이션 가능한 숙소 추천",
    "유네스코 세계유산 한국",
    "한국 로컬 푸드 트립 추천",
    "걷기 여행 좋은 길 추천",
    "봄 벚꽃 명소 추천",
    "한국 전통 시장 투어",
    "서울 예술 투어 코스",
    "무장애 관광지 추천",
]


def _retrieve_passages(query: str, embedder: SentenceTransformer, collection, top_k: int = 20) -> list[dict]:
    """ChromaDB에서 쿼리 기반 패시지 검색"""
    vec = embedder.encode(query, normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        passages.append({
            "text": doc,
            "metadata": meta,
            "chroma_distance": float(dist),
        })
    return passages


def _llm_judge_relevance(query: str, passages: list[dict], groq_client: Groq) -> float:
    """Groq LLM-as-judge: 상위 3개 결과의 한국어 관련도 (0-2점)"""
    if not passages or not groq_client:
        return 0.0

    top3_texts = "\n---\n".join(p["text"][:300] for p in passages[:3])
    prompt = f"""아래 쿼리와 검색 결과 3개의 관련도를 평가하세요.
각 결과에 0(무관), 1(부분 관련), 2(매우 관련) 점수를 매기고,
평균 점수만 숫자로 응답하세요 (예: 1.33).

쿼리: {query}

검색 결과:
{top3_texts}

평균 점수:"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = resp.choices[0].message.content.strip()
        return float(score_text)
    except Exception:
        return 0.0


def run_comparison():
    """메인 벤치마크 실행"""
    # ChromaDB + 임베딩 준비
    print("[reranker_cmp] ChromaDB 연결...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(PDF_COLLECTION)
    except Exception:
        print(f"[reranker_cmp] '{PDF_COLLECTION}' 컬렉션 없음. pdf_ingest.py를 먼저 실행하세요.")
        return

    print(f"[reranker_cmp] 컬렉션 '{PDF_COLLECTION}' — {collection.count()} docs")
    embedder = SentenceTransformer(EMBED_MODEL)

    groq_client = None
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)

    # 모델 로드
    models: dict[str, CrossEncoder] = {}
    for name, model_id in MODELS.items():
        print(f"[reranker_cmp] 모델 로딩: {name} ({model_id})")
        models[name] = CrossEncoder(model_id)

    results: dict[str, dict] = {name: {"latencies": [], "relevances": [], "top5_ids": []} for name in MODELS}

    # 쿼리별 평가
    for qi, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{qi}/{len(TEST_QUERIES)}] {query}")
        passages = _retrieve_passages(query, embedder, collection, top_k=20)
        if not passages:
            print("  → 패시지 없음, 건너뜀")
            continue

        for name, model in models.items():
            # Latency
            tracemalloc.start()
            t0 = time.perf_counter()
            pairs = [(query, p["text"]) for p in passages]
            scores = model.predict(pairs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _, peak_mb = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 정렬
            scored = sorted(
                zip(passages, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            top5 = [p["metadata"].get("source_pdf", "") + str(p["metadata"].get("page", "")) for p, _ in scored[:5]]

            results[name]["latencies"].append(elapsed_ms)
            results[name]["top5_ids"].append(top5)

            # LLM-as-judge (상위 3개)
            top3_passages = [p for p, _ in scored[:3]]
            if groq_client:
                rel = _llm_judge_relevance(query, top3_passages, groq_client)
                results[name]["relevances"].append(rel)
                time.sleep(1)  # Groq rate limit

            print(f"  {name}: {elapsed_ms:.1f}ms, peak={peak_mb/1024/1024:.1f}MB")

    # 집계
    report_lines = [
        "# 리랭커 비교: MiniLM vs BGE-reranker-v2-m3",
        f"\n테스트 쿼리: {len(TEST_QUERIES)}개 한국어 관광 쿼리",
        f"검색 소스: ChromaDB '{PDF_COLLECTION}' ({collection.count()} chunks)",
        "",
        "## 결과 요약",
        "",
        "| 메트릭 | MiniLM (22M) | BGE-M3 (560M) |",
        "|--------|-------------|---------------|",
    ]

    for name in MODELS:
        lats = results[name]["latencies"]
        results[name]["avg_latency"] = sum(lats) / len(lats) if lats else 0
        rels = results[name]["relevances"]
        results[name]["avg_relevance"] = sum(rels) / len(rels) if rels else 0

    mini_lat = results["MiniLM"]["avg_latency"]
    bge_lat = results["BGE-M3"]["avg_latency"]
    mini_rel = results["MiniLM"]["avg_relevance"]
    bge_rel = results["BGE-M3"]["avg_relevance"]

    report_lines.append(f"| 평균 Latency (ms) | {mini_lat:.1f} | {bge_lat:.1f} |")
    report_lines.append(f"| 한국어 관련도 (0-2) | {mini_rel:.2f} | {bge_rel:.2f} |")

    # Top-5 Overlap (Jaccard)
    overlaps = []
    for i in range(len(TEST_QUERIES)):
        if i < len(results["MiniLM"]["top5_ids"]) and i < len(results["BGE-M3"]["top5_ids"]):
            s1 = set(results["MiniLM"]["top5_ids"][i])
            s2 = set(results["BGE-M3"]["top5_ids"][i])
            if s1 or s2:
                overlaps.append(len(s1 & s2) / len(s1 | s2))
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    report_lines.append(f"| Top-5 Jaccard Overlap | {avg_overlap:.2f} | — |")

    # 결론
    report_lines.extend([
        "",
        "## 결론",
        "",
    ])
    if bge_rel > mini_rel:
        report_lines.append(
            f"**BGE-reranker-v2-m3 채택**: 한국어 관련도 {bge_rel:.2f} > {mini_rel:.2f} (MiniLM). "
            f"레이턴시 차이({bge_lat:.0f}ms vs {mini_lat:.0f}ms)는 챗봇 UX에서 허용 범위."
        )
    else:
        report_lines.append(
            f"**MiniLM 채택**: 한국어 관련도 유사({mini_rel:.2f} vs {bge_rel:.2f})하며 "
            f"레이턴시 {mini_lat:.0f}ms로 BGE({bge_lat:.0f}ms) 대비 빠름."
        )

    report_lines.extend([
        "",
        f"생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    # 파일 저장
    from chatbot.config import PROJECT_ROOT
    memo_dir = os.path.join(str(PROJECT_ROOT), ".ai", "memo")
    os.makedirs(memo_dir, exist_ok=True)
    out_path = os.path.join(memo_dir, "reranker_comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[reranker_cmp] 비교 문서 저장: {out_path}")


if __name__ == "__main__":
    run_comparison()
