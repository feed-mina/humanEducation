"""
K-Ride RAG 파이프라인 — Groq API + pgvector 버전
LLM: Groq (openai/gpt-oss-120b)
Embed: intfloat/multilingual-e5-small
VectorDB: PostgreSQL + pgvector (874K POI 전체)
추적: MLflow

실행: python scripts/rag_pgvector_pipeline.py
MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import sys
import os
import time

os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

# ──────────────────────────────────────────
# STEP 0 — 환경 확인
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 0 — 환경 확인")
print("=" * 55)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY or GROQ_API_KEY == "여기에_Groq_API_키_입력":
    print("❌ GROQ_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)
print(f"✅ GROQ_API_KEY 확인 완료 (gsk_...{GROQ_API_KEY[-4:]})")

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("❌ DATABASE_URL이 설정되지 않았습니다.")
    sys.exit(1)
print(f"✅ DATABASE_URL 확인 완료")

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("kride_rag_pipeline")
print("✅ MLflow 연결 완료")


# ──────────────────────────────────────────
# STEP 1 — Groq 연결 확인
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 1 — Groq API 연결 확인")
print("=" * 55)

from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-120b"

try:
    t0 = time.time()
    test_resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    ping_latency = round(time.time() - t0, 2)
    print(f"✅ Groq API 연결 성공! ({ping_latency}초)")
    print(f"   모델: {GROQ_MODEL}")
except Exception as e:
    print(f"❌ Groq API 연결 실패: {e}")
    sys.exit(1)


# ──────────────────────────────────────────
# STEP 2 — LLM 기본 대화 테스트
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — Groq LLM 기본 대화 테스트")
print("=" * 55)

queries = [
    ("ko", "서울에서 K-Culture 체험할 수 있는 곳 3곳만 알려줘"),
    ("en", "Recommend 3 K-Culture spots in Seoul"),
    ("ja", "ソウルでKカルチャーを体験できる場所を3つ教えてください"),
]

for lang, q in queries:
    t0 = time.time()
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": q}],
        temperature=0.7,
        max_tokens=256,
    )
    latency = round(time.time() - t0, 2)
    label = {"ko": "한국어", "en": "English", "ja": "日本語"}[lang]
    print(f"\n=== {label} ({latency}초) ===")
    print(resp.choices[0].message.content)


# ──────────────────────────────────────────
# STEP 3~8 — MLflow run
# ──────────────────────────────────────────
with mlflow.start_run(run_name=f"pgvector__{GROQ_MODEL.replace('/', '_')}"):

    EMBED_MODEL = "intfloat/multilingual-e5-small"
    TOP_K = 5

    mlflow.log_params({
        "embed_model":  EMBED_MODEL,
        "llm_model":    GROQ_MODEL,
        "llm_backend":  "groq_api",
        "vector_db":    "pgvector",
        "top_k":        TOP_K,
    })

    # ──────────────────────────────────────────
    # STEP 3 — 임베딩 모델 로드
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3 — 임베딩 모델 로드")
    print("=" * 55)

    from sentence_transformers import SentenceTransformer, util

    print(f"⏳ {EMBED_MODEL} 모델 로딩 중...")
    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL)
    load_time = time.time() - t0
    print(f"✅ 모델 로딩 완료 ({load_time:.1f}초)")

    test_vec = embedder.encode("test")
    vector_dim = test_vec.shape[0]
    print(f"   벡터 차원: {vector_dim}")
    mlflow.log_metric("embed_load_time_sec", round(load_time, 2))
    mlflow.log_metric("vector_dim", vector_dim)

    # 다국어 유사도 테스트
    multilingual = [
        ("ko", "BTS 뮤직비디오 촬영지 서울"),
        ("en", "BTS music video filming location Seoul"),
        ("ja", "BTSのミュージックビデオ撮影地ソウル"),
    ]
    vecs = embedder.encode([t for _, t in multilingual], normalize_embeddings=True)
    pairs = [(0, 1, "ko_en"), (0, 2, "ko_ja"), (1, 2, "en_ja")]
    print("\n=== 다국어 유사도 ===")
    for i, j, key in pairs:
        sim = util.cos_sim(vecs[i], vecs[j]).item()
        print(f"  {multilingual[i][0]} ↔ {multilingual[j][0]}: {sim:.3f}")
        mlflow.log_metric(f"multilingual_sim_{key}", round(sim, 4))


    # ──────────────────────────────────────────
    # STEP 4 — pgvector 연결 + 검색 함수
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4 — pgvector 연결 + 검색 함수 초기화")
    print("=" * 55)

    import psycopg2

    db_conn = psycopg2.connect(DATABASE_URL)
    db_cur  = db_conn.cursor()

    db_cur.execute("SELECT COUNT(*) FROM poi WHERE embedding IS NOT NULL")
    indexed_count = db_cur.fetchone()[0]
    db_cur.execute("SELECT COUNT(*) FROM poi")
    total_count = db_cur.fetchone()[0]
    print(f"✅ DB 연결 성공")
    print(f"   전체 POI: {total_count:,}개")
    print(f"   임베딩 인덱싱 완료: {indexed_count:,}개")
    mlflow.log_metric("db_poi_total",   total_count)
    mlflow.log_metric("db_poi_indexed", indexed_count)

    def to_pgvec(arr) -> str:
        return '[' + ','.join(f'{x:.8f}' for x in arr) + ']'

    def search_poi(query: str, top_k: int = TOP_K) -> list[dict]:
        q_vec = embedder.encode(query, normalize_embeddings=True)
        q_str = to_pgvec(q_vec)
        db_cur.execute("""
            SELECT
                id, name, name_en, category, sub_category,
                sido, sigungu, address, image_url,
                1 - (embedding <=> %s::vector) AS similarity
            FROM poi
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (q_str, q_str, top_k))
        cols = ["id", "name", "name_en", "category", "sub_category",
                "sido", "sigungu", "address", "image_url", "similarity"]
        return [dict(zip(cols, row)) for row in db_cur.fetchall()]


    # ──────────────────────────────────────────
    # STEP 5 — 벡터 검색 테스트
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5 — 벡터 검색 테스트 (전체 POI 874K)")
    print("=" * 55)

    eval_queries = [
        "BTS 팬이 꼭 가야 할 성지",
        "traditional Korean food market",
        "韓国の伝統文化を体験できる場所",
        "자전거 타기 좋은 한강",
        "제주도 자연 경관 유네스코",
    ]

    sim_scores = []
    search_latencies = []

    for query in eval_queries:
        t0 = time.time()
        results = search_poi(query)
        s_latency = round(time.time() - t0, 3)
        search_latencies.append(s_latency)

        if results:
            sim_scores.append(results[0]["similarity"])

        print(f"\n🔍 '{query}'  ({s_latency}초)")
        for r in results[:3]:
            bar = "█" * int(float(r["similarity"]) * 10)
            print(f"  {float(r['similarity']):.3f} {bar}  {r['name']} ({r['sido']} {r['sigungu']})")

    avg_sim     = sum(sim_scores) / len(sim_scores) if sim_scores else 0
    avg_s_lat   = sum(search_latencies) / len(search_latencies)
    mlflow.log_metric("avg_top1_similarity",   round(avg_sim, 4))
    mlflow.log_metric("avg_search_latency_sec", round(avg_s_lat, 4))
    print(f"\n📊 평균 유사도: {avg_sim:.3f}  |  평균 검색 시간: {avg_s_lat:.3f}초")


    # ──────────────────────────────────────────
    # STEP 6 — RAG 추천 테스트 (Groq + pgvector)
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6 — RAG 추천 테스트 (Groq + pgvector)")
    print("=" * 55)

    SYSTEM_PROMPTS = {
        "ko": "당신은 한국 여행 전문 AI 가이드입니다. 반드시 한국어로만 답하세요.",
        "en": "You are a Korean travel AI guide. Always answer in English only.",
        "ja": "あなたは韓国旅行の専門AIガイドです。必ず日本語だけで答えてください。",
    }

    USER_TEMPLATES = {
        "ko": """아래 검색된 장소만 참고해서 답하세요. 목록에 없는 장소는 절대 만들어내지 마세요.

[검색된 장소]
{context}

[사용자 질문]
{query}

친절하고 간결하게 3~5문장으로 답해주세요.""",
        "en": """Answer ONLY based on the search results below. Never invent places not listed.

[Search Results]
{context}

[User Question]
{query}

Answer kindly in 3-5 sentences.""",
        "ja": """以下の検索結果だけを参考に答えてください。リストにない場所は絶対に作り出さないでください。

[検索結果]
{context}

[ユーザーの質問]
{query}

3〜5文で親切に日本語で答えてください。""",
    }

    def kride_recommend(user_query: str, lang: str = "ko", top_k: int = TOP_K) -> dict:
        pois = search_poi(user_query, top_k=top_k)
        context = "\n".join([
            f"{i}. {p['name']} ({p['sido']} {p['sigungu']}, {p['category']}) — {p['address']}"
            for i, p in enumerate(pois, 1)
        ])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ko"])},
            {"role": "user",   "content": USER_TEMPLATES[lang].format(context=context, query=user_query)},
        ]
        t_start = time.time()
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        latency = round(time.time() - t_start, 2)
        return {
            "answer":      resp.choices[0].message.content,
            "pois":        pois,
            "query":       user_query,
            "lang":        lang,
            "latency_sec": latency,
        }

    rag_test_cases = [
        ("ko", "K-Pop 팬인데 서울에서 꼭 가야 할 곳 알려줘"),
        ("en", "I'm a BTS fan. Where should I visit in Seoul?"),
        ("ja", "韓国の伝統文化を感じられる場所に行きたいです"),
    ]

    rag_answers   = []
    total_latency = 0.0

    for lang, query in rag_test_cases:
        result = kride_recommend(query, lang=lang)
        total_latency += result["latency_sec"]

        print(f"\n{'=' * 50}")
        print(f"[{lang.upper()}] {result['query']}")
        print("=" * 50)
        print(result["answer"])
        print(f"⏱  응답 시간: {result['latency_sec']}초")
        print("\n📷 검색된 장소:")
        for poi in result["pois"][:3]:
            print(f"   {poi['name']} ({poi['sido']})  유사도: {float(poi['similarity']):.3f}")

        mlflow.log_metric(f"rag_latency_{lang}_sec", result["latency_sec"])
        rag_answers.append(f"[{lang.upper()}] Q: {query}\nA: {result['answer']}\n")

    avg_latency = round(total_latency / len(rag_test_cases), 2)
    mlflow.log_metric("rag_avg_latency_sec", avg_latency)

    artifact_path = "rag_answers_pgvector.txt"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(f"실험 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM: {GROQ_MODEL} (Groq API)  |  Embed: {EMBED_MODEL}  |  VectorDB: pgvector\n")
        f.write(f"인덱싱 POI: {indexed_count:,}개\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n\n".join(rag_answers))
    mlflow.log_artifact(artifact_path)
    os.remove(artifact_path)


    # ──────────────────────────────────────────
    # STEP 7 — DB 상태 최종 확인
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 7 — DB 상태 최종 확인")
    print("=" * 55)

    db_cur.execute("""
        SELECT schemaname, tablename, indexname
        FROM pg_indexes
        WHERE tablename = 'poi' AND indexname = 'poi_embedding_idx'
    """)
    idx_row = db_cur.fetchone()
    has_index = idx_row is not None
    print(f"  pgvector 인덱스: {'✅ poi_embedding_idx' if has_index else '❌ 없음 (load_pgvector_from_colab.py 실행 필요)'}")
    mlflow.log_metric("has_pgvector_index", int(has_index))

    db_cur.close()
    db_conn.close()


    # ──────────────────────────────────────────
    # STEP 8 — 최종 점검
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 8 — 최종 점검")
    print("=" * 55)

    checks = {
        "groq_api":       True,
        "embed_model":    vector_dim > 0,
        "pgvector_index": has_index,
        "poi_indexed":    indexed_count > 800_000,
        "rag_search":     avg_sim > 0.3,
    }

    all_passed = True
    for name, ok in checks.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
        mlflow.log_metric(f"check_{name}", int(ok))
        if not ok:
            all_passed = False

    mlflow.set_tags({
        "all_checks_passed": str(all_passed),
        "embed_model":       EMBED_MODEL,
        "llm_model":         GROQ_MODEL,
        "llm_backend":       "groq_api",
        "vector_db":         "pgvector",
        "poi_indexed":       str(indexed_count),
    })

    if all_passed:
        print(f"\n🎉 모든 항목 정상! pgvector RAG 시스템 준비 완료.")
        print(f"   검색 대상: {indexed_count:,}개 POI")
        print(f"   평균 RAG 응답 시간: {avg_latency}초")
        print(f"   평균 벡터 검색 시간: {avg_s_lat:.3f}초")
    else:
        print("\n⚠️  일부 항목 실패 — 위 로그 확인 후 재시도")

    run_id = mlflow.active_run().info.run_id
    print(f"\n📊 MLflow Run ID: {run_id}")
    print(f"   UI 확인: mlflow ui --backend-store-uri sqlite:///mlflow.db")
