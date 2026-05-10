"""
K-Ride RAG 파이프라인 — Groq API 버전
LLM: Groq (llama-3.3-70b-versatile) — 응답 1~2초
Embed: intfloat/multilingual-e5-small
VectorDB: ChromaDB
추적: MLflow

실행: python scripts/rag_groq_pipeline.py
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
    print("   .env 파일에 GROQ_API_KEY=gsk_... 를 추가하세요.")
    print("   발급: https://console.groq.com → API Keys → Create API Key")
    sys.exit(1)

print(f"✅ GROQ_API_KEY 확인 완료 (gsk_...{GROQ_API_KEY[-4:]})")

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
# STEP 2 — LLM 기본 대화 테스트 (RAG 없이)
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — Groq LLM 기본 대화 테스트 (RAG 없이)")
print("=" * 55)

def groq_chat(prompt: str, lang_hint: str = "") -> tuple[str, float]:
    t0 = time.time()
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    latency = round(time.time() - t0, 2)
    return resp.choices[0].message.content, latency

queries = [
    ("ko", "서울에서 K-Culture 체험할 수 있는 곳 3곳만 알려줘"),
    ("en", "Recommend 3 K-Culture spots in Seoul"),
    ("ja", "ソウルでKカルチャーを体験できる場所を3つ教えてください"),
]

for lang, q in queries:
    answer, latency = groq_chat(q)
    label = {"ko": "한국어", "en": "English", "ja": "日本語"}[lang]
    print(f"\n=== {label} ({latency}초) ===")
    print(answer)


# ──────────────────────────────────────────
# STEP 3~8 — MLflow run
# ──────────────────────────────────────────
with mlflow.start_run(run_name=f"groq__{GROQ_MODEL.replace('/', '_')}"):

    EMBED_MODEL = "intfloat/multilingual-e5-small"
    TOP_K = 3
    N_SAMPLE_POIS = 10

    mlflow.log_params({
        "embed_model": EMBED_MODEL,
        "llm_model": GROQ_MODEL,
        "llm_backend": "groq_api",
        "top_k": TOP_K,
        "n_sample_pois": N_SAMPLE_POIS,
        "chroma_space": "cosine",
    })

    # ──────────────────────────────────────────
    # STEP 3 — 임베딩 테스트
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3 — 임베딩 테스트")
    print("=" * 55)

    from sentence_transformers import SentenceTransformer, util

    print(f"⏳ {EMBED_MODEL} 모델 로딩 중...")
    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL)
    load_time = time.time() - t0
    print(f"✅ 모델 로딩 완료 ({load_time:.1f}초)")

    test_vec = embedder.encode("test")
    vector_dim = test_vec.shape[0]
    print(f"벡터 차원: {vector_dim}")
    mlflow.log_metric("embed_load_time_sec", round(load_time, 2))
    mlflow.log_metric("vector_dim", vector_dim)

    sentences = [
        "경복궁",
        "조선시대 궁궐 서울",
        "Gyeongbokgung Palace",
        "景福宮",
        "삼겹살 맛집 홍대",
        "자전거 도로 한강",
    ]
    base = embedder.encode(sentences[0], normalize_embeddings=True)
    others = embedder.encode(sentences[1:], normalize_embeddings=True)

    print(f"\n기준 문장: '{sentences[0]}'\n")
    for sent, vec in zip(sentences[1:], others):
        sim = util.cos_sim(base, vec).item()
        bar = "█" * int(sim * 20)
        print(f"  {sim:.3f} {bar}  '{sent}'")

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
    # STEP 4 — ChromaDB 샘플 POI 저장
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4 — ChromaDB 샘플 POI 저장")
    print("=" * 55)

    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("kride_poi_sample")
    except Exception:
        pass
    collection = client.create_collection(
        name="kride_poi_sample",
        metadata={"hnsw:space": "cosine"}
    )
    print("✅ ChromaDB 컬렉션 생성 완료")

    sample_pois = [
        {"id": "1",  "name": "경복궁",                "category": "kculture", "sido": "서울",
         "address": "서울시 종로구 사직로 161",
         "description": "조선시대 정궁, 역사적 건축물, K-Drama 촬영지",
         "image_url": "https://tong.visitkorea.or.kr/cms/resource/00/2678200_image2_1.jpg"},
        {"id": "2",  "name": "광장시장",               "category": "food",     "sido": "서울",
         "address": "서울시 종로구 창경궁로 88",
         "description": "전통시장, 육회비빔밥, 빈대떡 유명, 외국인 관광객 필수 방문",
         "image_url": "https://example.com/gwangjang.jpg"},
        {"id": "3",  "name": "BTS 방탄소년단 벽화거리", "category": "kculture", "sido": "서울",
         "address": "서울시 용산구 이태원로",
         "description": "BTS 팬 성지순례 명소, K-Pop 문화, 뮤직비디오 촬영지 인근",
         "image_url": "https://example.com/bts_wall.jpg"},
        {"id": "4",  "name": "한강공원 여의도",         "category": "nature",   "sido": "서울",
         "address": "서울시 영등포구 여의동로 330",
         "description": "자전거 도로, 피크닉, 벚꽃 명소, 한강 야경",
         "image_url": "https://example.com/hangang.jpg"},
        {"id": "5",  "name": "홍대 클럽거리",           "category": "kculture", "sido": "서울",
         "address": "서울시 마포구 와우산로",
         "description": "K-Pop 인디음악 발상지, 젊음의 거리, 스트리트 퍼포먼스",
         "image_url": "https://example.com/hongdae.jpg"},
        {"id": "6",  "name": "국립중앙박물관",          "category": "kculture", "sido": "서울",
         "address": "서울시 용산구 서빙고로 137",
         "description": "한국 역사 문화 전시, 외국인 방문 인기 명소, 무료 관람",
         "image_url": "https://example.com/museum.jpg"},
        {"id": "7",  "name": "명동교자",                "category": "food",     "sido": "서울",
         "address": "서울시 중구 명동10길 29",
         "description": "70년 전통 칼국수 맛집, 외국인 관광객 줄서는 집",
         "image_url": "https://example.com/myeongdong.jpg"},
        {"id": "8",  "name": "부산 해운대 해수욕장",    "category": "nature",   "sido": "부산",
         "address": "부산시 해운대구 해운대해변로 264",
         "description": "한국 최대 해수욕장, 여름 여행 필수 코스, K-Drama 촬영지",
         "image_url": "https://example.com/haeundae.jpg"},
        {"id": "9",  "name": "전주 한옥마을",            "category": "kculture", "sido": "전북",
         "address": "전북 전주시 완산구 기린대로 99",
         "description": "한옥 체험, 한국 전통 문화, 비빔밥 발상지, K-Culture 성지",
         "image_url": "https://example.com/jeonju.jpg"},
        {"id": "10", "name": "제주 성산일출봉",          "category": "nature",   "sido": "제주",
         "address": "제주시 서귀포시 성산읍 일출로 284-12",
         "description": "유네스코 세계자연유산, 일출 명소, 제주 여행 필수 코스",
         "image_url": "https://example.com/seongsan.jpg"},
    ]

    texts, ids, metadatas = [], [], []
    for poi in sample_pois:
        text = f"{poi['name']} {poi['category']} {poi['sido']} {poi['description']}"
        texts.append(text)
        ids.append(poi["id"])
        metadatas.append({k: poi[k] for k in ("name", "category", "sido", "address", "image_url")})

    t0 = time.time()
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    embed_time = time.time() - t0
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    mlflow.log_metric("index_poi_count", collection.count())
    mlflow.log_metric("embed_10poi_time_sec", round(embed_time, 3))
    print(f"✅ ChromaDB 저장 완료: {collection.count()}개 POI ({embed_time:.2f}초)")


    # ──────────────────────────────────────────
    # STEP 5 — 벡터 검색 테스트
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5 — 벡터 검색 테스트")
    print("=" * 55)

    def search_poi(query: str, top_k: int = TOP_K) -> list:
        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {**meta, "similarity": round(1 - dist, 3)}
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]

    eval_queries = [
        ("BTS 팬이 꼭 가야 할 성지",        "BTS 방탄소년단 벽화거리"),
        ("traditional Korean food market",  "광장시장"),
        ("韓国の伝統文化を体験できる場所",      "전주 한옥마을"),
        ("자전거 타기 좋은 곳",              "한강공원 여의도"),
    ]

    hit_count = 0
    sim_scores = []

    for query, expected_top1 in eval_queries:
        results = search_poi(query)
        top1_name = results[0]["name"] if results else ""
        top1_sim  = results[0]["similarity"] if results else 0
        is_hit = (top1_name == expected_top1)
        if is_hit:
            hit_count += 1
        sim_scores.append(top1_sim)

        print(f"\n🔍 '{query}'")
        for r in results:
            bar = "█" * int(r["similarity"] * 10)
            hit_marker = " ✅" if r["name"] == expected_top1 else ""
            print(f"  {r['name']} ({r['sido']}) 유사도:{r['similarity']} {bar}{hit_marker}")

    recall_at_1  = hit_count / len(eval_queries)
    avg_top1_sim = sum(sim_scores) / len(sim_scores)

    mlflow.log_metric("recall_at_1",        round(recall_at_1, 4))
    mlflow.log_metric("avg_top1_similarity", round(avg_top1_sim, 4))
    print(f"\n📊 Recall@1: {recall_at_1:.2%}  |  평균 유사도: {avg_top1_sim:.3f}")


    # ──────────────────────────────────────────
    # STEP 6 — RAG 추천 테스트 (Groq)
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6 — RAG 추천 테스트 (Groq)")
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

    def kride_recommend_groq(user_query: str, lang: str = "ko", top_k: int = TOP_K) -> dict:
        pois = search_poi(user_query, top_k=top_k)
        context = "\n".join([
            f"{i}. {p['name']} ({p['sido']}, {p['category']}) — {p['address']}"
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

    rag_answers    = []
    total_latency  = 0.0

    for lang, query in rag_test_cases:
        result = kride_recommend_groq(query, lang=lang)
        total_latency += result["latency_sec"]

        print(f"\n{'=' * 50}")
        print(f"[{lang.upper()}] {result['query']}")
        print("=" * 50)
        print(result["answer"])
        print(f"⏱  응답 시간: {result['latency_sec']}초")
        print("\n📷 이미지 URL:")
        for poi in result["pois"]:
            print(f"   {poi['name']}: {poi['image_url']}")

        mlflow.log_metric(f"rag_latency_{lang}_sec", result["latency_sec"])
        rag_answers.append(f"[{lang.upper()}] Q: {query}\nA: {result['answer']}\n")

    avg_latency = round(total_latency / len(rag_test_cases), 2)
    mlflow.log_metric("rag_avg_latency_sec", avg_latency)

    artifact_path = "rag_answers_groq.txt"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(f"실험 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM: {GROQ_MODEL} (Groq API)  |  Embed: {EMBED_MODEL}  |  top_k: {TOP_K}\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n\n".join(rag_answers))
    mlflow.log_artifact(artifact_path)
    os.remove(artifact_path)


    # ──────────────────────────────────────────
    # STEP 7 — PostgreSQL DB 연결
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 7 — 실제 PostgreSQL DB 연결 테스트")
    print("=" * 55)

    import psycopg2

    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM poi")
        poi_count = cur.fetchone()[0]
        print(f"✅ DB 연결 성공! POI 테이블: {poi_count:,}개")
        mlflow.log_metric("db_poi_count", poi_count)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️  DB 연결 실패 (계속 진행): {e}")
        mlflow.log_metric("db_poi_count", -1)


    # ──────────────────────────────────────────
    # STEP 8 — 최종 점검
    # ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 8 — 최종 점검")
    print("=" * 55)

    checks = {
        "groq_api":        True,
        "embed_model":     vector_dim > 0,
        "chromadb_sample": collection.count() == N_SAMPLE_POIS,
        "rag_search":      recall_at_1 > 0,
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
    })

    if all_passed:
        print(f"\n🎉 모든 항목 정상! Groq RAG 시스템 준비 완료.")
        print(f"   평균 RAG 응답 시간: {avg_latency}초 (이전 Ollama: ~46초)")
    else:
        print("\n⚠️  일부 항목 실패 — 위 로그 확인 후 재시도")

    run_id = mlflow.active_run().info.run_id
    print(f"\n📊 MLflow Run ID: {run_id}")
    print(f"   UI 확인: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"   브라우저: http://localhost:5000")
