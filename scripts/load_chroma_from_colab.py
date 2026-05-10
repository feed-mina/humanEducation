"""
Colab 임베딩 결과 → 로컬 ChromaDB 로드
실행: python scripts/load_chroma_from_colab.py

사전 준비:
  - data/poi_embeddings.npy  (Colab에서 다운로드)
  - data/poi_metadata.parquet (Colab에서 다운로드)
"""

import os
import time
import numpy as np
import pandas as pd
import chromadb

EMB_PATH  = "data/poi_embeddings.npy"
META_PATH = "data/poi_metadata.parquet"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "kride_poi_full"
BATCH_SIZE = 1_000

print("=" * 55)
print("Colab 임베딩 → ChromaDB 로드")
print("=" * 55)

# 파일 존재 확인
for path in [EMB_PATH, META_PATH]:
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        print("   Colab에서 다운로드 후 data/ 폴더에 넣어주세요.")
        exit(1)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"✅ {path} ({size_mb:.1f} MB)")

# 로드
print("\n데이터 로드 중...")
embeddings = np.load(EMB_PATH)
df         = pd.read_parquet(META_PATH).fillna("")

print(f"  임베딩: {embeddings.shape}  (dtype: {embeddings.dtype})")
print(f"  메타데이터: {len(df):,}행")
assert len(df) == embeddings.shape[0], "행 수 불일치!"

# ChromaDB 초기화
print(f"\nChromaDB 초기화: {CHROMA_PATH}")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# 기존 컬렉션 삭제 후 재생성
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"  기존 '{COLLECTION_NAME}' 컬렉션 삭제")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ 컬렉션 '{COLLECTION_NAME}' 생성 완료")

# 배치 삽입
n = len(df)
print(f"\n{n:,}개 POI 삽입 시작...")
t_start = time.time()

for i in range(0, n, BATCH_SIZE):
    batch_df  = df.iloc[i : i + BATCH_SIZE]
    batch_emb = embeddings[i : i + BATCH_SIZE]

    ids        = batch_df["id"].astype(str).tolist()
    emb_list   = batch_emb.tolist()
    metadatas  = batch_df[[
        "name", "name_en", "category", "sub_category",
        "sido", "sigungu", "address", "image_url"
    ]].to_dict("records")
    documents  = [
        f"{r['name']} {r['category']} {r['sido']} {r['sigungu']}".strip()
        for r in metadatas
    ]

    collection.add(
        ids=ids,
        embeddings=emb_list,
        metadatas=metadatas,
        documents=documents,
    )

    done = min(i + BATCH_SIZE, n)
    if done % 50_000 < BATCH_SIZE or done == n:
        elapsed = time.time() - t_start
        rate    = done / elapsed
        remain  = (n - done) / rate / 60 if done < n else 0
        print(f"  {done:,}/{n:,} ({done/n*100:.1f}%)  "
              f"속도: {rate:.0f}/초  남은: {remain:.1f}분")

elapsed = time.time() - t_start
final_count = collection.count()

print(f"\n🎉 ChromaDB 로드 완료!")
print(f"   총 {final_count:,}개  |  소요 시간: {elapsed/60:.1f}분")
print(f"   저장 위치: {CHROMA_PATH}/{COLLECTION_NAME}")

# 간단 검색 테스트
print("\n=== 검색 테스트 ===")
from sentence_transformers import SentenceTransformer
import os as _os
_os.environ["HF_HOME"] = "D:/hf_cache"
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

test_queries = [
    "BTS 팬이 꼭 가야 할 성지",
    "traditional Korean food market",
]
for q in test_queries:
    q_vec   = embedder.encode(q, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        include=["metadatas", "distances"]
    )
    print(f"\n🔍 '{q}'")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        sim = round(1 - dist, 3)
        print(f"  {sim:.3f}  {meta['name']} ({meta['sido']} {meta['sigungu']})")

print(f"\n✅ 완료! rag_groq_pipeline.py에서 컬렉션명을 '{COLLECTION_NAME}'으로 변경 후 사용하세요.")
