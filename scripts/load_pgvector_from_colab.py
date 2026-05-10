"""
Colab 임베딩 결과 → PostgreSQL pgvector 로드
실행: python scripts/load_pgvector_from_colab.py

사전 준비:
  - setup_pgvector.py 실행 완료
  - data/poi_embeddings.npy  (Colab에서 다운로드)
  - data/poi_metadata.parquet (Colab에서 다운로드)
"""

import os
import time
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv(".env")

EMB_PATH  = "data/poi_embeddings.npy"
META_PATH = "data/poi_metadata.parquet"
BATCH_SIZE = 1_000

print("=" * 55)
print("Colab 임베딩 → PostgreSQL pgvector 로드")
print("=" * 55)

# 파일 존재 확인
for path in [EMB_PATH, META_PATH]:
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        exit(1)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"✅ {path} ({size_mb:.1f} MB)")

# 데이터 로드
print("\n데이터 로드 중...")
embeddings = np.load(EMB_PATH)
df = pd.read_parquet(META_PATH)

print(f"  임베딩: {embeddings.shape}  (dtype: {embeddings.dtype})")
print(f"  메타데이터: {len(df):,}행")
assert len(df) == embeddings.shape[0], f"행 수 불일치! npy={embeddings.shape[0]}, parquet={len(df)}"

n = len(df)

# DB 연결
print(f"\nPostgreSQL 연결 중...")
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM poi WHERE embedding IS NOT NULL")
already = cur.fetchone()[0]
if already > 0:
    print(f"⚠️  이미 {already:,}개 임베딩이 존재합니다. 전체 덮어씁니다.")

# 임베딩을 pgvector 문자열로 변환하는 함수
def to_pgvec(arr: np.ndarray) -> str:
    return '[' + ','.join(f'{x:.8f}' for x in arr) + ']'

# 배치 업데이트
print(f"\n{n:,}개 임베딩 삽입 시작...")
print(f"  배치 크기: {BATCH_SIZE:,}")
t_start = time.time()

for i in range(0, n, BATCH_SIZE):
    batch_df  = df.iloc[i : i + BATCH_SIZE]
    batch_emb = embeddings[i : i + BATCH_SIZE]

    data = [
        (int(row["id"]), to_pgvec(emb))
        for (_, row), emb in zip(batch_df.iterrows(), batch_emb)
    ]

    execute_values(
        cur,
        """
        UPDATE poi
        SET embedding = data.v::vector
        FROM (VALUES %s) AS data(id, v)
        WHERE poi.id = data.id
        """,
        data,
        template="(%s, %s)"
    )
    conn.commit()

    done = min(i + BATCH_SIZE, n)
    if done % 50_000 < BATCH_SIZE or done == n:
        elapsed = time.time() - t_start
        rate    = done / elapsed
        remain  = (n - done) / rate / 60 if done < n else 0
        print(f"  {done:,}/{n:,} ({done/n*100:.1f}%)  "
              f"속도: {rate:.0f}/초  남은: {remain:.1f}분")

elapsed = time.time() - t_start

# 결과 확인
cur.execute("SELECT COUNT(*) FROM poi WHERE embedding IS NOT NULL")
loaded = cur.fetchone()[0]
print(f"\n✅ 로드 완료!")
print(f"   임베딩 저장 완료: {loaded:,}개  |  소요 시간: {elapsed/60:.1f}분")

# IVFFlat 인덱스 생성 (검색 가속)
print(f"\nIVFFlat 인덱스 생성 중... (수 분 소요)")
lists = max(100, int(loaded ** 0.5))
print(f"  lists = {lists}")
t0 = time.time()
cur.execute(f"""
    CREATE INDEX IF NOT EXISTS poi_embedding_idx
    ON poi USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = {lists})
""")
conn.commit()
idx_time = time.time() - t0
print(f"✅ 인덱스 생성 완료 ({idx_time:.0f}초)")

# 간단 검색 테스트
print("\n=== 검색 테스트 ===")
import sys, os as _os
_os.environ["HF_HOME"] = "D:/hf_cache"
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("intfloat/multilingual-e5-small")

test_queries = [
    "BTS 팬이 꼭 가야 할 성지",
    "traditional Korean food market",
    "자전거 타기 좋은 한강",
]

for q in test_queries:
    q_vec = embedder.encode(q, normalize_embeddings=True)
    q_str = to_pgvec(q_vec)
    cur.execute("""
        SELECT name, sido, sigungu,
               1 - (embedding <=> %s::vector) AS similarity
        FROM poi
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT 3
    """, (q_str, q_str))
    rows = cur.fetchall()
    print(f"\n🔍 '{q}'")
    for name, sido, sigungu, sim in rows:
        print(f"  {sim:.3f}  {name} ({sido} {sigungu})")

cur.close()
conn.close()

print(f"\n🎉 완료! rag_pgvector_pipeline.py 를 실행하세요.")
