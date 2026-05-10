"""
pgvector 설정 — extension 활성화 + poi 테이블에 embedding 컬럼 추가
실행: python scripts/setup_pgvector.py
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(".env")

print("=" * 55)
print("pgvector 설정")
print("=" * 55)

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
conn.autocommit = True
cur = conn.cursor()

# 1. pgvector extension 활성화
print("\n[1] pgvector extension 활성화...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
row = cur.fetchone()
print(f"✅ vector extension v{row[0]}")

# 2. embedding 컬럼 추가 (이미 있으면 무시)
print("\n[2] embedding 컬럼 추가 (vector(384))...")
cur.execute("ALTER TABLE poi ADD COLUMN IF NOT EXISTS embedding vector(384)")
cur.execute("""
    SELECT column_name, data_type, udt_name
    FROM information_schema.columns
    WHERE table_name = 'poi' AND column_name = 'embedding'
""")
row = cur.fetchone()
print(f"✅ {row[0]} ({row[2]}) 컬럼 확인")

# 3. 현재 상태 확인
print("\n[3] poi 테이블 상태 확인...")
cur.execute("SELECT COUNT(*) FROM poi")
total = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM poi WHERE embedding IS NOT NULL")
with_emb = cur.fetchone()[0]
print(f"   전체 POI: {total:,}개")
print(f"   임베딩 있음: {with_emb:,}개")
print(f"   임베딩 없음: {total - with_emb:,}개 → load_pgvector_from_colab.py 로 채울 예정")

cur.close()
conn.close()

print("\n✅ 완료!")
print("   다음 단계: python scripts/load_pgvector_from_colab.py")
