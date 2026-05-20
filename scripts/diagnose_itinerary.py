"""
diagnose_itinerary.py — FOCUS 빈 일정 원인 진단
=================================================
실행: cd D:/kride-project && python scripts/diagnose_itinerary.py
"""
import os
import sys

# .env 로드
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("FOCUS 일정 생성 파이프라인 진단")
print("=" * 60)

# ── 1. 환경변수 확인 ──
print("\n[1] 환경변수 확인")
env_keys = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
for key in env_keys:
    val = os.environ.get(key, "")
    status = "✅" if val else "❌ 비어있음"
    masked = val[:10] + "..." if len(val) > 10 else val
    print(f"  {key}: {masked} {status}")

# ── 2. Neo4j 연결 테스트 ──
print("\n[2] Neo4j 연결 테스트")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    db = os.environ.get("NEO4J_DATABASE", "") or None
    with driver.session(database=db) as session:
        # 전체 노드 수
        result = session.run("MATCH (n) RETURN count(n) AS cnt")
        total = result.single()["cnt"]
        print(f"  전체 노드 수: {total}")

        # Artist 노드 확인
        result = session.run("MATCH (a:Artist) RETURN count(a) AS cnt")
        artist_cnt = result.single()["cnt"]
        print(f"  Artist 노드: {artist_cnt}개")

        # Artist name 샘플
        result = session.run("MATCH (a:Artist) RETURN a.name AS name LIMIT 10")
        names = [r["name"] for r in result]
        print(f"  Artist name 샘플: {names}")

        # POI 노드 확인
        result = session.run("MATCH (p:POI) RETURN count(p) AS cnt")
        poi_cnt = result.single()["cnt"]
        print(f"  POI 노드: {poi_cnt}개")

        # FILMING_AT 관계 확인
        result = session.run("MATCH ()-[r:FILMING_AT]->() RETURN count(r) AS cnt")
        edge_cnt = result.single()["cnt"]
        print(f"  FILMING_AT 엣지: {edge_cnt}개")

        # BTS 테스트 쿼리
        result = session.run("""
            MATCH (p:POI)-[:FILMING_AT]->(a:Artist)
            WHERE a.name IN ['BTS']
            RETURN count(p) AS cnt
        """)
        bts_cnt = result.single()["cnt"]
        print(f"  BTS 관련 POI: {bts_cnt}개")

        if bts_cnt == 0 and artist_cnt > 0:
            # 실제 아티스트 name 확인 — 매칭 실패일 수 있음
            result = session.run("""
                MATCH (a:Artist)
                WHERE a.name CONTAINS 'BTS' OR a.name CONTAINS '방탄' OR a.name_en CONTAINS 'BTS'
                RETURN a.name AS name, a.name_en AS name_en, a.id AS id
            """)
            records = [dict(r) for r in result]
            if records:
                print(f"  ⚠️ BTS 관련 아티스트 발견 (name 불일치 가능): {records}")
            else:
                print(f"  ❌ BTS 관련 아티스트 없음")

        # 지역 테스트 — 서울
        result = session.run("""
            MATCH (p:POI)
            WHERE p.address CONTAINS '서울'
            RETURN count(p) AS cnt
        """)
        seoul_cnt = result.single()["cnt"]
        print(f"  서울 POI (address CONTAINS): {seoul_cnt}개")

    driver.close()
    print("  ✅ Neo4j 연결 성공")
except Exception as e:
    print(f"  ❌ Neo4j 연결 실패: {e}")

# ── 3. ChromaDB 컬렉션 확인 ──
print("\n[3] ChromaDB 컬렉션 확인")
try:
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.list_collections()
    print(f"  컬렉션 수: {len(collections)}")
    for col in collections:
        count = col.count()
        print(f"  - {col.name}: {count}개 문서")

    # POI 컬렉션 확인
    needed = ["kride_poi_kculture", "kride_poi_food", "kride_poi_nature", "kride_poi_history"]
    existing_names = [col.name for col in collections]
    for name in needed:
        status = "✅ 존재" if name in existing_names else "❌ 없음"
        print(f"  필수 컬렉션 {name}: {status}")
except Exception as e:
    print(f"  ❌ ChromaDB 오류: {e}")

# ── 4. Groq API 테스트 ──
print("\n[4] Groq API 테스트")
try:
    from groq import Groq
    groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = "openai/gpt-oss-120b"
    resp = groq.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'OK' in one word."}],
        max_tokens=10,
    )
    reply = resp.choices[0].message.content.strip()
    print(f"  모델: {model}")
    print(f"  응답: {reply}")
    print(f"  ✅ Groq API 정상")
except Exception as e:
    print(f"  ❌ Groq API 실패: {e}")

# ── 5. Supabase 테스트 ──
print("\n[5] Supabase 테스트")
try:
    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    resp = client.table("nodes").select("id").like("id", "artist_%").limit(5).execute()
    print(f"  artist 노드: {len(resp.data or [])}개 (상위 5개)")
    for row in (resp.data or [])[:3]:
        print(f"    - {row['id']}")
    print(f"  ✅ Supabase 연결 성공")
except Exception as e:
    print(f"  ❌ Supabase 실패: {e}")

# ── 결론 ──
print("\n" + "=" * 60)
print("진단 완료. 위 결과에서 ❌ 항목을 확인하세요.")
print("=" * 60)
