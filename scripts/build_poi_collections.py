"""
build_poi_collections.py — Neo4j POI → ChromaDB 4개 컬렉션 인덱싱
==================================================================
실행: cd D:/kride-project && python scripts/build_poi_collections.py

기능:
  Neo4j에 적재된 POI 노드를 category별로 4개 ChromaDB 컬렉션에 인덱싱.
  rag_client.py의 search_pois_by_purpose()가 이 컬렉션을 검색함.

컬렉션:
  kride_poi_kculture — category='kculture' 또는 'kpop'
  kride_poi_food     — category='food'
  kride_poi_nature   — category='nature'
  kride_poi_history  — category='history' 또는 'tourism'
"""
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer

# ── 설정 ──
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 100

# category → collection 매핑
CATEGORY_TO_COLLECTION = {
    "kculture": "kride_poi_kculture",
    "kpop":     "kride_poi_kculture",
    "food":     "kride_poi_food",
    "nature":   "kride_poi_nature",
    "history":  "kride_poi_history",
    "tourism":  "kride_poi_history",
}


def fetch_all_pois() -> list[dict]:
    """Neo4j에서 전체 POI 조회"""
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    db = os.environ.get("NEO4J_DATABASE", "") or None
    query = """
    MATCH (p:POI)
    OPTIONAL MATCH (p)-[:FILMING_AT]->(a:Artist)
    RETURN
        p.id       AS id,
        p.name     AS name,
        p.category AS category,
        p.address  AS address,
        p.sido     AS sido,
        p.lat      AS lat,
        p.lon      AS lon,
        p.image_url AS image_url,
        collect(DISTINCT a.name) AS artists
    """
    with driver.session(database=db) as session:
        result = session.run(query)
        pois = [dict(r) for r in result]
    driver.close()
    print(f"Neo4j에서 POI {len(pois)}건 로드")
    return pois


def build_document_text(poi: dict) -> str:
    """POI → 검색용 텍스트 생성"""
    parts = []
    if poi.get("name"):
        parts.append(poi["name"])
    if poi.get("category"):
        parts.append(f"카테고리: {poi['category']}")
    if poi.get("address"):
        parts.append(poi["address"])
    if poi.get("sido"):
        parts.append(poi["sido"])
    artists = poi.get("artists", [])
    if artists and artists != [None]:
        parts.append(f"아티스트: {', '.join([a for a in artists if a])}")
    return " | ".join(parts) if parts else poi.get("id", "unknown")


def main():
    # 1. Neo4j에서 POI 조회
    pois = fetch_all_pois()
    if not pois:
        print("❌ POI가 없습니다. Neo4j 연결 및 데이터를 확인하세요.")
        sys.exit(1)

    # 2. category별 분류
    categorized: dict[str, list[dict]] = {}
    uncategorized = []
    for poi in pois:
        cat = (poi.get("category") or "").lower().strip()
        col_name = CATEGORY_TO_COLLECTION.get(cat)
        if col_name:
            categorized.setdefault(col_name, []).append(poi)
        else:
            uncategorized.append(poi)

    print(f"\n카테고리별 POI 수:")
    for col_name, col_pois in sorted(categorized.items()):
        print(f"  {col_name}: {len(col_pois)}")
    if uncategorized:
        cats = set((p.get("category") or "none") for p in uncategorized)
        print(f"  미분류: {len(uncategorized)}건 (카테고리: {cats})")
        # 미분류 POI는 kculture에 추가
        categorized.setdefault("kride_poi_kculture", []).extend(uncategorized)
        print(f"  → kride_poi_kculture에 미분류 {len(uncategorized)}건 추가")

    # 3. 임베딩 모델 로드
    print(f"\n임베딩 모델 로드: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # 4. ChromaDB 클라이언트
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    # 5. 각 컬렉션별 인덱싱
    for col_name, col_pois in categorized.items():
        print(f"\n── {col_name} ({len(col_pois)}건) ──")

        # 기존 컬렉션 삭제 후 재생성
        try:
            chroma.delete_collection(col_name)
            print(f"  기존 컬렉션 삭제")
        except Exception:
            pass

        collection = chroma.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 배치 처리
        for i in range(0, len(col_pois), BATCH_SIZE):
            batch = col_pois[i:i + BATCH_SIZE]

            ids = []
            documents = []
            metadatas = []
            embeddings = []

            for poi in batch:
                poi_id = poi.get("id") or poi.get("name", f"unknown_{i}")
                doc_text = build_document_text(poi)

                # 임베딩
                vec = embedder.encode(doc_text, normalize_embeddings=True).tolist()

                # 메타데이터 (ChromaDB는 None 값 불허)
                meta = {
                    "id": poi_id,
                    "name": poi.get("name") or "",
                    "category": poi.get("category") or "",
                    "address": poi.get("address") or "",
                    "sido": poi.get("sido") or "",
                    "lat": float(poi["lat"]) if poi.get("lat") is not None else 0.0,
                    "lon": float(poi["lon"]) if poi.get("lon") is not None else 0.0,
                    "image_url": poi.get("image_url") or "",
                }
                artists = [a for a in (poi.get("artists") or []) if a]
                if artists:
                    meta["artists"] = ", ".join(artists)

                ids.append(poi_id)
                documents.append(doc_text)
                metadatas.append(meta)
                embeddings.append(vec)

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            print(f"  upsert {i+1}~{i+len(batch)} / {len(col_pois)}")

        print(f"  ✅ {col_name}: {collection.count()}건 인덱싱 완료")

    # 6. 최종 확인
    print(f"\n{'=' * 50}")
    print("전체 컬렉션 현황:")
    for col in chroma.list_collections():
        print(f"  {col.name}: {col.count()}건")
    print("✅ POI 인덱싱 완료")


if __name__ == "__main__":
    main()
