"""
check_rag_status.py
===================
현재 구축된 RAG/GraphRAG 데이터 요약 보고 (No Emoji for Windows Compatibility)
"""
import os
import json
import psycopg2
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

def check_status():
    print("="*50)
    print("K-Ride RAG/GraphRAG Build Status Summary")
    print("="*50)

    # 1. GraphRAG (NetworkX/JSON)
    graph_path = "models/kride_graph.json"
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding='utf-8') as f:
            graph_data = json.load(f)
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        poi_nodes = len([n for n in nodes if n.get('type') == 'POI'])
        artist_nodes = len([n for n in nodes if n.get('type') == 'Artist'])
        
        print(f"\n[1] GraphRAG (Knowledge Graph)")
        print(f"  - Total Nodes: {len(nodes):,}")
        print(f"    * POI Nodes: {poi_nodes:,}")
        print(f"    * Artist Nodes: {artist_nodes:,}")
        print(f"  - Total Edges: {len(edges):,}")
        
        communities = set([n.get('community') for n in nodes if n.get('community') is not None])
        print(f"  - Communities: {len(communities)}")
    else:
        print("\n[1] GraphRAG: File not found.")

    # 2. VectorDB (ChromaDB)
    chroma_path = "./chroma_db"
    if HAS_CHROMA and os.path.exists(chroma_path):
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            collections = client.list_collections()
            print(f"\n[2] VectorDB (ChromaDB)")
            if not collections:
                print("  - No collections found.")
            for col in collections:
                print(f"  - Collection: '{col.name}'")
                print(f"    * Embedded Docs: {col.count():,}")
        except Exception as e:
            print(f"\n[2] VectorDB: Error - {e}")
    elif not HAS_CHROMA:
        print("\n[2] VectorDB: chromadb library not installed.")
    else:
        print("\n[2] VectorDB: ./chroma_db folder not found.")

    # 3. Relational DB (PostgreSQL)
    if DATABASE_URL:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            
            cur.execute("SELECT count(*) FROM artist")
            artist_cnt = cur.fetchone()[0]
            
            cur.execute("SELECT count(*) FROM artist_poi")
            link_cnt = cur.fetchone()[0]
            
            cur.execute("SELECT category, count(*) FROM poi GROUP BY category")
            poi_stats = cur.fetchall()
            
            print(f"\n[3] Relational DB (PostgreSQL)")
            print(f"  - Registered Artists: {artist_cnt}")
            print(f"  - Artist-POI Links: {link_cnt}")
            print(f"  - POI Categories:")
            for cat, cnt in poi_stats:
                print(f"    * {cat}: {cnt:,}")
                
            conn.close()
        except Exception as e:
            print(f"\n[3] Relational DB: Connection error - {e}")

    print("\n" + "="*50)

if __name__ == "__main__":
    check_status()
