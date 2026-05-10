"""
evaluate_kculture_graph.py
==========================
K-Culture 데이터 및 GraphRAG 네트워크 성능/통계 분석
- 그래프 통계 (노드, 엣지, 밀도)
- 드라마별 촬영지 분포
- 지역별(시도) 촬영지 분포
- 커뮤니티 분포 시각화용 데이터 추출
"""
import os
import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import psycopg2
from dotenv import load_dotenv

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
GRAPH_JSON_PATH = "models/kride_graph.json"

def evaluate_graph():
    print("Loading graph data...")
    if not os.path.exists(GRAPH_JSON_PATH):
        print(f"Error: {GRAPH_JSON_PATH} not found. Run kride_graph_builder.py first.")
        return

    with open(GRAPH_JSON_PATH, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    nodes_df = pd.DataFrame(data['nodes']).fillna("Unknown")
    edges_df = pd.DataFrame(data['edges']).fillna("Unknown")
    
    # 1. 그래프 기본 통계
    stats = {
        "total_nodes": len(nodes_df),
        "total_edges": len(edges_df),
        "poi_nodes": len(nodes_df[nodes_df['type'] == 'POI']),
        "artist_nodes": len(nodes_df[nodes_df['type'] == 'Artist']),
        "density": len(edges_df) / (len(nodes_df) * (len(nodes_df)-1) / 2) if len(nodes_df) > 1 else 0
    }
    
    print("Graph Stats:", stats)
    
    # 2. 드라마별 촬영지 수 TOP 10
    artist_edges = edges_df[edges_df['source'].str.contains('artist_')]
    artist_counts = artist_edges.merge(nodes_df[['id', 'name']], left_on='source', right_on='id')
    top_dramas = artist_counts['name'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    plt.barh(top_dramas.index[::-1], top_dramas.values[::-1], color='skyblue')
    # sns.barplot(x=top_dramas.values, y=top_dramas.index, palette='viridis')
    plt.title("드라마별 촬영지 수 TOP 10")
    plt.xlabel("촬영지 수")
    plt.tight_layout()
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/kculture_top_dramas.png")
    print("Saved: report/figures/kculture_top_dramas.png")
    
    # 3. 지역별 촬영지 분포 (POI 중 kculture 카테고리)
    # DB에서 시도 정보 가져오기
    conn = psycopg2.connect(DATABASE_URL)
    region_df = pd.read_sql("""
        SELECT COALESCE(sido, 'Unknown') as sido, count(*) as count 
        FROM poi 
        WHERE category = 'kculture' 
        GROUP BY sido 
        ORDER BY count DESC
    """, conn)
    conn.close()
    
    if not region_df.empty:
        plt.figure(figsize=(12, 6))
        plt.barh(region_df.head(10)['sido'][::-1], region_df.head(10)['count'][::-1], color='salmon')
        # sns.barplot(x='count', y='sido', data=region_df.head(10), palette='magma')
        plt.title("지역별 K-Culture 촬영지 분포 (TOP 10)")
        plt.xlabel("촬영지 수")
        plt.tight_layout()
        plt.savefig("report/figures/kculture_region_dist.png")
        print("Saved: report/figures/kculture_region_dist.png")
    
    # 4. 커뮤니티 분포
    if 'community' in nodes_df.columns:
        comm_counts = nodes_df['community'].value_counts().head(15)
        plt.figure(figsize=(10, 6))
        comm_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, cmap='Set3')
        plt.title("GraphRAG 커뮤니티 노드 분포 (Top 15)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig("report/figures/kculture_community_pie.png")
        print("Saved: report/figures/kculture_community_pie.png")

    # 5. 테이블 산출물 저장
    os.makedirs("report/tables", exist_ok=True)
    nodes_df[['type', 'category', 'name']].head(10).to_csv("report/tables/graph_nodes_sample.csv", index=False)
    region_df.to_csv("report/tables/kculture_region_stats.csv", index=False)
    
    # 요약 JSON 저장
    with open("report/data/kculture_eval_summary.json", "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("Evaluation complete. Check report/ folder.")

if __name__ == "__main__":
    evaluate_graph()
