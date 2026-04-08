"""
build_route_graph.py
====================
road_scored.csv → networkx 그래프 → models/route_graph.pkl

[ 실행 순서 ]
  python kride-project/build_route_graph.py

[ 단계 요약 ]
  STEP 1 : road_scored.csv 로드
  STEP 2 : 그래프 노드/엣지 구성 (start↔end 좌표)
  STEP 3 : 연결성 분석
  STEP 4 : route_graph.pkl 저장

[ 출력 파일 ]
  kride-project/models/route_graph.pkl
"""

import os
import sys
import math
import pickle
import warnings

import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_DIR    = os.path.join(BASE_DIR, "data", "raw_ml")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCORED_PATH = os.path.join(RAW_DIR,    "road_scored.csv")
GRAPH_PATH  = os.path.join(MODELS_DIR, "route_graph.pkl")

# 좌표 반올림 정밀도 (노드 병합 기준 — 소수점 4자리 ≈ 11m)
COORD_PRECISION = 4


# ──────────────────────────────────────────────────────────────────────────────
def haversine(c1, c2) -> float:
    """두 (lat, lon) 좌표 사이 거리 (km)"""
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def round_coord(lat, lon, precision=COORD_PRECISION):
    """좌표를 지정 정밀도로 반올림 (노드 키 생성용)"""
    return (round(float(lat), precision), round(float(lon), precision))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: road_scored.csv 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: road_scored.csv 로드")
print("=" * 65)

if not os.path.exists(SCORED_PATH):
    print(f"  ❌ {SCORED_PATH} 없음. build_tourism_model.py를 먼저 실행하세요.")
    sys.exit(1)

df = pd.read_csv(SCORED_PATH, encoding="utf-8-sig")
print(f"  shape: {df.shape}")
print(f"  컬럼: {list(df.columns)}\n")

# 필수 컬럼 확인
REQUIRED = ["start_lat", "start_lon", "end_lat", "end_lon",
            "length_km", "safety_score", "tourism_score", "final_score"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    print(f"  ❌ 필수 컬럼 없음: {missing}")
    sys.exit(1)

# 좌표 결측 제거
before = len(df)
df = df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"])
print(f"  좌표 결측 제거: {before:,} → {len(df):,}행\n")

# 수치 변환
for col in REQUIRED:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=REQUIRED)
print(f"  수치 변환 후: {len(df):,}행\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 그래프 구성
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 2: networkx 그래프 구성")
print("=" * 65)

G = nx.Graph()

added = 0
skipped = 0

for _, row in df.iterrows():
    s = round_coord(row["start_lat"], row["start_lon"])
    e = round_coord(row["end_lat"],   row["end_lon"])

    # 시작 == 끝인 세그먼트 스킵 (self-loop)
    if s == e:
        skipped += 1
        continue

    # 길이가 0이면 haversine 계산값으로 대체

    # 자전거 도로 데이터에서 length_km 값이 0 또는 누락된 행이 있을 수 있다. 도로명: 한강변 자전거도로 3구간
    # start: (37.5123, 126.9876)
    # end:   (37.5145, 126.9901)
    # length_km: 0   ← 데이터 오류
    # 이런 경우 haversine 함수를 사용해서 실제 두 좌표 간의 직선거리를 계산해 length_km 값을 대체한다.여기서haversine은 지구 곡률을 고려한 두 좌표 사이의 실제 거리 공식이다 


    length = float(row["length_km"])
    if length <= 0:
        length = haversine(s, e)
        if length < 0.001:
            skipped += 1
            continue

    safety  = float(row["safety_score"])
    tourism = float(row["tourism_score"])
    final   = float(row["final_score"])

    # 기존 엣지가 있으면 점수가 더 높은 쪽으로 업데이트
    if G.has_edge(s, e):
        if final > G[s][e]["final_score"]:
            G[s][e].update(dict(
                weight=1.0 - final,
                safety_score=safety,
                tourism_score=tourism,
                final_score=final,
                length_km=length,
            ))
    else:
        edge_attrs = dict(
            weight=1.0 - final,   # Dijkstra: 낮을수록 선호
            safety_score=safety,
            tourism_score=tourism,
            final_score=final,
            length_km=length,
        )
        # 선택적 컬럼 추가
        for opt in ["route_name", "sigungu", "road_type", "tourist_count",
                    "facility_count", "cultural_count"]:
            if opt in df.columns:
                edge_attrs[opt] = row.get(opt, "")

        G.add_edge(s, e, **edge_attrs)
        added += 1

print(f"  추가된 엣지: {added:,}개")
print(f"  스킵된 세그먼트: {skipped:,}개")
print(f"  노드 수: {G.number_of_nodes():,}")
print(f"  엣지 수: {G.number_of_edges():,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: 연결성 분석
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 3: 연결성 분석")
print("=" * 65)

components = list(nx.connected_components(G))
print(f"  연결 컴포넌트 수: {len(components):,}")
largest = max(components, key=len)
print(f"  최대 컴포넌트 노드 수: {len(largest):,} "
      f"({len(largest)/G.number_of_nodes()*100:.1f}%)")
print(f"  최대 컴포넌트 엣지 수: {G.subgraph(largest).number_of_edges():,}\n")

# 최대 연결 서브그래프도 함께 저장 (경로 탐색 시 활용)
G_main = G.subgraph(largest).copy()
print(f"  G_main (최대 연결 그래프) 노드/엣지: "
      f"{G_main.number_of_nodes():,} / {G_main.number_of_edges():,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: 저장
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 4: route_graph.pkl 저장")
print("=" * 65)

os.makedirs(MODELS_DIR, exist_ok=True)

# G (전체) + G_main (최대 연결) 함께 저장
# G(전체) VS G_main(최대 연결) 비교
# 자전거 도로데이터로 그래프를 만들면 모든 도로가 하나로 이어지지않고 섬처럼 분리된 구간들이 생긴다. 
# G (전체 그래프)
# ├── 컴포넌트 A: 서울 도심 (노드 800개) ← 가장 큰 덩어리
# ├── 컴포넌트 B: 한강변 일부 (노드 50개)
# ├── 컴포넌트 C: 경기 외곽 (노드 20개)
# └── 컴포넌트 D: 고립된 도로 (노드 3개)

# G_main은 이 중 가장 큰 덩어리 만 추출한다 G(전체) 지도 시각화, 통계, 모든 도로 표시용
# G_main 경로 탐색(다익스트라 알고리즘)용 , 연결 안된 섬에서는 경로 탐색 자체가 불가능
# 경로 탐색시 G_main만 사용한다. 출발지와 도착지가 서로 다른 컴포넌트에 있으면 nx.shortest_path가 에러를 뱉는다.

graph_data = {
    "G":      G,
    "G_main": G_main,
    "meta": {
        "nodes":      G.number_of_nodes(),
        "edges":      G.number_of_edges(),
        "components": len(components),
        "main_nodes": G_main.number_of_nodes(),
        "main_edges": G_main.number_of_edges(),
        "coord_precision": COORD_PRECISION,
    }
}
# [메모] coord_precision은 어떤 의미인가요 ?  
# coord_precision = 4 는 위경도 좌표를 소수점 몇 자리까지 반올림해서 그래프로 쏠지 결정하는 값이다

# 여기서는 반올림 정밀도 = 거리 허용 오차입니다. 
# COORD_PRECISION	소수점 자리	허용 오차 (대략)
# 2	0.01도	~1,100m 
# 3	0.001도	~111m
# 4	0.0001도	~11m ← 현재 설정 
# 값을 높이면 연결성은 좋아지지만 너무 많은 노드가 합쳐져 정확도가 떨어지고 낮추면 연결이 끊기는 섬이 많아 집니다.abs

with open(GRAPH_PATH, "wb") as f:
    pickle.dump(graph_data, f)

print(f"  ✅ route_graph.pkl → {GRAPH_PATH}")
print(f"     전체 그래프: {G.number_of_nodes():,} 노드 / {G.number_of_edges():,} 엣지")
print(f"     최대 연결:   {G_main.number_of_nodes():,} 노드 / {G_main.number_of_edges():,} 엣지")

print("\n" + "=" * 65)
print("✅ build_route_graph.py 완료")
print("=" * 65)
