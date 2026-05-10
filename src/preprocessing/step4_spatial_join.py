# =============================================================
# Step 4: Spatial Join - 자전거도로 경로 기준 POI 수 집계
# =============================================================
# 입력 1: data/raw_ml/road_clean.csv      (자전거도로, 5,319행)
# 입력 2: data/raw_ml/tour_poi.csv        (관광지 POI, 3,407행)
# 입력 3: data/raw_ml/facility_clean.csv  (편의시설, 3,368행)
#
# 출력  : data/raw_ml/road_features.csv
#         - road_clean 컬럼 + 집계 피처 추가
#
# 추가 피처:
#   tourist_count   - 경로 1km 반경 내 관광지(type=12) 수
#   cultural_count  - 경로 1km 반경 내 문화시설(type=14) 수
#   leisure_count   - 경로 1km 반경 내 레저스포츠(type=28) 수
#   facility_count  - 경로 500m 반경 내 편의시설 수
#
# 좌표계:
#   원본 데이터: WGS84 (EPSG:4326) - 위경도 십진수
#   버퍼 계산용: EPSG:5179 (Korea 2000 Unified) - 미터 단위
#   → 버퍼 계산 후 결과는 다시 EPSG:4326으로 저장
#
# [SQL 비유] PostGIS 기준
#   SELECT r.*, COUNT(t.contentid) AS tourist_count
#   FROM road_clean r
#   LEFT JOIN tour_poi t
#     ON ST_DWithin(r.geom::geography, t.geom::geography, 1000)
#   GROUP BY r.rowid;
# =============================================================

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import os

# ── 경로 설정 (Jupyter 호환) ──────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

ROAD_PATH     = os.path.join(BASE_DIR, "data", "raw_ml", "road_clean.csv")
TOUR_PATH     = os.path.join(BASE_DIR, "data", "raw_ml", "tour_poi.csv")
FACILITY_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "facility_clean.csv")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "raw_ml", "road_features.csv")

# 좌표계 설정
CRS_WGS84 = "EPSG:4326"   # 위경도 (원본)
CRS_KOREA = "EPSG:5179"   # Korea 2000 Unified CS (미터 단위, 버퍼 계산용)

print(f"BASE_DIR : {BASE_DIR}\n")

# =============================================================
# STEP ①: 데이터 로드
# =============================================================
print("▶ 데이터 로드 중...")

df_road     = pd.read_csv(ROAD_PATH,     encoding="utf-8-sig")
df_tour     = pd.read_csv(TOUR_PATH,     encoding="utf-8-sig")
df_facility = pd.read_csv(FACILITY_PATH, encoding="utf-8-sig")

print(f"  road_clean     : {df_road.shape}")
print(f"  tour_poi       : {df_tour.shape}")
print(f"  facility_clean : {df_facility.shape}\n")

# 수치형 변환 (안전하게)
for col in ["start_lat", "start_lon", "end_lat", "end_lon"]:
    df_road[col] = pd.to_numeric(df_road[col], errors="coerce")

for col in ["mapx", "mapy"]:
    df_tour[col] = pd.to_numeric(df_tour[col], errors="coerce")

for col in ["x", "y"]:
    df_facility[col] = pd.to_numeric(df_facility[col], errors="coerce")

# =============================================================
# STEP ②: GeoDataFrame 생성 (WGS84)
# =============================================================
print("▶ GeoDataFrame 생성 중...")

# --- 자전거도로: 기점 → 종점 LineString ---
# 좌표 쌍이 모두 있는 행만 사용
road_valid = df_road.dropna(subset=["start_lon", "start_lat", "end_lon", "end_lat"]).copy()
road_valid["geometry"] = road_valid.apply(
    lambda r: LineString([
        (r["start_lon"], r["start_lat"]),  # (경도, 위도) 순서
        (r["end_lon"],   r["end_lat"])
    ]),
    axis=1
)
gdf_road = gpd.GeoDataFrame(road_valid, geometry="geometry", crs=CRS_WGS84)
print(f"  gdf_road (LineString) : {len(gdf_road):,}행")

# --- 관광지 POI: (mapx=경도, mapy=위도) Point ---
tour_valid = df_tour.dropna(subset=["mapx", "mapy"]).copy()
tour_valid["geometry"] = gpd.points_from_xy(tour_valid["mapx"], tour_valid["mapy"])
gdf_tour = gpd.GeoDataFrame(tour_valid, geometry="geometry", crs=CRS_WGS84)
print(f"  gdf_tour (Point)      : {len(gdf_tour):,}행")

# --- 편의시설: (x=경도, y=위도) Point ---
fac_valid = df_facility.dropna(subset=["x", "y"]).copy()
fac_valid["geometry"] = gpd.points_from_xy(fac_valid["x"], fac_valid["y"])
gdf_facility = gpd.GeoDataFrame(fac_valid, geometry="geometry", crs=CRS_WGS84)
print(f"  gdf_facility (Point)  : {len(gdf_facility):,}행\n")

# =============================================================
# STEP ③: 좌표계 변환 (WGS84 → EPSG:5179, 미터 단위)
# =============================================================
# 위경도 십진수로 buffer(1000)를 하면 1000도(!) 가 되어버림
# EPSG:5179로 변환해야 buffer(1000) = 실제 1000m
print("▶ 좌표계 변환 중 (WGS84 → EPSG:5179)...")

gdf_road_m     = gdf_road.to_crs(CRS_KOREA)
gdf_tour_m     = gdf_tour.to_crs(CRS_KOREA)
gdf_facility_m = gdf_facility.to_crs(CRS_KOREA)

print("  변환 완료\n")

# =============================================================
# STEP ④: 버퍼 생성 + Spatial Join
# =============================================================

def count_poi_in_buffer(gdf_lines, gdf_points, buffer_m, count_col_name,
                        filter_col=None, filter_val=None):
    """
    각 LineString 기준 buffer_m 미터 반경 내 Point 수를 집계하여
    count_col_name 컬럼으로 반환.

    filter_col / filter_val : gdf_points 에서 특정 값만 필터링할 때 사용
    """
    # POI 필터링
    if filter_col and filter_val is not None:
        pts = gdf_points[gdf_points[filter_col].astype(str) == str(filter_val)].copy()
    else:
        pts = gdf_points.copy()

    # 도로 버퍼 생성
    gdf_buf = gdf_lines.copy()
    gdf_buf["geometry"] = gdf_buf.geometry.buffer(buffer_m)

    # Spatial Join: 버퍼 내 POI 연결
    joined = gpd.sjoin(pts, gdf_buf[["geometry"]], how="left", predicate="within")
    # index_right = 도로 인덱스
    counts = joined.groupby("index_right").size().rename(count_col_name)

    # 도로 DataFrame에 병합 (없으면 0)
    result = gdf_lines.index.to_series().map(counts).fillna(0).astype(int)
    return result


# --- 관광지 (1km 버퍼) ---
print("▶ 관광지 Spatial Join (1km 반경) 중...")
gdf_road_m["tourist_count"]  = count_poi_in_buffer(
    gdf_road_m, gdf_tour_m, buffer_m=1000,
    count_col_name="tourist_count",
    filter_col="contentTypeId", filter_val=12
)
gdf_road_m["cultural_count"] = count_poi_in_buffer(
    gdf_road_m, gdf_tour_m, buffer_m=1000,
    count_col_name="cultural_count",
    filter_col="contentTypeId", filter_val=14
)
gdf_road_m["leisure_count"]  = count_poi_in_buffer(
    gdf_road_m, gdf_tour_m, buffer_m=1000,
    count_col_name="leisure_count",
    filter_col="contentTypeId", filter_val=28
)
print(f"  tourist_count  평균: {gdf_road_m['tourist_count'].mean():.2f}")
print(f"  cultural_count 평균: {gdf_road_m['cultural_count'].mean():.2f}")
print(f"  leisure_count  평균: {gdf_road_m['leisure_count'].mean():.2f}\n")

# --- 편의시설 (500m 버퍼) ---
print("▶ 편의시설 Spatial Join (500m 반경) 중...")
gdf_road_m["facility_count"] = count_poi_in_buffer(
    gdf_road_m, gdf_facility_m, buffer_m=500,
    count_col_name="facility_count"
)
print(f"  facility_count 평균: {gdf_road_m['facility_count'].mean():.2f}\n")

# =============================================================
# STEP ⑤: 결과 정리 및 저장
# =============================================================
# geometry 컬럼 제거 후 일반 DataFrame으로 변환
NEW_FEATURE_COLS = ["tourist_count", "cultural_count", "leisure_count", "facility_count"]
df_result = pd.DataFrame(gdf_road_m.drop(columns=["geometry"]))

print("▶ 최종 결과 요약")
print(f"  shape  : {df_result.shape}")
print(f"  컬럼   : {list(df_result.columns)}\n")

print("새로 추가된 피처 기술 통계:")
print(df_result[NEW_FEATURE_COLS].describe().round(2))
print()

print("피처 분포 (0이 아닌 행 수):")
for col in NEW_FEATURE_COLS:
    nonzero = (df_result[col] > 0).sum()
    print(f"  {col:20s}: {nonzero:,}행 ({nonzero/len(df_result)*100:.1f}%) 에서 1개 이상")
print()

print("샘플 (상위 5행, 피처 컬럼만):")
print(df_result[["start_lat", "start_lon"] + NEW_FEATURE_COLS].head(5).to_string())
print()

# CSV 저장
df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 저장 완료 → {OUTPUT_PATH}")
print(f"   파일 크기 : {os.path.getsize(OUTPUT_PATH):,} bytes")
