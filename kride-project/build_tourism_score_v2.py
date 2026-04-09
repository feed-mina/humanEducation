"""
build_tourism_score_v2.py
=========================
poi_attraction.csv × road_scored.csv → tourism_score_v2 생성

[ 목적 ]
  기존 tourism_score (POI 밀도 규칙 기반)에
  TabNet으로 예측한 POI 매력도(attraction_score_norm)를 보정값으로 결합.

  tourism_score_v2 = 0.7 × tourism_score + 0.3 × attraction_mean

[ 선행 조건 ]
  1. python kride-project/build_attraction_model.py  → poi_attraction.csv
  2. python kride-project/build_tourism_model.py     → road_scored.csv

[ 실행 ]
  python kride-project/build_tourism_score_v2.py

[ 출력 ]
  data/raw_ml/road_scored_v2.csv   ← tourism_score_v2, final_score_v2 추가
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_DIR = os.path.join(BASE_DIR, "data", "raw_ml")

ROAD_SCORED_PATH   = os.path.join(RAW_DIR, "road_scored.csv")
POI_ATTRACT_PATH   = os.path.join(RAW_DIR, "poi_attraction.csv")
OUTPUT_PATH        = os.path.join(RAW_DIR, "road_scored_v2.csv")

W_OLD        = 0.7   # 기존 tourism_score 가중치
W_ATTRACTION = 0.3   # 매력도 보정 가중치
RADIUS_DEG   = 0.005 # Spatial Join 반경 (≈500m at 서울 위도)

W_SAFETY_V2  = 0.6
W_TOURISM_V2 = 0.4


# ══════════════════════════════════════════════════════════════════════════════
def load_files():
    """road_scored.csv 와 poi_attraction.csv 로드"""
    for path, name in [(ROAD_SCORED_PATH, "road_scored.csv"),
                       (POI_ATTRACT_PATH, "poi_attraction.csv")]:
        if not os.path.exists(path):
            print(f"  ❌ {name} 없음: {path}")
            sys.exit(1)

    road = pd.read_csv(ROAD_SCORED_PATH, encoding="utf-8-sig")
    poi  = pd.read_csv(POI_ATTRACT_PATH, encoding="utf-8-sig")
    print(f"  road_scored.csv : {road.shape[0]:,}개 세그먼트")
    print(f"  poi_attraction  : {poi.shape[0]:,}개 POI")
    return road, poi


# ══════════════════════════════════════════════════════════════════════════════
def spatial_join_attraction(road: pd.DataFrame, poi: pd.DataFrame) -> np.ndarray:
    """
    각 도로 세그먼트 중심점 반경 RADIUS_DEG 내 POI attraction_score_norm 평균.

    road 좌표: start_lat (위도), start_lon (경도)
    poi  좌표: y_coord   (위도), x_coord   (경도)

    반환: shape (len(road),) — 매칭 POI 없는 세그먼트는 NaN
    """
    poi_lat = poi["y_coord"].values
    poi_lon = poi["x_coord"].values
    poi_sc  = poi["attraction_score_norm"].values

    # 도로 중심점 = (start + end) / 2 (end 없으면 start만)
    if "end_lat" in road.columns and "end_lon" in road.columns:
        seg_lat = (road["start_lat"].values + road["end_lat"].values) / 2
        seg_lon = (road["start_lon"].values + road["end_lon"].values) / 2
    else:
        seg_lat = road["start_lat"].values
        seg_lon = road["start_lon"].values

    attraction_mean = np.full(len(road), np.nan)

    for i, (rlat, rlon) in enumerate(zip(seg_lat, seg_lon)):
        mask = (
            (np.abs(poi_lat - rlat) <= RADIUS_DEG) &
            (np.abs(poi_lon - rlon) <= RADIUS_DEG)
        )
        if mask.any():
            attraction_mean[i] = poi_sc[mask].mean()

    matched = np.sum(~np.isnan(attraction_mean))
    print(f"  Spatial Join 결과: {matched:,}/{len(road):,} 세그먼트 매칭 "
          f"({matched/len(road)*100:.1f}%)")
    return attraction_mean


# ══════════════════════════════════════════════════════════════════════════════
def build_v2(road: pd.DataFrame, attraction_mean: np.ndarray) -> pd.DataFrame:
    """tourism_score_v2 및 final_score_v2 계산"""
    road = road.copy()

    # NaN (매칭 없는 세그먼트) → 기존 tourism_score 그대로 사용 (보정 없음)
    road["attraction_mean"] = attraction_mean
    no_match = road["attraction_mean"].isna()
    road["attraction_mean"] = road["attraction_mean"].fillna(road["tourism_score"])

    road["tourism_score_v2"] = (
        W_OLD        * road["tourism_score"] +
        W_ATTRACTION * road["attraction_mean"]
    ).clip(0, 1).round(6)

    road["final_score_v2"] = (
        W_SAFETY_V2  * road["safety_score"] +
        W_TOURISM_V2 * road["tourism_score_v2"]
    ).clip(0, 1).round(6)

    print(f"\n  tourism_score    평균: {road['tourism_score'].mean():.4f}")
    print(f"  tourism_score_v2 평균: {road['tourism_score_v2'].mean():.4f}  "
          f"(변화: {road['tourism_score_v2'].mean() - road['tourism_score'].mean():+.4f})")
    print(f"  final_score      평균: {road['final_score'].mean():.4f}")
    print(f"  final_score_v2   평균: {road['final_score_v2'].mean():.4f}  "
          f"(변화: {road['final_score_v2'].mean() - road['final_score'].mean():+.4f})")
    print(f"\n  매칭 없는 세그먼트: {no_match.sum():,}개 (기존 tourism_score 유지)")

    road.drop(columns=["attraction_mean"], inplace=True)
    return road


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  tourism_score_v2 생성 (POI 매력도 TabNet 보정)")
    print("=" * 65)

    print("\n[STEP 1] 파일 로드")
    road, poi = load_files()

    print("\n[STEP 2] Spatial Join (반경 ≈500m)")
    attraction_mean = spatial_join_attraction(road, poi)

    print("\n[STEP 3] tourism_score_v2 계산")
    road_v2 = build_v2(road, attraction_mean)

    print("\n[STEP 4] road_scored_v2.csv 저장")
    road_v2.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  ✅ {OUTPUT_PATH}")
    print(f"     shape: {road_v2.shape}")
    print(f"     신규 컬럼: tourism_score_v2, final_score_v2")

    print("\n" + "=" * 65)
    print("  완료")
    print()
    print("  [ 다음 단계 ]")
    print("    python kride-project/build_route_graph.py")
    print("    → road_scored_v2.csv 기반 route_graph.pkl 재빌드")
    print("=" * 65)


if __name__ == "__main__":
    main()
