"""
build_poi_recommender_v2.py
===========================
Co-occurrence 기반 방문지 추천 모델 v2

[ v1 대비 개선 사항 ]
  1. 카테고리 부스트 — 시드와 같은 VISIT_AREA_TYPE_CD 장소 점수 가중 증가
  2. 현재 위치(current_lat/lon) 파라미터 추가 — 시드 중심이 아닌 사용자 위치 기준 반경 필터
  3. tour_poi.csv 연동 — 관광 카테고리(cat1) 기반 부스트 지원
  4. recommend() 인터페이스 확장 (category_boost, current_lat, current_lon)
  5. 카테고리별 Recall@5 분석 추가

[ 알고리즘 ]
  1. 여행별 방문 장소 집합 (관광지 전용, VISIT_AREA_TYPE_CD < 21)
  2. 장소 쌍 (A, B) Co-occurrence 카운트 → Jaccard 유사도 정규화
  3. recommend_v2():
       ① Jaccard 점수 합산
       ② [옵션] 현재 위치 or 시드 중심 기준 반경 필터
       ③ [옵션] 시드와 같은 카테고리 장소 점수 × category_boost_factor
       ④ Top-N 반환

[ 평가 ]
  TRAVEL_ID 기준 70 / 20 / 10 분할
  각 test 여행: 앞 50% 방문지 → 뒤 50% 예측
  Recall@5, Recall@10 측정

[ 출력 파일 ]
  models/poi_cooccurrence_v2.pkl
  models/poi_rec_meta_v2.json
"""

import argparse
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--min_trip_freq", type=int, default=2,
                    help="최소 여행 등장 횟수 (기본 2)")
parser.add_argument("--top_n", type=int, default=10,
                    help="추천 반환 개수 (기본 10)")
parser.add_argument("--max_dist_km", type=float, default=20.0,
                    help="추천 반경 km (0이면 무제한, 기본 20)")
parser.add_argument("--category_boost_factor", type=float, default=1.5,
                    help="같은 카테고리 장소 점수 배율 (기본 1.5)")
parser.add_argument("--exclude_type_ge", type=int, default=21,
                    help="VISIT_AREA_TYPE_CD >= 이 값인 장소 제외 (기본 21)")
args = parser.parse_args()

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

AIHUB_DIR  = os.path.join(BASE_DIR, "data", "ai-hub",
                           "국내 여행로그 수도권_2023", "02.라벨링데이터")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw_ml")

VISIT_CSV    = os.path.join(AIHUB_DIR, "tn_visit_area_info_방문지정보_E.csv")
TOUR_POI_CSV = os.path.join(RAW_DIR, "tour_poi.csv")
MODEL_PKL    = os.path.join(MODELS_DIR, "poi_cooccurrence_v2.pkl")
META_JSON    = os.path.join(MODELS_DIR, "poi_rec_meta_v2.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Haversine 거리 ─────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 + 카테고리 정보 구성
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: 데이터 로드 + 카테고리 / 좌표 정보 추출")
print("=" * 65)

if not os.path.exists(VISIT_CSV):
    print(f"  ❌ {VISIT_CSV} 없음")
    sys.exit(1)

df = pd.read_csv(VISIT_CSV, encoding="utf-8-sig")
df = df[["TRAVEL_ID", "VISIT_ORDER", "VISIT_AREA_NM",
         "X_COORD", "Y_COORD", "VISIT_AREA_TYPE_CD"]].copy()
df = df.dropna(subset=["VISIT_AREA_NM"])
df["VISIT_AREA_NM"]     = df["VISIT_AREA_NM"].astype(str).str.strip()
df["X_COORD"]           = pd.to_numeric(df["X_COORD"], errors="coerce")
df["Y_COORD"]           = pd.to_numeric(df["Y_COORD"], errors="coerce")
df["VISIT_AREA_TYPE_CD"] = pd.to_numeric(df["VISIT_AREA_TYPE_CD"], errors="coerce")
df = df[df["VISIT_AREA_NM"] != ""].sort_values(
    ["TRAVEL_ID", "VISIT_ORDER"]).reset_index(drop=True)

print(f"  전체: {len(df):,}행 / {df['TRAVEL_ID'].nunique():,}여행 / "
      f"{df['VISIT_AREA_NM'].nunique():,}장소")

# ── 비관광 제거 ───────────────────────────────────────────────────────────────
before = len(df)
df_tourist = df[
    df["VISIT_AREA_TYPE_CD"].isna() |
    (df["VISIT_AREA_TYPE_CD"] < args.exclude_type_ge)
].copy()
print(f"  비관광 제거: {before - len(df_tourist):,}행 → {len(df_tourist):,}행 잔존")

# ── 장소별 대표 좌표 + 대표 타입 ─────────────────────────────────────────────
place_stats = (
    df_tourist.groupby("VISIT_AREA_NM")
    .agg(lat=("Y_COORD", "median"),
         lon=("X_COORD", "median"),
         type_cd=("VISIT_AREA_TYPE_CD", "median"))
    .dropna(subset=["lat", "lon"])
)
has_coord_set = set(place_stats.index)
print(f"  좌표 보유 장소: {len(has_coord_set):,}개")

# ── tour_poi.csv 카테고리 연동 (선택적) ──────────────────────────────────────
poi_cat_map: dict = {}   # 장소명 → cat1 코드 ("A01"~)
if os.path.exists(TOUR_POI_CSV):
    try:
        df_poi = pd.read_csv(TOUR_POI_CSV, encoding="utf-8-sig")
        if "title" in df_poi.columns and "cat1" in df_poi.columns:
            df_poi["title"] = df_poi["title"].astype(str).str.strip()
            poi_cat_map = df_poi.set_index("title")["cat1"].to_dict()
            print(f"  tour_poi.csv 카테고리 연동: {len(poi_cat_map):,}개")
    except Exception as e:
        print(f"  ⚠️ tour_poi.csv 로드 실패: {e}")
else:
    print(f"  ℹ️ tour_poi.csv 없음 → VISIT_AREA_TYPE_CD 기반 카테고리 사용")

# 방문지 타입 → 카테고리 그룹 매핑 (대분류)
# VISIT_AREA_TYPE_CD 1~5=자연 6~10=역사 11~15=레저 16~20=문화/쇼핑
def type_to_cat_group(type_cd) -> str:
    try:
        t = int(type_cd)
        if t <= 5:   return "자연"
        if t <= 10:  return "역사"
        if t <= 15:  return "레저"
        return "문화"
    except (ValueError, TypeError):
        return "기타"

place_stats["cat_group"] = place_stats["type_cd"].apply(type_to_cat_group)
# tour_poi.csv의 cat1이 있으면 덮어씌우기
for place_nm in place_stats.index:
    if place_nm in poi_cat_map:
        cat1 = str(poi_cat_map[place_nm])
        cat_map = {"A01": "자연", "A02": "문화", "A03": "레저",
                   "A04": "쇼핑", "A05": "음식"}
        group = cat_map.get(cat1[:3], "기타")
        place_stats.at[place_nm, "cat_group"] = group

print(f"  카테고리 분포:\n{place_stats['cat_group'].value_counts().to_string()}")

df_final = df_tourist[df_tourist["VISIT_AREA_NM"].isin(has_coord_set)].copy()
print(f"  최종 데이터: {len(df_final):,}행 / {df_final['VISIT_AREA_NM'].nunique():,}장소")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 여행별 방문 장소 집합 + min_trip_freq 필터
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: 여행별 방문 장소 집합 + 빈도 필터")
print("=" * 65)

trip_sequences: dict = {}
for tid, grp in df_final.groupby("TRAVEL_ID"):
    seq = grp["VISIT_AREA_NM"].tolist()
    if seq:
        trip_sequences[tid] = seq

place_trip_count = defaultdict(int)
for seq in trip_sequences.values():
    for place in set(seq):
        place_trip_count[place] += 1

known_places = {p for p, cnt in place_trip_count.items()
                if cnt >= args.min_trip_freq}
print(f"  min_trip_freq={args.min_trip_freq} 후 vocab: {len(known_places):,}개")

place_list = sorted(known_places)
place2idx  = {p: i for i, p in enumerate(place_list)}
idx2place  = {i: p for p, i in place2idx.items()}
VOCAB      = len(place_list)

# 좌표 / 카테고리 배열
place_lat  = np.array([place_stats.loc[p, "lat"] if p in place_stats.index else np.nan
                        for p in place_list], dtype=np.float32)
place_lon  = np.array([place_stats.loc[p, "lon"] if p in place_stats.index else np.nan
                        for p in place_list], dtype=np.float32)
place_cat  = np.array([place_stats.loc[p, "cat_group"] if p in place_stats.index else "기타"
                        for p in place_list])

print(f"  위도 범위: {np.nanmin(place_lat):.3f} ~ {np.nanmax(place_lat):.3f}")
print(f"  경도 범위: {np.nanmin(place_lon):.3f} ~ {np.nanmax(place_lon):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: 70 / 20 / 10 분할
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: TRAVEL_ID 기준 70 / 20 / 10 분할")
print("=" * 65)

np.random.seed(42)
all_tids = np.array(list(trip_sequences.keys()))
np.random.shuffle(all_tids)
n       = len(all_tids)
n_train = int(n * 0.70)
n_val   = int(n * 0.20)
train_tids = set(all_tids[:n_train])
val_tids   = set(all_tids[n_train: n_train + n_val])
test_tids  = set(all_tids[n_train + n_val:])
print(f"  train={len(train_tids):,} / val={len(val_tids):,} / test={len(test_tids):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Co-occurrence 행렬 + Jaccard 정규화
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: Co-occurrence 행렬 (train set)")
print("=" * 65)

co_occ    = np.zeros((VOCAB, VOCAB), dtype=np.float32)
place_cnt = np.zeros(VOCAB, dtype=np.float32)

for tid in train_tids:
    seq = trip_sequences[tid]
    idxs = list({place2idx[p] for p in seq if p in place2idx})
    for i in idxs:
        place_cnt[i] += 1
    for k, i in enumerate(idxs):
        for j in idxs[k + 1:]:
            co_occ[i][j] += 1
            co_occ[j][i] += 1

print(f"  co-occurrence 행렬: {VOCAB} × {VOCAB}")
print(f"  비어있지 않은 셀: {int((co_occ > 0).sum()):,}개")

print("  Jaccard 정규화 중...")
jaccard = np.zeros((VOCAB, VOCAB), dtype=np.float32)
for i in range(VOCAB):
    for j in range(VOCAB):
        if i == j:
            continue
        denom = place_cnt[i] + place_cnt[j] - co_occ[i][j]
        if denom > 0:
            jaccard[i][j] = co_occ[i][j] / denom

print(f"  Jaccard 최대={jaccard.max():.4f} / 평균(비영)={jaccard[jaccard > 0].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: 추천 함수 v2 (카테고리 부스트 + 현재 위치 파라미터)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 5: 추천 함수 v2 (반경={args.max_dist_km}km, "
      f"카테고리 부스트 x{args.category_boost_factor})")
print("=" * 65)


def recommend_v2(
    seed_places: list,
    top_n: int = 10,
    exclude: set = None,
    max_dist_km: float = None,
    current_lat: float = None,
    current_lon: float = None,
    category_boost: bool = True,
    category_boost_factor: float = None,
) -> list:
    """
    v2 추천 함수

    seed_places      : 이미 방문한 장소명 리스트
    top_n            : 추천 개수
    exclude          : 제외 장소 집합
    max_dist_km      : 반경 제한 km (None이면 args.max_dist_km 사용)
    current_lat/lon  : 사용자 현재 위치 (None이면 시드 중심 사용 — v1 동작)
    category_boost   : 시드와 같은 카테고리 장소 점수 부스트 여부
    category_boost_factor : 부스트 배율 (None이면 args.category_boost_factor 사용)

    반환: [(장소명, 점수), ...]
    """
    if exclude is None:
        exclude = set(seed_places)
    if max_dist_km is None:
        max_dist_km = args.max_dist_km
    if category_boost_factor is None:
        category_boost_factor = args.category_boost_factor

    seed_idx = [place2idx[p] for p in seed_places if p in place2idx]

    # ── 기준 좌표 결정 ─────────────────────────────────────────────────────
    #   current_lat/lon 제공 시 → 사용자 위치 기준
    #   없으면 → 시드 중심 (v1 동작)
    center_lat, center_lon = current_lat, current_lon
    if center_lat is None and seed_idx and max_dist_km > 0:
        lats = [place_lat[i] for i in seed_idx if not np.isnan(place_lat[i])]
        lons = [place_lon[i] for i in seed_idx if not np.isnan(place_lon[i])]
        if lats:
            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))

    # ── Jaccard 점수 계산 ──────────────────────────────────────────────────
    if not seed_idx:
        scores = place_cnt.copy()
    else:
        scores = jaccard[seed_idx].sum(axis=0)

    # ── 반경 필터 ─────────────────────────────────────────────────────────
    if center_lat is not None and center_lon is not None and max_dist_km > 0:
        for j in range(VOCAB):
            if np.isnan(place_lat[j]) or np.isnan(place_lon[j]):
                scores[j] = -1.0
                continue
            d = haversine_km(center_lat, center_lon,
                             float(place_lat[j]), float(place_lon[j]))
            if d > max_dist_km:
                scores[j] = -1.0

    # ── 카테고리 부스트 ────────────────────────────────────────────────────
    if category_boost and seed_idx:
        # 시드의 카테고리 집합
        seed_cats = {place_cat[i] for i in seed_idx}
        for j in range(VOCAB):
            if scores[j] > 0 and place_cat[j] in seed_cats:
                scores[j] *= category_boost_factor

    # ── 이미 방문한 장소 제외 ─────────────────────────────────────────────
    for p in exclude:
        if p in place2idx:
            scores[place2idx[p]] = -1.0

    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(idx2place[i], float(scores[i])) for i in top_idx if scores[i] >= 0]


# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def evaluate_split(tids: set, split_name: str, use_boost: bool = True):
    recall5_list, recall10_list = [], []
    for tid in tids:
        seq = trip_sequences.get(tid, [])
        known = [p for p in seq if p in place2idx]
        if len(known) < 2:
            continue
        split_pt = max(1, len(known) // 2)
        seed   = known[:split_pt]
        target = set(known[split_pt:])
        if not target:
            continue
        recs5  = [p for p, _ in recommend_v2(seed, top_n=5,  exclude=set(seed),
                                              category_boost=use_boost)]
        recs10 = [p for p, _ in recommend_v2(seed, top_n=10, exclude=set(seed),
                                              category_boost=use_boost)]
        recall5_list.append(len(set(recs5) & target) / len(target))
        recall10_list.append(len(set(recs10) & target) / len(target))

    r5  = float(np.mean(recall5_list))  if recall5_list  else 0.0
    r10 = float(np.mean(recall10_list)) if recall10_list else 0.0
    label = "(부스트 O)" if use_boost else "(부스트 X)"
    print(f"  [{split_name} {label}] Recall@5={r5:.4f}  Recall@10={r10:.4f}")
    return r5, r10


val_r5,      val_r10      = evaluate_split(val_tids,  "val ", use_boost=True)
test_r5,     test_r10     = evaluate_split(test_tids, "test", use_boost=True)
test_r5_nb,  test_r10_nb  = evaluate_split(test_tids, "test", use_boost=False)

print(f"\n  카테고리 부스트 효과 (test): "
      f"Recall@5 {test_r5_nb:.4f} → {test_r5:.4f} "
      f"({'↑' if test_r5 > test_r5_nb else '↓'}{abs(test_r5 - test_r5_nb):.4f})")


# ── 인기도 베이스라인 ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: 인기도 베이스라인 비교")
print("=" * 65)

def evaluate_popularity_baseline(tids: set, split_name: str):
    recall5_list, recall10_list = [], []
    top_popular = [p for p in sorted(place2idx.keys(),
                   key=lambda x: place_cnt[place2idx[x]], reverse=True)]
    for tid in tids:
        seq = trip_sequences.get(tid, [])
        known = [p for p in seq if p in place2idx]
        if len(known) < 2:
            continue
        split_pt = max(1, len(known) // 2)
        seed   = set(known[:split_pt])
        target = set(known[split_pt:])
        if not target:
            continue
        recs = [p for p in top_popular if p not in seed]
        recall5_list.append(len(set(recs[:5])  & target) / len(target))
        recall10_list.append(len(set(recs[:10]) & target) / len(target))

    r5  = float(np.mean(recall5_list))  if recall5_list  else 0.0
    r10 = float(np.mean(recall10_list)) if recall10_list else 0.0
    print(f"  [{split_name} baseline] Recall@5={r5:.4f}  Recall@10={r10:.4f}")
    return r5, r10

bl_r5, bl_r10 = evaluate_popularity_baseline(test_tids, "test")
print(f"\n  Co-occ v2 vs 베이스라인 (test):")
print(f"  Recall@5 : {test_r5:.4f} vs {bl_r5:.4f}  "
      f"({'↑' if test_r5 > bl_r5 else '↓'}{abs(test_r5 - bl_r5):.4f})")


# ── 샘플 추천 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: 샘플 추천 결과 (카테고리 부스트 포함)")
print("=" * 65)

top5 = [idx2place[i] for i in np.argsort(place_cnt)[::-1][:5]]
print(f"  가장 많이 방문된 Top-5: {top5}")

for seed_place in top5[:3]:
    recs = recommend_v2([seed_place], top_n=5, category_boost=True)
    if recs:
        idx = place2idx[seed_place]
        slat, slon = float(place_lat[idx]), float(place_lon[idx])
        scat = place_cat[idx]
        print(f"\n  '{seed_place}' (카테고리={scat}) 방문 후 추천:")
        for rank, (p, score) in enumerate(recs, 1):
            pidx = place2idx[p]
            plat, plon = float(place_lat[pidx]), float(place_lon[pidx])
            pcat = place_cat[pidx]
            dist = haversine_km(slat, slon, plat, plon)
            same = "★" if pcat == scat else "  "
            print(f"    {rank}. {same}{p} (카테고리={pcat}, score={score:.4f}, "
                  f"거리={dist:.1f}km)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: 저장
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 8: 모델 저장")
print("=" * 65)

model_data = {
    "place2idx":  place2idx,
    "idx2place":  idx2place,
    "place_list": place_list,
    "co_occ":     co_occ,
    "jaccard":    jaccard,
    "place_cnt":  place_cnt,
    "place_lat":  place_lat,
    "place_lon":  place_lon,
    "place_cat":  place_cat,   # v2 추가: 카테고리 배열
    "meta": {
        "vocab":                   VOCAB,
        "min_trip_freq":           args.min_trip_freq,
        "max_dist_km":             args.max_dist_km,
        "category_boost_factor":   args.category_boost_factor,
        "exclude_type_ge":         args.exclude_type_ge,
        "n_trips_train":           len(train_tids),
        "n_trips_val":             len(val_tids),
        "n_trips_test":            len(test_tids),
        "val_recall5":             round(val_r5, 4),
        "val_recall10":            round(val_r10, 4),
        "test_recall5":            round(test_r5, 4),
        "test_recall10":           round(test_r10, 4),
        "test_recall5_no_boost":   round(test_r5_nb, 4),
        "baseline_test_recall5":   round(bl_r5, 4),
        "baseline_test_recall10":  round(bl_r10, 4),
        "model_version":           "v2",
    }
}

with open(MODEL_PKL, "wb") as f:
    pickle.dump(model_data, f)
print(f"  ✅ poi_cooccurrence_v2.pkl → {MODEL_PKL}")

with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(model_data["meta"], f, ensure_ascii=False, indent=2)
print(f"  ✅ poi_rec_meta_v2.json   → {META_JSON}")

print("\n" + "=" * 65)
print("✅ build_poi_recommender_v2.py 완료")
print(f"   test Recall@5={test_r5:.4f}  (부스트 없이={test_r5_nb:.4f})")
print(f"   베이스라인 @5={bl_r5:.4f}")
print("=" * 65)
