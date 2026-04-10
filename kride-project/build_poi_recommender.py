"""
build_poi_recommender.py
=============================================================
Co-occurrence 기반 방문지 추천 모델 (지리 필터 + 관광지 전용)

[ 개선사항 ]
  v2: X_COORD / Y_COORD 활용 지리적 거리 필터 추가
      VISIT_AREA_TYPE_CD 필터로 비관광 장소 (집, 사무실 등) 제외

[ 알고리즘 ]
  1. 여행별 방문 장소 집합 구성 (관광지 전용)
  2. 장소 쌍 (A, B)의 co-occurrence 카운트 집계
  3. Jaccard 유사도로 정규화: |A∩B| / |A∪B|
  4. 추천: 입력 장소 집합 → 시드 중심 반경 max_dist_km 이내 후보만 →
           Jaccard 합산 점수 → Top-N 반환

[ VISIT_AREA_TYPE_CD 분류 ]
  관광지: 1~20 (자연관광지, 역사관광지, 레저, 쇼핑, 음식점, 숙박 등)
  비관광: 21 (자택/지인집), 22 (직장/학교), 23 (기타) → 제외

[ 평가 방식 ]
  TRAVEL_ID 기준 70 / 20 / 10 분할
  각 test 여행에서 앞 50% 방문지 → 뒤 50% 예측
  Recall@5, Recall@10 측정

[ 출력 파일 ]
  models/poi_cooccurrence.pkl  ← co-occurrence 행렬 + Jaccard + 좌표
  models/poi_rec_meta.json     ← 메타 + 평가 결과
"""

import os
import sys
import json
import math
import pickle
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── ArgumentParser ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--min_trip_freq", type=int, default=2,
                    help="최소 여행 등장 횟수 (이 이하 장소는 추천 대상에서 제외)")
parser.add_argument("--top_n", type=int, default=10,
                    help="추천 반환 개수 (Recall@N 기준)")
parser.add_argument("--max_dist_km", type=float, default=20.0,
                    help="시드 위치 기준 추천 반경 (km). 0이면 거리 제한 없음")
parser.add_argument("--exclude_type_ge", type=int, default=21,
                    help="VISIT_AREA_TYPE_CD >= 이 값인 장소 제외 (기본 21: 자택/직장 제외)")
args = parser.parse_args()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

AIHUB_DIR = os.path.join(BASE_DIR, "data", "ai-hub",
                          "국내 여행로그 수도권_2023", "02.라벨링데이터")
MODELS_DIR = os.path.join(BASE_DIR, "models")

VISIT_CSV  = os.path.join(AIHUB_DIR, "tn_visit_area_info_방문지정보_E.csv")
MODEL_PKL  = os.path.join(MODELS_DIR, "poi_cooccurrence.pkl")
META_JSON  = os.path.join(MODELS_DIR, "poi_rec_meta.json")

os.makedirs(MODELS_DIR, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표 간 Haversine 거리 (km)"""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 + 좌표 및 타입 정보 추출
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: 데이터 로드 + 좌표 / 타입 정보 추출")
print("=" * 65)

if not os.path.exists(VISIT_CSV):
    print(f"  ❌ {VISIT_CSV} 없음.")
    sys.exit(1)

df = pd.read_csv(VISIT_CSV, encoding="utf-8-sig")
df = df[["TRAVEL_ID", "VISIT_ORDER", "VISIT_AREA_NM",
         "X_COORD", "Y_COORD", "VISIT_AREA_TYPE_CD"]].copy()
df = df.dropna(subset=["VISIT_AREA_NM"])
df["VISIT_AREA_NM"] = df["VISIT_AREA_NM"].astype(str).str.strip()
df = df[df["VISIT_AREA_NM"] != ""]
df["X_COORD"] = pd.to_numeric(df["X_COORD"], errors="coerce")
df["Y_COORD"] = pd.to_numeric(df["Y_COORD"], errors="coerce")
df["VISIT_AREA_TYPE_CD"] = pd.to_numeric(df["VISIT_AREA_TYPE_CD"], errors="coerce")
df = df.sort_values(["TRAVEL_ID", "VISIT_ORDER"]).reset_index(drop=True)

print(f"  전체 행수: {len(df):,} / 여행 수: {df['TRAVEL_ID'].nunique():,} / 고유 장소: {df['VISIT_AREA_NM'].nunique():,}")

# ── 비관광 장소 필터 (VISIT_AREA_TYPE_CD >= exclude_type_ge) ──────────────────
before = len(df)
df_tourist = df[
    df["VISIT_AREA_TYPE_CD"].isna() |           # 타입 정보 없는 경우 유지
    (df["VISIT_AREA_TYPE_CD"] < args.exclude_type_ge)
].copy()
removed_type = before - len(df_tourist)
print(f"  비관광 장소 제거 (type_cd >= {args.exclude_type_ge}): {removed_type:,}행 제거 → {len(df_tourist):,}행 잔존")
print(f"  잔존 고유 장소: {df_tourist['VISIT_AREA_NM'].nunique():,}개")

# ── 장소별 대표 좌표 계산 (중앙값) ────────────────────────────────────────────
place_coords = (
    df_tourist.groupby("VISIT_AREA_NM")[["Y_COORD", "X_COORD"]]
    .median()
    .rename(columns={"Y_COORD": "lat", "X_COORD": "lon"})
)
# 좌표가 있는 장소만 추출 (집, 사무실 등은 좌표 없음 → 자연 필터링)
place_coords = place_coords.dropna()
has_coord_set = set(place_coords.index)
print(f"  좌표 보유 장소: {len(has_coord_set):,}개 / 좌표 없는 장소(자동 제외): "
      f"{df_tourist['VISIT_AREA_NM'].nunique() - len(has_coord_set):,}개")

# ── 최종 데이터: 관광지 + 좌표 보유 장소만 ───────────────────────────────────
df_final = df_tourist[df_tourist["VISIT_AREA_NM"].isin(has_coord_set)].copy()
print(f"  최종 데이터: {len(df_final):,}행 / {df_final['VISIT_AREA_NM'].nunique():,}개 장소")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 여행별 방문 장소 집합 구성 + min_trip_freq 필터
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: 여행별 방문 장소 집합 + 빈도 필터")
print("=" * 65)

trip_sequences = {}
for tid, grp in df_final.groupby("TRAVEL_ID"):
    seq = grp["VISIT_AREA_NM"].tolist()
    if seq:
        trip_sequences[tid] = seq

# 장소별 등장 여행 수
place_trip_count = defaultdict(int)
for tid, seq in trip_sequences.items():
    for place in set(seq):
        place_trip_count[place] += 1

known_places = {p for p, cnt in place_trip_count.items() if cnt >= args.min_trip_freq}
print(f"  관광지+좌표 장소: {len(place_trip_count):,}개")
print(f"  min_trip_freq={args.min_trip_freq} 후 vocab: {len(known_places):,}개")

place_list = sorted(known_places)
place2idx  = {p: i for i, p in enumerate(place_list)}
idx2place  = {i: p for p, i in place2idx.items()}
VOCAB      = len(place_list)

# 장소별 좌표 배열 (vocab 순서)
place_lat = np.array([place_coords.loc[p, "lat"] if p in place_coords.index else np.nan
                       for p in place_list], dtype=np.float32)
place_lon = np.array([place_coords.loc[p, "lon"] if p in place_coords.index else np.nan
                       for p in place_list], dtype=np.float32)

print(f"  서울/수도권 위도 범위: {np.nanmin(place_lat):.3f} ~ {np.nanmax(place_lat):.3f}")
print(f"  서울/수도권 경도 범위: {np.nanmin(place_lon):.3f} ~ {np.nanmax(place_lon):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAVEL_ID 기준 70 / 20 / 10 분할
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: TRAVEL_ID 기준 train / val / test 분할")
print("=" * 65)

np.random.seed(42)
all_tids = np.array(list(trip_sequences.keys()))
np.random.shuffle(all_tids)

n       = len(all_tids)
n_train = int(n * 0.70)
n_val   = int(n * 0.20)

train_tids = set(all_tids[:n_train])
val_tids   = set(all_tids[n_train : n_train + n_val])
test_tids  = set(all_tids[n_train + n_val :])

print(f"  TRAVEL_ID — train: {len(train_tids):,} / val: {len(val_tids):,} / test: {len(test_tids):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Co-occurrence 행렬 구성 (train set만)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: Co-occurrence 행렬 구성 (train set)")
print("=" * 65)

co_occ    = np.zeros((VOCAB, VOCAB), dtype=np.float32)
place_cnt = np.zeros(VOCAB, dtype=np.float32)

for tid in train_tids:
    seq = trip_sequences[tid]
    known_in_trip = list({place2idx[p] for p in seq if p in place2idx})
    for idx_a in known_in_trip:
        place_cnt[idx_a] += 1
    for i, idx_a in enumerate(known_in_trip):
        for idx_b in known_in_trip[i + 1:]:
            co_occ[idx_a][idx_b] += 1
            co_occ[idx_b][idx_a] += 1

print(f"  co-occurrence 행렬: {VOCAB} × {VOCAB}")
print(f"  비어있지 않은 셀: {int((co_occ > 0).sum()):,}개")
print(f"  최대 공동 출현 횟수: {int(co_occ.max()):,}회")

# Jaccard 정규화
print("  Jaccard 정규화 중...")
jaccard = np.zeros((VOCAB, VOCAB), dtype=np.float32)
for i in range(VOCAB):
    for j in range(VOCAB):
        if i == j:
            continue
        denom = place_cnt[i] + place_cnt[j] - co_occ[i][j]
        if denom > 0:
            jaccard[i][j] = co_occ[i][j] / denom

print(f"  Jaccard 최대값: {jaccard.max():.4f}")
print(f"  Jaccard 평균값(비영): {jaccard[jaccard > 0].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: 추천 함수 (지리 필터 포함) + 평가
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 5: 추천 함수 (거리 제한 {args.max_dist_km}km) + 평가")
print("=" * 65)


def recommend(seed_places: list, top_n: int = 10,
              exclude: set = None, max_dist_km: float = None) -> list:
    """
    seed_places : 이미 방문한 장소명 리스트
    top_n       : 추천 개수
    exclude     : 제외할 장소명 집합
    max_dist_km : 시드 중심 반경 제한 (None이면 무제한)
    반환: [(장소명, 점수), ...]
    """
    if exclude is None:
        exclude = set(seed_places)
    if max_dist_km is None:
        max_dist_km = args.max_dist_km

    seed_idx = [place2idx[p] for p in seed_places if p in place2idx]

    # 시드 중심 좌표 계산
    center_lat, center_lon = None, None
    if seed_idx and max_dist_km > 0:
        lats = [place_lat[i] for i in seed_idx if not np.isnan(place_lat[i])]
        lons = [place_lon[i] for i in seed_idx if not np.isnan(place_lon[i])]
        if lats and lons:
            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))

    # 점수 계산
    if not seed_idx:
        scores = place_cnt.copy()
    else:
        scores = jaccard[seed_idx].sum(axis=0)

    # 거리 필터: 시드 중심 기준 max_dist_km 초과 장소 제거
    if center_lat is not None and center_lon is not None and max_dist_km > 0:
        for j in range(VOCAB):
            if np.isnan(place_lat[j]) or np.isnan(place_lon[j]):
                scores[j] = -1.0
                continue
            d = haversine_km(center_lat, center_lon, float(place_lat[j]), float(place_lon[j]))
            if d > max_dist_km:
                scores[j] = -1.0

    # 이미 방문한 장소 제외
    for p in exclude:
        if p in place2idx:
            scores[place2idx[p]] = -1.0

    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(idx2place[i], float(scores[i])) for i in top_idx if scores[i] >= 0]


def evaluate_split(tids: set, split_name: str):
    recall5_list, recall10_list = [], []
    valid_trips = 0

    for tid in tids:
        seq = trip_sequences.get(tid, [])
        known_seq = [p for p in seq if p in place2idx]
        if len(known_seq) < 2:
            continue

        split_pt = max(1, len(known_seq) // 2)
        seed   = known_seq[:split_pt]
        target = set(known_seq[split_pt:])
        if not target:
            continue

        recs5  = [p for p, _ in recommend(seed, top_n=5,  exclude=set(seed))]
        recs10 = [p for p, _ in recommend(seed, top_n=10, exclude=set(seed))]

        recall5_list.append(len(set(recs5) & target) / len(target))
        recall10_list.append(len(set(recs10) & target) / len(target))
        valid_trips += 1

    r5  = float(np.mean(recall5_list))  if recall5_list  else 0.0
    r10 = float(np.mean(recall10_list)) if recall10_list else 0.0
    print(f"  [{split_name}] valid_trips={valid_trips:,} | "
          f"Recall@5={r5:.4f} | Recall@10={r10:.4f}")
    return r5, r10


val_r5,  val_r10  = evaluate_split(val_tids,  "val ")
test_r5, test_r10 = evaluate_split(test_tids, "test")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: 인기도 베이스라인 비교 (거리 필터 동일 적용)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6: 인기도 베이스라인 비교")
print("=" * 65)

def evaluate_popularity_baseline(tids: set, split_name: str):
    recall5_list, recall10_list = [], []
    for tid in tids:
        seq = trip_sequences.get(tid, [])
        known_seq = [p for p in seq if p in place2idx]
        if len(known_seq) < 2:
            continue
        split_pt = max(1, len(known_seq) // 2)
        seed   = set(known_seq[:split_pt])
        target = set(known_seq[split_pt:])
        if not target:
            continue
        # popularity: 인기도 순 (거리 필터 동일 적용)
        recs_all = [p for p, _ in recommend(list(seed), top_n=10, exclude=seed,
                                             max_dist_km=0)]  # 거리 제한 없음
        top_popular_filtered = [p for p in sorted(place2idx.keys(),
                                 key=lambda x: place_cnt[place2idx[x]], reverse=True)
                                 if p not in seed]
        recs5  = top_popular_filtered[:5]
        recs10 = top_popular_filtered[:10]
        recall5_list.append(len(set(recs5) & target) / len(target))
        recall10_list.append(len(set(recs10) & target) / len(target))

    r5  = float(np.mean(recall5_list))  if recall5_list  else 0.0
    r10 = float(np.mean(recall10_list)) if recall10_list else 0.0
    print(f"  [{split_name} baseline] Recall@5={r5:.4f} | Recall@10={r10:.4f}")
    return r5, r10

bl_val_r5,  bl_val_r10  = evaluate_popularity_baseline(val_tids,  "val ")
bl_test_r5, bl_test_r10 = evaluate_popularity_baseline(test_tids, "test")

print(f"\n  Co-occurrence vs 인기도 베이스라인 (test, max_dist={args.max_dist_km}km)")
print(f"  Recall@5  : {test_r5:.4f} vs {bl_test_r5:.4f}  "
      f"({'↑' if test_r5 > bl_test_r5 else '↓'} {abs(test_r5-bl_test_r5):.4f})")
print(f"  Recall@10 : {test_r10:.4f} vs {bl_test_r10:.4f}  "
      f"({'↑' if test_r10 > bl_test_r10 else '↓'} {abs(test_r10-bl_test_r10):.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: 샘플 추천 출력 (관광지 Top-3 기준)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7: 샘플 추천 결과")
print("=" * 65)

top5_tourist = [idx2place[i] for i in np.argsort(place_cnt)[::-1][:5]]
print(f"  가장 많이 방문된 관광지 Top-5: {top5_tourist}")

for seed_place in top5_tourist[:3]:
    recs = recommend([seed_place], top_n=5)
    if recs:
        slat = place_lat[place2idx[seed_place]]
        slon = place_lon[place2idx[seed_place]]
        print(f"\n  '{seed_place}' (위도={slat:.4f}, 경도={slon:.4f}) 방문 후 추천 (반경 {args.max_dist_km}km):")
        for rank, (p, score) in enumerate(recs, 1):
            plat = place_lat[place2idx[p]]
            plon = place_lon[place2idx[p]]
            dist = haversine_km(float(slat), float(slon), float(plat), float(plon))
            print(f"    {rank}. {p} (Jaccard={score:.4f}, 거리={dist:.1f}km)")


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
    "meta": {
        "vocab":             VOCAB,
        "min_trip_freq":     args.min_trip_freq,
        "max_dist_km":       args.max_dist_km,
        "exclude_type_ge":   args.exclude_type_ge,
        "n_trips_train":     len(train_tids),
        "n_trips_val":       len(val_tids),
        "n_trips_test":      len(test_tids),
        "val_recall5":       round(val_r5, 4),
        "val_recall10":      round(val_r10, 4),
        "test_recall5":      round(test_r5, 4),
        "test_recall10":     round(test_r10, 4),
        "baseline_test_recall5":  round(bl_test_r5, 4),
        "baseline_test_recall10": round(bl_test_r10, 4),
    }
}

with open(MODEL_PKL, "wb") as f:
    pickle.dump(model_data, f)

with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(model_data["meta"], f, ensure_ascii=False, indent=2)

print(f"  ✅ poi_cooccurrence.pkl → {MODEL_PKL}")
print(f"  ✅ poi_rec_meta.json   → {META_JSON}")

print("\n" + "=" * 65)
print("✅ build_poi_recommender.py 완료")
print(f"   test Recall@5={test_r5:.4f}  Recall@10={test_r10:.4f}")
print(f"   (베이스라인 @5={bl_test_r5:.4f}  @10={bl_test_r10:.4f})")
print("=" * 65)
