"""
upload_to_hf.py
===============================================================
K-Ride 모델 파일을 Hugging Face Hub Dataset 레포에 업로드하는
1회성 스크립트입니다.

[ 사용법 ]
1. pip install huggingface_hub
2. huggingface-cli login   ← HF 토큰 입력
3. python kride-project/upload_to_hf.py --repo YOUR_HF_USERNAME/kride-models

[ 업로드 대상 파일 ]
  models/route_graph.pkl          (69 MB) — 경로 탐색 그래프
  models/poi_cooccurrence.pkl     (20 MB) — 관광지 추천 Co-occurrence
  models/safety_regressor.pkl     ( 1 MB) — 안전 회귀 모델
  models/safety_classifier.pkl    ( 0.3 MB) — 안전 분류 모델
  models/safety_scaler.pkl        — 스케일러
  models/safety_meta.pkl          — 메타
  models/tourism_scaler.pkl       — 관광 스케일러
  models/attraction_regressor.zip — TabNet POI 매력도
  models/attraction_scaler.pkl
  models/attraction_meta.json
  models/consume_regressor.zip    — TabNet 소비 예측
  models/consume_scaler.pkl
  models/consume_meta.json
  models/poi_rec_meta.json
  models/dl/weather_lstm.pt       — LSTM 날씨 예측
  models/dl/weather_scaler.pkl
  models/dl/weather_meta.json
  data/raw_ml/road_scored.csv     — 도로 점수 데이터
  data/raw_ml/tour_poi.csv        — 관광 POI
  data/raw_ml/facility_clean.csv  — 편의시설
  data/raw_ml/district_danger.csv — 지역 위험도

[ 제외 파일 ]
  osm_bike_cache.graphml (175MB) — OSM 캐시, 앱 실행에 불필요
"""

import argparse
import os
import sys

# ── 의존성 확인 ────────────────────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub가 설치되어 있지 않습니다.")
    print("   pip install huggingface_hub")
    sys.exit(1)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath("kride-project")

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw_ml")

# ── 업로드 대상 (로컬 경로 → HF Hub 내 경로) ──────────────────────────────────
UPLOAD_FILES = [
    # 모델 파일
    (os.path.join(MODELS_DIR, "route_graph.pkl"),          "models/route_graph.pkl"),
    (os.path.join(MODELS_DIR, "poi_cooccurrence.pkl"),     "models/poi_cooccurrence.pkl"),
    (os.path.join(MODELS_DIR, "safety_regressor.pkl"),     "models/safety_regressor.pkl"),
    (os.path.join(MODELS_DIR, "safety_classifier.pkl"),    "models/safety_classifier.pkl"),
    (os.path.join(MODELS_DIR, "safety_scaler.pkl"),        "models/safety_scaler.pkl"),
    (os.path.join(MODELS_DIR, "safety_meta.pkl"),          "models/safety_meta.pkl"),
    (os.path.join(MODELS_DIR, "tourism_scaler.pkl"),       "models/tourism_scaler.pkl"),
    (os.path.join(MODELS_DIR, "attraction_regressor.zip"), "models/attraction_regressor.zip"),
    (os.path.join(MODELS_DIR, "attraction_scaler.pkl"),    "models/attraction_scaler.pkl"),
    (os.path.join(MODELS_DIR, "attraction_meta.json"),     "models/attraction_meta.json"),
    (os.path.join(MODELS_DIR, "consume_regressor.zip"),    "models/consume_regressor.zip"),
    (os.path.join(MODELS_DIR, "consume_scaler.pkl"),       "models/consume_scaler.pkl"),
    (os.path.join(MODELS_DIR, "consume_meta.json"),        "models/consume_meta.json"),
    (os.path.join(MODELS_DIR, "poi_rec_meta.json"),        "models/poi_rec_meta.json"),
    # DL 모델
    (os.path.join(MODELS_DIR, "dl", "weather_lstm.pt"),   "models/dl/weather_lstm.pt"),
    (os.path.join(MODELS_DIR, "dl", "weather_scaler.pkl"),"models/dl/weather_scaler.pkl"),
    (os.path.join(MODELS_DIR, "dl", "weather_meta.json"), "models/dl/weather_meta.json"),
    # 데이터 파일
    (os.path.join(DATA_DIR, "road_scored.csv"),            "data/road_scored.csv"),
    (os.path.join(DATA_DIR, "tour_poi.csv"),               "data/tour_poi.csv"),
    (os.path.join(DATA_DIR, "facility_clean.csv"),         "data/facility_clean.csv"),
    (os.path.join(DATA_DIR, "district_danger.csv"),        "data/district_danger.csv"),
]


def main():
    parser = argparse.ArgumentParser(description="Upload K-Ride models to HF Hub")
    parser.add_argument(
        "--repo",
        default="",
        help="HF Hub 레포 이름 (예: your-username/kride-models)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="HF 토큰 (미입력 시 환경변수 HF_TOKEN 또는 huggingface-cli login 사용)",
    )
    args = parser.parse_args()

    if not args.repo:
        print("❌ --repo 인자가 필요합니다.")
        print("   예: python upload_to_hf.py --repo your-username/kride-models")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN", None)
    api   = HfApi(token=token)

    # ── 레포 생성 (없으면 자동 생성) ──────────────────────────────────────────
    print(f"\n📦 HF Hub 레포 확인/생성: {args.repo}")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            exist_ok=True,
            token=token,
        )
        print(f"   ✅ 레포 준비 완료: https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"   ⚠️  레포 생성 실패 (이미 존재하거나 권한 문제): {e}")

    # ── 파일 업로드 ────────────────────────────────────────────────────────────
    print(f"\n📤 파일 업로드 시작 ({len(UPLOAD_FILES)}개)")
    success, skipped, failed = 0, 0, 0

    for local_path, hf_path in UPLOAD_FILES:
        if not os.path.exists(local_path):
            size_str = "파일 없음"
            print(f"   ⏭️  SKIP  {hf_path}  ({size_str})")
            skipped += 1
            continue

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"   ⬆️  업로드 중  {hf_path}  ({size_mb:.1f} MB) ...", end="", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path,
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=f"Upload {hf_path}",
            )
            print("  ✅")
            success += 1
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            failed += 1

    # ── 요약 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"업로드 완료: ✅ {success}개  ⏭️ {skipped}개 스킵  ❌ {failed}개 실패")
    print(f"\n다음 단계:")
    print(f"  1. streamlit_kride.py에서 HF_REPO_ID = \"{args.repo}\" 로 설정")
    print(f"  2. pip install huggingface_hub 를 requirements.txt에 추가")
    print(f"  3. Streamlit Cloud Secrets에 HF_TOKEN 추가 (private repo인 경우)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
