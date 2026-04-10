# -*- coding: utf-8 -*-
"""
streamlit_kride.py
K-Ride 자전거 경로 안전 분석 & 추천 앱

실행: streamlit run kride-project/streamlit_kride.py
"""

import datetime
import math
import os
import pickle
import sys

# ── Hugging Face Hub 연동 ─────────────────────────────────────────────────────
# Streamlit Cloud에서 모델 파일 자동 다운로드
# 로컬에 파일이 있으면 로컬 우선 사용 (개발 환경 호환)
try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False

# ★ 본인의 HF Hub 레포 ID로 변경하세요 (upload_to_hf.py 실행 시 --repo 인자와 동일)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# .env 로드 (python-dotenv 있으면 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# weather_kma 임포트 (같은 디렉토리)
sys.path.insert(0, os.path.dirname(__file__))
try:
    from weather_kma import get_weather_weight, get_current_weather
    HAS_WEATHER = True
except ImportError:
    HAS_WEATHER = False

try:
    from build_consume_model import predict_consume
    HAS_CONSUME = True
except ImportError:
    HAS_CONSUME = False

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ─────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────
def set_korean_font():
    candidates = [
        # Linux (Streamlit Cloud) — fonts-nanum 패키지
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 폴백: 시스템에서 한글 지원 폰트 자동 탐색
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ("Nanum", "Gothic", "Malgun", "CJK")):
            plt.rcParams["font.family"] = f.name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DATA_PATH       = os.path.join(BASE_DIR, "data", "raw_ml", "road_scored.csv")
GRAPH_PATH      = os.path.join(MODELS_DIR, "route_graph.pkl")
FACILITY_PATH   = os.path.join(BASE_DIR, "data", "raw_ml", "facility_clean.csv")
POI_REC_PATH    = os.path.join(MODELS_DIR, "poi_cooccurrence.pkl")
POI_META_PATH   = os.path.join(MODELS_DIR, "poi_rec_meta.json")
DISTRICT_PATH   = os.path.join(BASE_DIR, "data", "raw_ml", "district_danger.csv")


def _hf_path(local_path: str, hf_filename: str) -> str:
    """
    로컬 파일이 있으면 그대로 반환.
    없으면 HF Hub에서 다운로드 후 캐시 경로 반환.
    HF_REPO_ID 미설정 시 로컬 경로 그대로 반환 (FileNotFoundError 자연 발생).
    """
    if os.path.exists(local_path):
        return local_path
    if not HAS_HF or not HF_REPO_ID:
        return local_path  # HF 미설정 → 로컬 없으면 로드 시 오류 발생
    try:
        token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            repo_type="dataset",
            token=token,
        )
    except Exception as e:
        st.warning(f"HF Hub 다운로드 실패 ({hf_filename}): {e}")
        return local_path

# ─────────────────────────────────────────────
# 리소스 로드 (캐시)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf     = joblib.load(os.path.join(MODELS_DIR, "safety_classifier.pkl"))
    reg     = joblib.load(os.path.join(MODELS_DIR, "safety_regressor.pkl"))
    scaler  = joblib.load(os.path.join(MODELS_DIR, "safety_scaler.pkl"))
    meta    = joblib.load(os.path.join(MODELS_DIR, "safety_meta.pkl"))
    return clf, reg, scaler, meta

@st.cache_resource
def load_models():
    """안전 모델 4개 로드 (로컬 없으면 HF Hub에서 다운로드)"""
    clf    = joblib.load(_hf_path(os.path.join(MODELS_DIR, "safety_classifier.pkl"), "models/safety_classifier.pkl"))
    reg    = joblib.load(_hf_path(os.path.join(MODELS_DIR, "safety_regressor.pkl"),  "models/safety_regressor.pkl"))
    scaler = joblib.load(_hf_path(os.path.join(MODELS_DIR, "safety_scaler.pkl"),     "models/safety_scaler.pkl"))
    meta   = joblib.load(_hf_path(os.path.join(MODELS_DIR, "safety_meta.pkl"),       "models/safety_meta.pkl"))
    return clf, reg, scaler, meta

@st.cache_resource
def load_graph():
    """route_graph.pkl 로드 (로컬 없으면 HF Hub에서 다운로드)"""
    if not HAS_NX:
        return None, None
    path = _hf_path(GRAPH_PATH, "models/route_graph.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("G_main"), data.get("meta", {})

@st.cache_resource
def load_poi_rec():
    """poi_cooccurrence.pkl + poi_rec_meta.json 로드 (로컬 없으면 HF Hub에서 다운로드)
    sorted_places 를 캐시 안에서 미리 계산 -> 매 렌더링 argsort 비용 제거 (이슈 D)
    """
    import json
    path = _hf_path(POI_REC_PATH, "models/poi_cooccurrence.pkl")
    if not os.path.exists(path):
        return None, {}
    with open(path, "rb") as f:
        rec_data = pickle.load(f)
    meta_poi = {}
    meta_path = _hf_path(POI_META_PATH, "models/poi_rec_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_poi = json.load(f)
    # 미리 인기도 순 정렬 (좌표 있는 장소만)
    _p2i  = rec_data["place2idx"]
    _plat = rec_data["place_lat"]
    _plon = rec_data["place_lon"]
    _pcnt = rec_data["place_cnt"]
    _i2p  = {v: k for k, v in _p2i.items()}
    rec_data["sorted_places"] = [
        _i2p[i]
        for i in np.argsort(_pcnt)[::-1]
        if not (np.isnan(_plat[i]) or np.isnan(_plon[i]))
    ]
    rec_data["idx2place"] = _i2p
    return rec_data, meta_poi

@st.cache_data
def load_data():
    """road_scored.csv 로드 (로컬 없으면 HF Hub에서 다운로드)"""
    path = _hf_path(DATA_PATH, "data/road_scored.csv")
    return pd.read_csv(path)

@st.cache_data
def load_facility():
    """facility_clean.csv 로드 (로컬 없으면 HF Hub에서 다운로드)"""
    path = _hf_path(FACILITY_PATH, "data/facility_clean.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="cp949")

@st.cache_data
def load_district():
    """district_danger.csv 로드 (로컬 없으면 HF Hub에서 다운로드) (이슈 A)"""
    path = _hf_path(DISTRICT_PATH, "data/district_danger.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="cp949")

# ─────────────────────────────────────────────
# 날씨 API (30분 캐시)
# ─────────────────────────────────────────────
WEATHER_ICON = {
    "맑음":    "☀️",
    "구름많음": "⛅",
    "흐림":    "🌥️",
    "소나기":  "🌦️",
    "비":      "🌧️",
    "비/눈":   "🌨️",
    "눈":      "❄️",
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_cached(lat: float, lon: float, api_key: str):
    """날씨 API 호출 (30분 캐시)"""
    if not HAS_WEATHER or not api_key:
        return None
    try:
        w_safety, w_tourism, info = get_weather_weight(lat, lon, api_key=api_key)
        return w_safety, w_tourism, info
    except Exception as e:
        return str(e)


@st.cache_data(show_spinner=False)
def cached_predict_consume(distance_km: float, travel_duration_h: float,
                           companion_cnt: int, season: int, day_of_week: int) -> dict | None:
    """TabNet 소비 예측 (입력 파라미터 기준 캐시)"""
    if not HAS_CONSUME:
        return None
    return predict_consume(
        sgg_code=0,
        travel_duration_h=travel_duration_h,
        distance_km=distance_km,
        companion_cnt=companion_cnt,
        season=season,
        day_of_week=day_of_week,
        has_lodging=0,
    )


def haversine(c1, c2) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

SGG_COORDS = {
    # 서울 25개구
    "종로구": (37.5735, 126.9790), "중구":    (37.5641, 126.9979), "용산구":   (37.5311, 126.9810),
    "성동구": (37.5637, 127.0367), "광진구":  (37.5384, 127.0826), "동대문구": (37.5744, 127.0396),
    "중랑구": (37.6063, 127.0926), "성북구":  (37.5894, 127.0167), "도봉구":   (37.6688, 127.0470),
    "노원구": (37.6560, 126.9495), "은평구":  (37.6026, 126.9291), "서대문구": (37.5794, 126.9368),
    "마포구": (37.5663, 126.9014), "양천구":  (37.5270, 126.8561), "강서구":   (37.5509, 126.8496),
    "구로구": (37.4954, 126.8874), "영등포구":(37.5150, 126.9066), "동작구":   (37.4965, 126.9518),
    "관악구": (37.4784, 126.9516), "서초구":  (37.4836, 127.0327), "강남구":   (37.5173, 127.0473),
    "송파구": (37.5145, 127.1066), "강동구":  (37.5301, 127.1238),
    # 경기 주요 시
    "수원시": (37.2636, 127.0286), "하남시":  (37.5398, 127.2149), "성남시":   (37.4138, 127.1327),
    "안양시": (37.3943, 126.9568), "부천시":  (37.5034, 126.7660), "시흥시":   (37.3800, 126.8030),
    "광명시": (37.4794, 126.8644), "고양시":  (37.6584, 126.8320), "의정부시": (37.7382, 127.0337),
    "파주시": (37.7598, 126.9800), "구리시":  (37.5943, 127.1397), "남양주시": (37.6347, 127.2143),
    "김포시": (37.6152, 126.7155), "양주시":  (37.7852, 127.0456), "가평군":   (37.8316, 127.5099),
    "이천시": (37.2724, 127.4347), "양평군":  (37.4917, 127.4879),
}

def nearest_node(graph, lat, lon):
    target = (lat, lon)
    return min(graph.nodes, key=lambda n: haversine(n, target))




def _guess_sgg_name(lat: float, lon: float) -> str:
    """차중 좌표로 가장 가까운 구(구) 이름 반환 (haversine 기반)"""
    best_name, best_dist = "한국", float("inf")
    for name, (slat, slon) in SGG_COORDS.items():
        d = haversine((lat, lon), (slat, slon))
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


@st.cache_data
def build_location_names(data: pd.DataFrame) -> pd.Series:
    """데이터프레임의 start_lat/start_lon 에서 구 이름 추론 (\ucee8 \ub370이터 1회 실행 \ud6c4 \ucee8 \uce90시)"""
    return data.apply(
        lambda r: _guess_sgg_name(r["start_lat"], r["start_lon"]),
        axis=1,
    )

clf, reg, scaler, meta = load_models()
df = load_data()

G_main, graph_meta = load_graph()
df_facility = load_facility()
district_df = load_district()
poi_rec_data, poi_rec_meta = load_poi_rec()

FEATURES      = meta["features"]          # ['width_m', 'length_km', 'district_danger', 'road_attr_score']
DANGER_LABEL  = {0: "안전", 1: "보통", 2: "위험"}
DANGER_COLOR  = {0: "🟢", 1: "🟡", 2: "🔴"}

ROAD_TYPE_MAP = {
    "자전거전용도로": "자전거전용도로",
    "자전거보행자겸용도로": "자전거보행자겸용도로",
    "자전거우선도로": "자전거우선도로",
    "자전거전용차로": "자전거전용차로",
}

# ─────────────────────────────────────────────
# 페이지 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="K-Ride 안전 경로 분석",
    page_icon="🚴",
    layout="wide",
)

# ─────────────────────────────────────────────
# 사이드바 — 모드 선택
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚴 K-Ride")
    st.caption("자전거 안전 경로 분석 & 추천")
    st.divider()

    mode = st.radio(
        "경로 추천 모드",
        options=["safe", "balanced", "tourist"],
        format_func=lambda x: {
            "safe":     "🛡️ 안전 우선",
            "balanced": "⚖️ 균형",
            "tourist":  "🗺️ 관광 우선",
        }[x],
        index=1,
    )

    st.divider()

    # ── 날씨 섹션 (이슈 E: KMA_KEY 없으면 경고 없이 숨김) ────────────────
    KMA_KEY = os.environ.get("KMA_API_KEY", "")

    if HAS_WEATHER and KMA_KEY:
        st.subheader("🌤️ 현재 날씨")
        # 기본 위치: 서울 시청
        w_lat = st.number_input("위도", value=37.5665, format="%.4f", key="w_lat",
                                help="현재 위치 위도 (기본: 서울 시청)")
        w_lon = st.number_input("경도", value=126.9780, format="%.4f", key="w_lon",
                                help="현재 위치 경도 (기본: 서울 시청)")

        weather_auto = st.toggle("날씨 기반 안전 가중치 자동 조정", value=True,
                                 help="강수확률·풍속에 따라 w_safety를 자동 상향합니다")

        with st.spinner("날씨 조회 중…"):
            weather_result = fetch_weather_cached(w_lat, w_lon, KMA_KEY)

        if weather_result is None:
            st.info("날씨 정보를 불러올 수 없습니다.")
            weather_auto = False
        elif isinstance(weather_result, str):
            st.error(f"날씨 오류: {weather_result}")
            weather_auto = False
        else:
            w_s_adj, w_t_adj, winfo = weather_result
            label = winfo.get("weather_label", "맑음")
            icon  = WEATHER_ICON.get(label, "🌡️")

            col_w1, col_w2 = st.columns(2)
            col_w1.metric("날씨", f"{icon} {label}")
            col_w1.metric("기온", f"{winfo.get('tmp', '-')} °C")
            col_w2.metric("강수확률", f"{winfo.get('pop', 0)} %")
            col_w2.metric("풍속", f"{winfo.get('wsd', 0)} m/s")

            if weather_auto:
                st.caption(
                    f"⚙️ 자동 보정 → 안전 {w_s_adj:.0%} / 관광 {w_t_adj:.0%}"
                )
            st.caption("(30분 캐시)")

    st.divider()
    st.caption(f"데이터: {len(df):,}개 도로 세그먼트")
    st.caption(f"모델 성능  R²={meta['r2_regressor']}  F1={meta['f1_classifier']}")

# ─────────────────────────────────────────────
# 모드별 가중치
# ─────────────────────────────────────────────
MODE_WEIGHTS = {
    "safe":     {"safety": 0.7, "tourism": 0.3},
    "balanced": {"safety": 0.5, "tourism": 0.5},
    "tourist":  {"safety": 0.3, "tourism": 0.7},
}

# 날씨 자동 보정 가중치 결정
_base = MODE_WEIGHTS[mode].copy()
try:
    if (
        HAS_WEATHER
        and weather_auto  # noqa: F821 — 사이드바에서 설정됨
        and isinstance(weather_result, tuple)  # noqa: F821
    ):
        _base["safety"]  = float(w_s_adj)   # noqa: F821
        _base["tourism"] = float(w_t_adj)    # noqa: F821
except NameError:
    pass  # weather_auto / weather_result 미정의 시 무시

EFFECTIVE_WEIGHTS = _base

def compute_route_score(row, mode: str) -> float:
    w = EFFECTIVE_WEIGHTS
    return row["safety_score"] * w["safety"] + row["tourism_score"] * w["tourism"]

df["route_score"] = df.apply(compute_route_score, axis=1, mode=mode)

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 안전등급 예측", "📍 경로 추천 Top-10", "📊 데이터 탐색", "🗺️ 경로 탐색", "🏛️ 관광지 추천"])


# ══════════════════════════════════════════════
# 탭 1: 안전등급 예측
# ══════════════════════════════════════════════
with tab1:
    st.header("도로 안전등급 예측")
    st.caption("도로 정보를 입력하면 AI 모델이 안전등급을 예측합니다.")

    col1, col2 = st.columns(2)

    with col1:
        width_m = st.slider(
            "도로 너비 (m)",
            min_value=0.5, max_value=6.0, value=2.0, step=0.5,
            help="자전거도로의 너비(m)"
        )
        length_km = st.slider(
            "도로 길이 (km)",
            min_value=0.1, max_value=30.0, value=3.0, step=0.1,
            help="도로 세그먼트의 총 길이(km)"
        )

    with col2:
        # ── 이슈 A: district_danger.csv 로드 → 시군구 selectbox 교체 ──────
        if district_df is not None and ("sigungu" in district_df.columns or "시군구명" in district_df.columns):
            _sgg_col = "sigungu" if "sigungu" in district_df.columns else "시군구명"
            _sgg_names = ["직접 입력"] + district_df[_sgg_col].dropna().tolist()
            _sgg_sel = st.selectbox(
                "시·군·구 선택",
                _sgg_names,
                help="지역을 선택하면 해당 위험도가 자동 반영됩니다.",
            )
            if _sgg_sel == "직접 입력":
                district_danger = st.slider(
                    "구(시군구) 위험도",
                    min_value=0.0, max_value=1.0, value=0.25, step=0.05,
                    help="해당 지역의 자전거 사고 발생 위험도 (0=안전, 1=위험)"
                )
            else:
                _danger_val = district_df.loc[
                    district_df[_sgg_col] == _sgg_sel, "danger_score"
                ].values
                district_danger = float(_danger_val[0]) if len(_danger_val) else 0.25
                st.metric(
                    "해당 지역 위험도", f"{district_danger:.3f}",
                    help="build_safety_model.py 가 계산한 구별 사고 위험 점수"
                )
        else:
            district_danger = st.slider(
                "구(시군구) 위험도",
                min_value=0.0, max_value=1.0, value=0.25, step=0.05,
                help="해당 지역의 자전거 사고 발생 위험도 (0=안전, 1=위험)"
            )

        # road_attr_score는 너비·길이로 자동 계산
        width_norm  = min(width_m  / 6.0, 1.0)
        length_norm = min(length_km / 30.0, 1.0)
        road_attr_score = width_norm * 0.7 + length_norm * 0.3
        st.metric("도로 속성 점수 (자동 계산)", f"{road_attr_score:.3f}",
                  help="너비(70%) + 길이(30%) 정규화 합산")

    if st.button("안전등급 예측", type="primary", width="stretch"):
        input_df = pd.DataFrame([[width_m, length_km, district_danger, road_attr_score]],
                                 columns=FEATURES)
        input_scaled = scaler.transform(input_df)

        pred_level = int(clf.predict(input_scaled)[0])
        pred_score = float(reg.predict(input_scaled)[0])
        proba      = clf.predict_proba(input_scaled)[0]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("안전등급",
                  f"{DANGER_COLOR[pred_level]} {DANGER_LABEL[pred_level]}")
        c2.metric("안전 점수 (회귀)", f"{pred_score:.4f}")
        c3.metric("예측 신뢰도",
                  f"{proba[pred_level]*100:.1f}%")

        # 클래스별 확률 바
        st.caption("등급별 예측 확률")
        prob_df = pd.DataFrame({
            "등급": [f"{DANGER_COLOR[i]} {DANGER_LABEL[i]}" for i in range(3)],
            "확률": proba,
        })
        st.bar_chart(prob_df.set_index("등급"), width="stretch")


# ══════════════════════════════════════════════
# 탭 2: 경로 추천 Top-10
# ══════════════════════════════════════════════
with tab2:
    mode_label = {
        "safe":     "🛡️ 안전 우선",
        "balanced": "⚖️ 균형",
        "tourist":  "🗺️ 관광 우선",
    }[mode]
    st.header(f"경로 추천 Top-10  —  {mode_label} 모드")

    # ── 이슈 B: 순위 기준 설명 추가 ──────────────────────────────────────
    st.info(
        f"📊 **순위 기준**: `route_score = 안전점수 × {EFFECTIVE_WEIGHTS['safety']:.1f} + "
        f"관광점수 × {EFFECTIVE_WEIGHTS['tourism']:.1f}`  "
        f"| **1위가 현재 모드({mode_label})에서 가장 추천되는 경로입니다.**"
    )

    top10 = (
        df.sort_values("route_score", ascending=False)
          .head(10)
          .reset_index(drop=True)
    )
    top10.index += 1  # 1-based rank

    # ── 경로 위치명 컬럼 생성: 구명 + 도로유형 조합 ─────────────────────────
    _loc_names = build_location_names(df.sort_values("route_score", ascending=False).head(10).reset_index(drop=True))
    _road_types = df.sort_values("route_score", ascending=False).head(10)["road_type"].reset_index(drop=True)
    _road_type_short = {
        "자전거전용도로":       "전용",
        "자전거보행자겸용도로":  "겸용",
        "자전거우선도로":       "우선",
        "자전거전용차로":       "전용차로",
    }
    top10["경로 위치"] = [
        f"{_loc_names.iloc[i]} "
        f"{_road_type_short.get(str(_road_types.iloc[i]), '자전거도로')}"
        for i in range(len(top10))
    ]

    # ── 이슈 B + 이름표시: 경로 위치 컬럼 맨 앞에 추가 ─────────────────────
    display_cols = {"경로 위치": "경로 위치"}
    if "노선명" in df.columns:
        display_cols["노선명"] = "도로명"
    display_cols.update({
        "route_score":    "추천 점수",
        "safety_score":   "안전 점수",
        "tourism_score":  "관광 점수",
        "width_m":        "너비(m)",
        "length_km":      "길이(km)",
        "tourist_count":  "관광지 수",
        "cultural_count": "문화시설 수",
        "facility_count": "편의시설 수",
        "start_lat":      "시작 위도",
        "start_lon":      "시작 경도",
    })

    _top10_cols = [c for c in display_cols.keys() if c in top10.columns]
    display_df = top10[_top10_cols].rename(columns=display_cols)

    # 수치 포맷
    for col in ["추천 점수", "안전 점수", "관광 점수"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:.4f}".format)

    # ── 이슈 B: column_config 로 hover 설명 추가 ─────────────────────────
    _col_cfg = {
        "경로 위치": st.column_config.TextColumn(
            "경로 위치",
            help="출발 좌표 기준 가장 가까운 행정구역 + 도로 유형",
        ),
        "추천 점수": st.column_config.TextColumn(
            "추천 점수",
            help=f"route_score = 안전 × {EFFECTIVE_WEIGHTS['safety']:.1f} + 관광 × {EFFECTIVE_WEIGHTS['tourism']:.1f}",
        ),
        "관광지 수": st.column_config.NumberColumn(
            "관광지 수", help="도로 1km 반경 내 관광지 개수"
        ),
        "문화시설 수": st.column_config.NumberColumn(
            "문화시설 수", help="도로 1km 반경 내 문화시설 개수"
        ),
        "편의시설 수": st.column_config.NumberColumn(
            "편의시설 수", help="도로 1km 반경 내 편의시설(공기주입소 등) 개수"
        ),
    }
    st.dataframe(display_df, column_config=_col_cfg, use_container_width=True)

    # 추천 점수 분포 (전체 vs Top10)
    st.subheader("추천 점수 분포")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df["route_score"], bins=40, color="#4DA6FF", alpha=0.7, label="전체")
    ax.axvline(top10["route_score"].min(), color="#FF6B6B", lw=2,
               linestyle="--", label="Top-10 경계")
    ax.set_xlabel("추천 점수")
    ax.set_ylabel("세그먼트 수")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════
# 탭 3: 데이터 탐색
# ══════════════════════════════════════════════
with tab3:
    st.header("데이터 탐색")

    # 요약 지표
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 세그먼트", f"{len(df):,}")
    m2.metric("안전 점수 평균", f"{df['safety_score'].mean():.3f}")
    m3.metric("관광 점수 평균", f"{df['tourism_score'].mean():.3f}")
    _zero_pct = (df["tourism_score"] == 0).sum() / len(df) * 100
    m4.metric("관광점수 0 비율", f"{_zero_pct:.1f}%")

    # ── 이슈 C: 관광 점수가 낮은 이유 설명 ───────────────────────────────
    with st.expander("📌 관광 점수 평균이 낮은 이유 (정상 동작)"):
        _tour_mean = df["tourism_score"].mean()
        _safe_mean = df["safety_score"].mean()
        st.markdown(f"""
        > **안전 점수 평균 {_safe_mean:.3f} vs 관광 점수 평균 {_tour_mean:.3f}** — 이 차이는 정상입니다.

        관광 점수(`tourism_score`)는 도로 세그먼트 **1km 반경 내 POI 밀도**를
        `MinMaxScaler`로 정규화한 값입니다.

        | 원인 | 내용 |
        |------|------|
        | POI 희소 분포 | 전체 POI 2,529개가 1,647개 세그먼트에 희소하게 배치 |
        | 레저스포츠 POI 부족 | 전체 83개 (경기도 타임아웃으로 미수집) |
        | MinMaxScaler 특성 | 관광지 없는 {_zero_pct:.1f}% 세그먼트 → 점수 **0** 으로 수렴 |

        **평균 {_tour_mean:.3f}은 정상 결과입니다.**  
        관광지 인근 경로만 높은 점수를 받는 구조이며, `route_score`에서는
        안전 점수와 합산되어 경로 추천에 활용됩니다.

        > **근본 개선 방향**: 경기도 레저스포츠 POI 재수집 + 반경 2km 확대 시 분포 개선 가능.
        """)

    st.divider()

    col_left, col_right = st.columns(2)

    # 안전 점수 히스토그램
    with col_left:
        st.subheader("안전 점수 분포")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.hist(df["safety_score"], bins=30, color="#4DA6FF", edgecolor="white")
        ax1.set_xlabel("safety_score")
        ax1.set_ylabel("count")
        q33, q66 = meta["q33"], meta["q66"] # [메모] meta, q33, q66은 어떤 의미인가요 ? 
        # [메모] axvline은 어떤 의미인가요 ? s
        ax1.axvline(q33, color="#FF6B6B", lw=1.5, linestyle="--", label=f"q33={q33:.3f}")
        ax1.axvline(q66, color="#FFA500", lw=1.5, linestyle="--", label=f"q66={q66:.3f}")
        ax1.legend(fontsize=8)
        st.pyplot(fig1)
        plt.close(fig1)

    # 관광 점수 히스토그램
    with col_right:
        st.subheader("관광 점수 분포")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        nonzero = df[df["tourism_score"] > 0]["tourism_score"]
        ax2.hist(nonzero, bins=30, color="#57CC99", edgecolor="white")
        ax2.set_xlabel("tourism_score (0 제외)")
        ax2.set_ylabel("count")
        ax2.set_title(f"0 값 제외 세그먼트: {len(nonzero):,}개")
        st.pyplot(fig2)
        plt.close(fig2)

    # 안전 vs 관광 산점도
    st.subheader("안전 점수 vs 관광 점수")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    scatter = ax3.scatter(
        df["safety_score"], df["tourism_score"],
        c=df["route_score"], cmap="RdYlGn",
        alpha=0.4, s=10,
    )
    plt.colorbar(scatter, ax=ax3, label="추천 점수")
    ax3.set_xlabel("safety_score")
    ax3.set_ylabel("tourism_score")
    st.pyplot(fig3)
    plt.close(fig3)

    # 피처 통계
    st.subheader("주요 피처 통계")
    stat_cols = ["width_m", "length_km", "tourist_count",
                 "cultural_count", "facility_count", "safety_score", "tourism_score", "final_score"]
    st.dataframe(df[stat_cols].describe().T.style.format("{:.3f}"),
                 width="stretch")


# ══════════════════════════════════════════════
# 탭 4: 경로 탐색
# ══════════════════════════════════════════════
with tab4:
    st.header("경로 탐색")

    if not HAS_NX or G_main is None:
        st.warning(
            "route_graph.pkl이 없습니다.  \n"
            "`python kride-project/build_route_graph.py` 를 먼저 실행하세요."
        )
        if not HAS_NX:
            st.info("networkx가 설치되어 있지 않습니다: `pip install networkx`")
    else:
        st.caption(
            f"그래프: {graph_meta.get('main_nodes', '?'):,} 노드 / "
            f"{graph_meta.get('main_edges', '?'):,} 엣지"
        )

        # ── 모드 선택 ──────────────────────────────────
        route_mode = st.radio(
            "탐색 모드",
            options=["route", "course"],
            format_func=lambda x: "📍 A→B 최적 경로" if x == "route" else "🔄 순환 코스 생성",
            horizontal=True,
        )
        st.divider()

        col_l, col_r = st.columns(2)

        if route_mode == "route":
            # ── A→B 경로 탐색 ─────────────────────────
            with col_l:
                st.subheader("출발지")
                s_lat = st.number_input("출발 위도",  value=37.5665, format="%.6f", key="s_lat")
                s_lon = st.number_input("출발 경도",  value=126.9780, format="%.6f", key="s_lon")

            with col_r:
                st.subheader("도착지")
                e_lat = st.number_input("도착 위도",  value=37.5172, format="%.6f", key="e_lat")
                e_lon = st.number_input("도착 경도",  value=127.0473, format="%.6f", key="e_lon")

            w_s = st.slider("안전 가중치", 0.1, 0.9, 0.6, 0.1, key="r_ws")
            w_t = round(1.0 - w_s, 1)
            st.caption(f"관광 가중치: {w_t}")
            companion_cnt_r = st.number_input("동반자 수 (본인 포함)", min_value=1, max_value=20, value=1, step=1, key="r_companion")

            if st.button("경로 탐색", type="primary"):
                import networkx as nx_rt

                G_work = G_main.copy()
                for u, v, d in G_work.edges(data=True):
                    score = w_s * d.get("safety_score", 0.5) + w_t * d.get("tourism_score", 0.5)
                    d["weight"] = max(1.0 - score, 1e-6)

                start_node = nearest_node(G_work, s_lat, s_lon)
                end_node   = nearest_node(G_work, e_lat, e_lon)

                try:
                    path_nodes = nx_rt.shortest_path(G_work, source=start_node, target=end_node, weight="weight")
                except (nx_rt.NetworkXNoPath, nx_rt.NodeNotFound):
                    st.error("경로를 찾을 수 없습니다. 출발/도착 좌표를 조정해 보세요.")
                    path_nodes = []

                if path_nodes:
                    total_dist = 0.0
                    safety_sum = tourism_sum = edge_cnt = 0.0
                    for i in range(len(path_nodes) - 1):
                        u, v = path_nodes[i], path_nodes[i + 1]
                        if G_work.has_edge(u, v):
                            d = G_work[u][v]
                            total_dist  += d.get("length_km", haversine(u, v))
                            safety_sum  += d.get("safety_score", 0.5)
                            tourism_sum += d.get("tourism_score", 0.5)
                            edge_cnt    += 1

                    avg_s = safety_sum  / edge_cnt if edge_cnt else 0.0
                    avg_t = tourism_sum / edge_cnt if edge_cnt else 0.0

                    _today = datetime.date.today()
                    _month = _today.month
                    _season = 1 if _month in (3,4,5) else 2 if _month in (6,7,8) else 3 if _month in (9,10,11) else 4
                    _dow = _today.weekday()
                    _dur_h = round(total_dist / 15, 2)   # 평균 자전거 속도 15 km/h
                    consume_result = cached_predict_consume(
                        round(total_dist, 2), _dur_h, int(companion_cnt_r), _season, _dow
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("총 거리", f"{total_dist:.2f} km")
                    m2.metric("평균 안전 점수", f"{avg_s:.3f}")
                    m3.metric("평균 관광 점수", f"{avg_t:.3f}")
                    if consume_result and "estimated_cost_krw" in consume_result:
                        cost_krw = consume_result["estimated_cost_krw"]
                        m4.metric("예상 지출 (참고)", f"{cost_krw:,} 원",
                                  help=f"TabNet 추정 (MAE ±{consume_result.get('model_mae_krw',0):,}원 / R²={consume_result.get('model_r2',0):.3f})")
                    elif not HAS_CONSUME:
                        m4.metric("예상 지출", "모델 미설치")

                    # [메모] folium은 어떤 라이브러리인가요 ? 
                    # folium 지도
                    if HAS_FOLIUM:
                        center_lat = (s_lat + e_lat) / 2
                        center_lon = (s_lon + e_lon) / 2
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

                        coords = [[n[0], n[1]] for n in path_nodes]
                        folium.PolyLine(
                            coords, color="#2563EB", weight=4, opacity=0.8,
                            tooltip=f"경로 ({total_dist:.2f} km)"
                        ).add_to(m)
                        folium.Marker([s_lat, s_lon],
                                      icon=folium.Icon(color="green", icon="play"),
                                      tooltip="출발").add_to(m)
                        folium.Marker([e_lat, e_lon],
                                      icon=folium.Icon(color="red", icon="stop"),
                                      tooltip="도착").add_to(m)

                        # 편의시설 마커
                        if df_facility is not None:
                            lat_col = next((c for c in ["lat", "latitude", "위도", "y"] if c in df_facility.columns), None)
                            lon_col = next((c for c in ["lon", "longitude", "경도", "x"] if c in df_facility.columns), None)
                            name_col = next((c for c in ["name", "시설명", "명칭"] if c in df_facility.columns), None)
                            if lat_col and lon_col:
                                fac_layer = folium.FeatureGroup(name="편의시설")
                                for _, row in df_facility.iterrows():
                                    try:
                                        fc = (float(row[lat_col]), float(row[lon_col]))
                                        if any(haversine(fc, (n[0], n[1])) <= 0.5 for n in path_nodes[::5]):
                                            folium.CircleMarker(
                                                location=list(fc),
                                                radius=5, color="#F59E0B",
                                                fill=True, fill_opacity=0.8,
                                                tooltip=row.get(name_col, "편의시설") if name_col else "편의시설",
                                            ).add_to(fac_layer)
                                    except (ValueError, TypeError):
                                        pass
                                fac_layer.add_to(m)
                                folium.LayerControl().add_to(m)

                        st_folium(m, width=700, height=450)
                    else:
                        st.info("지도 표시를 위해 folium을 설치하세요: `pip install folium streamlit-folium`")
                        st.write("경로 노드 수:", len(path_nodes))

        else:
            # ── 순환 코스 생성 ────────────────────────
            st.subheader("시작점")
            c_lat = st.number_input("시작 위도",  value=37.5665, format="%.6f", key="c_lat")
            c_lon = st.number_input("시작 경도",  value=126.9780, format="%.6f", key="c_lon")
            target_km = st.slider("목표 거리 (km)", 5, 50, 20, 5)
            w_s_c = st.slider("안전 가중치", 0.1, 0.9, 0.6, 0.1, key="c_ws")
            w_t_c = round(1.0 - w_s_c, 1)
            st.caption(f"관광 가중치: {w_t_c}")
            companion_cnt_c = st.number_input("동반자 수 (본인 포함)", min_value=1, max_value=20, value=1, step=1, key="c_companion")

            if st.button("코스 생성", type="primary"):
                G_work = G_main.copy()
                for u, v, d in G_work.edges(data=True):
                    score = w_s_c * d.get("safety_score", 0.5) + w_t_c * d.get("tourism_score", 0.5)
                    d["weight"] = max(1.0 - score, 1e-6)

                start_node = nearest_node(G_work, c_lat, c_lon)
                best_course, best_dist = [start_node], 0.0
                stack = [(start_node, [start_node], 0.0)]
                visited = set()
                MAX_ITER = 30_000
                iters = 0

                while stack and iters < MAX_ITER:
                    iters += 1
                    node, path, dist = stack.pop()
                    if dist >= target_km * 0.9 and dist > best_dist:
                        best_dist, best_course = dist, path
                    if dist >= target_km:
                        break
                    neighbors = sorted(
                        [n for n in G_work.neighbors(node) if n not in visited],
                        key=lambda n: -G_work[node][n].get("final_score", 0),
                    )
                    for nb in neighbors[:5]:
                        edge = G_work[node][nb]
                        new_dist = dist + edge.get("length_km", haversine(node, nb))
                        if new_dist <= target_km * 1.2:
                            visited.add(nb)
                            stack.append((nb, path + [nb], new_dist))

                _today_c = datetime.date.today()
                _month_c = _today_c.month
                _season_c = 1 if _month_c in (3,4,5) else 2 if _month_c in (6,7,8) else 3 if _month_c in (9,10,11) else 4
                _dur_h_c = round(best_dist / 15, 2)
                course_consume = cached_predict_consume(
                    round(best_dist, 2), _dur_h_c, int(companion_cnt_c), _season_c, _today_c.weekday()
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("생성된 코스 거리", f"{best_dist:.2f} km")
                m2.metric("경유 노드 수", f"{len(best_course):,}")
                if course_consume and "estimated_cost_krw" in course_consume:
                    m3.metric("예상 지출 (참고)", f"{course_consume['estimated_cost_krw']:,} 원",
                              help=f"TabNet 추정 (MAE ±{course_consume.get('model_mae_krw',0):,}원)")
                elif not HAS_CONSUME:
                    m3.metric("예상 지출", "모델 미설치")

                if HAS_FOLIUM and len(best_course) > 1:
                    m = folium.Map(location=[c_lat, c_lon], zoom_start=13)
                    coords = [[n[0], n[1]] for n in best_course]
                    folium.PolyLine(
                        coords, color="#16A34A", weight=4, opacity=0.8,
                        tooltip=f"순환 코스 ({best_dist:.2f} km)"
                    ).add_to(m)
                    folium.Marker(
                        [c_lat, c_lon],
                        icon=folium.Icon(color="blue", icon="home"),
                        tooltip="시작/종료",
                    ).add_to(m)

                    # 코스 주변 POI 추천 오버레이
                    if poi_rec_data is not None:
                        _p2i  = poi_rec_data["place2idx"]
                        _jac  = poi_rec_data["jaccard"]
                        _plat = poi_rec_data["place_lat"]
                        _plon = poi_rec_data["place_lon"]
                        _pcnt = poi_rec_data["place_cnt"]
                        _i2p  = {v: k for k, v in _p2i.items()}
                        VOCAB = len(_p2i)
                        # 코스 경유 노드 중심 계산
                        _course_lats = [n[0] for n in best_course]
                        _course_lons = [n[1] for n in best_course]
                        _cx = float(np.mean(_course_lats))
                        _cy = float(np.mean(_course_lons))
                        # 인기도 기준 Top-5 (코스 경로 중심 20km 이내)
                        _scores = _pcnt.copy().astype(float)
                        for j in range(VOCAB):
                            if np.isnan(_plat[j]) or np.isnan(_plon[j]):
                                _scores[j] = -1.0
                            elif haversine((_cx, _cy), (float(_plat[j]), float(_plon[j]))) > 20.0:
                                _scores[j] = -1.0
                        _top_poi_idx = np.argsort(_scores)[::-1][:5]
                        poi_layer = folium.FeatureGroup(name="주변 관광지 추천 Top-5")
                        for rank, pidx in enumerate(_top_poi_idx):
                            if _scores[pidx] < 0:
                                continue
                            plat_v, plon_v = float(_plat[pidx]), float(_plon[pidx])
                            pname = _i2p.get(pidx, f"POI-{pidx}")
                            folium.Marker(
                                location=[plat_v, plon_v],
                                icon=folium.Icon(color="orange", icon="star"),
                                tooltip=f"#{rank+1} {pname}",
                                popup=folium.Popup(
                                    f"<b>{pname}</b><br>방문 빈도: {int(_scores[pidx])}회",
                                    max_width=200,
                                ),
                            ).add_to(poi_layer)
                        poi_layer.add_to(m)
                        folium.LayerControl().add_to(m)

                    st_folium(m, width=700, height=450)
                elif not HAS_FOLIUM:
                    st.info("지도 표시를 위해 folium을 설치하세요: `pip install folium streamlit-folium`")
                    st.write("코스 노드 수:", len(best_course))


# ══════════════════════════════════════════════
# 탭 5: 관광지 추천 & 지도
# ══════════════════════════════════════════════
with tab5:
    st.header("관광지 추천")

    if poi_rec_data is None:
        st.warning(
            "poi_cooccurrence.pkl이 없습니다.  \n"
            "`python kride-project/build_poi_recommender.py` 를 먼저 실행하세요."
        )
    else:
        # ── 이슈 D: 로딩 스피너 (첫 진입 시 사용자 피드백) + 캐시 활용 ──────
        with st.spinner("관광지 추천 모델 데이터 준비 중…"):
            # 캐시에서 미리 계산된 값 사용 (매 렌더링 argsort 제거)
            _p2i_t5      = poi_rec_data["place2idx"]
            _jac_t5      = poi_rec_data["jaccard"]
            _plat_t5     = poi_rec_data["place_lat"]
            _plon_t5     = poi_rec_data["place_lon"]
            _pcnt_t5     = poi_rec_data["place_cnt"]
            _i2p_t5      = poi_rec_data.get("idx2place") or {v: k for k, v in _p2i_t5.items()}
            VOCAB_T5     = len(_p2i_t5)
            _sorted_places = poi_rec_data.get("sorted_places") or [
                _i2p_t5[i]
                for i in np.argsort(_pcnt_t5)[::-1]
                if not (np.isnan(_plat_t5[i]) or np.isnan(_plon_t5[i]))
            ]

        # 성능 지표 (메타 파일)
        if poi_rec_meta:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("vocab 크기", f"{poi_rec_meta.get('vocab', VOCAB_T5):,}개")
            mc2.metric("test Recall@5",  f"{poi_rec_meta.get('test_recall5',  0):.4f}")
            mc3.metric("test Recall@10", f"{poi_rec_meta.get('test_recall10', 0):.4f}")
            mc4.metric("베이스라인 @5",  f"{poi_rec_meta.get('baseline_recall5', 0):.4f}")
            st.caption("Co-occurrence + Jaccard 정규화 | VISIT_AREA_TYPE_CD < 21 관광지만")
        st.divider()

        # ── 입력 컬럼 ─────────────────────────────────
        col_input, col_map = st.columns([1, 2])

        with col_input:
            st.subheader("시드(방문) 장소 선택")
            seed_sel = st.multiselect(
                "이미 방문했거나 기준이 될 장소 (1개 이상)",
                options=_sorted_places,
                default=_sorted_places[:1],
                help="인기도 순으로 정렬됨. 여러 장소 선택 가능.",
            )
            max_dist_sel = st.slider("추천 반경 (km)", 5, 50, 20, 5, key="poi_dist")
            top_n_sel    = st.slider("추천 개수",       3, 20, 10, 1, key="poi_topn")
            show_route   = st.toggle(
                "추천 장소 → 자전거 경로 연결",
                value=False,
                help="route_graph.pkl이 있을 때 추천 장소를 순서대로 Dijkstra로 연결합니다.",
                disabled=(G_main is None),
            )
            run_btn = st.button("추천 실행", type="primary", key="poi_run")

        # ── 추천 함수 (인라인) ──────────────────────────
        def _poi_recommend(seed_places, top_n, max_dist_km):
            """Co-occurrence Jaccard 기반 추천. seed 좌표 중심 max_dist_km 이내 필터."""
            seed_idx = [_p2i_t5[p] for p in seed_places if p in _p2i_t5]
            exclude  = set(seed_places)

            # 시드 중심 좌표
            lats = [float(_plat_t5[i]) for i in seed_idx if not np.isnan(_plat_t5[i])]
            lons = [float(_plon_t5[i]) for i in seed_idx if not np.isnan(_plon_t5[i])]
            center_lat = float(np.mean(lats)) if lats else None
            center_lon = float(np.mean(lons)) if lons else None

            if not seed_idx:
                scores = _pcnt_t5.copy().astype(float)
            else:
                scores = np.array(_jac_t5[seed_idx].sum(axis=0)).flatten()

            for j in range(VOCAB_T5):
                if _i2p_t5.get(j) in exclude:
                    scores[j] = -1.0
                    continue
                if np.isnan(_plat_t5[j]) or np.isnan(_plon_t5[j]):
                    scores[j] = -1.0
                    continue
                if center_lat is not None and max_dist_km > 0:
                    d = haversine(
                        (center_lat, center_lon),
                        (float(_plat_t5[j]), float(_plon_t5[j])),
                    )
                    if d > max_dist_km:
                        scores[j] = -1.0

            top_idx = np.argsort(scores)[::-1][:top_n]
            results = []
            for i in top_idx:
                if scores[i] < 0:
                    continue
                plat_v = float(_plat_t5[i])
                plon_v = float(_plon_t5[i])
                dist_km = (
                    haversine((center_lat, center_lon), (plat_v, plon_v))
                    if center_lat is not None else None
                )
                results.append({
                    "장소":       _i2p_t5[i],
                    "Jaccard":    round(float(scores[i]), 4),
                    "거리(km)":   round(dist_km, 1) if dist_km is not None else "-",
                    "lat":        plat_v,
                    "lon":        plon_v,
                })
            return results

        # ── 실행 결과 ──────────────────────────────────
        if run_btn:
            if not seed_sel:
                st.warning("시드 장소를 1개 이상 선택해 주세요.")
            else:
                recs = _poi_recommend(seed_sel, top_n_sel, max_dist_sel)

                with col_input:
                    st.divider()
                    if recs:
                        st.success(f"추천 결과 {len(recs)}개")
                        rec_df = pd.DataFrame(recs)[["장소", "Jaccard", "거리(km)"]]
                        rec_df.index += 1
                        st.dataframe(rec_df, width="stretch")
                    else:
                        st.info(f"반경 {max_dist_sel}km 이내 추천 결과가 없습니다. 반경을 늘려 보세요.")

                with col_map:
                    if not HAS_FOLIUM:
                        st.info("`pip install folium streamlit-folium` 설치 필요")
                    else:
                        # 중심 좌표 계산 (시드 + 추천 전체)
                        all_lats, all_lons = [], []
                        for p in seed_sel:
                            if p in _p2i_t5:
                                i = _p2i_t5[p]
                                if not np.isnan(_plat_t5[i]):
                                    all_lats.append(float(_plat_t5[i]))
                                    all_lons.append(float(_plon_t5[i]))
                        for r in recs:
                            all_lats.append(r["lat"])
                            all_lons.append(r["lon"])
                        map_center = (
                            [float(np.mean(all_lats)), float(np.mean(all_lons))]
                            if all_lats else [37.5665, 126.9780]
                        )
                        m5 = folium.Map(location=map_center, zoom_start=12)

                        # 시드 마커 (초록)
                        seed_layer = folium.FeatureGroup(name="시드 장소")
                        for p in seed_sel:
                            if p not in _p2i_t5:
                                continue
                            i = _p2i_t5[p]
                            if np.isnan(_plat_t5[i]):
                                continue
                            folium.Marker(
                                location=[float(_plat_t5[i]), float(_plon_t5[i])],
                                icon=folium.Icon(color="green", icon="flag"),
                                tooltip=p,
                                popup=folium.Popup(f"<b>[시드] {p}</b>", max_width=200),
                            ).add_to(seed_layer)
                        seed_layer.add_to(m5)

                        # 추천 마커 (빨강, 크기 = Jaccard 비례)
                        rec_layer = folium.FeatureGroup(name="추천 관광지")
                        if recs:
                            max_jac = max(r["Jaccard"] for r in recs) or 1.0
                            for rank, r in enumerate(recs):
                                radius = 8 + int(r["Jaccard"] / max_jac * 14)
                                dist_txt = f"{r['거리(km)']} km" if r["거리(km)"] != "-" else ""
                                folium.CircleMarker(
                                    location=[r["lat"], r["lon"]],
                                    radius=radius,
                                    color="#DC2626",
                                    fill=True,
                                    fill_color="#FCA5A5",
                                    fill_opacity=0.8,
                                    tooltip=f"#{rank+1} {r['장소']} (Jaccard={r['Jaccard']:.4f})",
                                    popup=folium.Popup(
                                        f"<b>#{rank+1} {r['장소']}</b><br>"
                                        f"Jaccard: {r['Jaccard']:.4f}<br>"
                                        f"거리: {dist_txt}",
                                        max_width=220,
                                    ),
                                ).add_to(rec_layer)
                                # 순번 텍스트 마커
                                folium.Marker(
                                    location=[r["lat"], r["lon"]],
                                    icon=folium.DivIcon(
                                        html=f'<div style="font-size:11px;font-weight:bold;'
                                             f'color:#DC2626;text-shadow:1px 1px 0 #fff">#{rank+1}</div>',
                                        icon_size=(30, 20),
                                        icon_anchor=(0, 10),
                                    ),
                                ).add_to(rec_layer)
                        rec_layer.add_to(m5)

                        # 자전거 경로 연결 (route_graph 사용)
                        if show_route and G_main is not None and recs:
                            import networkx as nx_poi
                            waypoints = []
                            for p in seed_sel:
                                if p in _p2i_t5:
                                    i = _p2i_t5[p]
                                    if not np.isnan(_plat_t5[i]):
                                        waypoints.append((float(_plat_t5[i]), float(_plon_t5[i])))
                            for r in recs[:5]:  # 상위 5개만 경유
                                waypoints.append((r["lat"], r["lon"]))

                            route_layer = folium.FeatureGroup(name="자전거 경로")
                            route_total_km = 0.0
                            for wi in range(len(waypoints) - 1):
                                sn = nearest_node(G_main, waypoints[wi][0],    waypoints[wi][1])
                                en = nearest_node(G_main, waypoints[wi+1][0], waypoints[wi+1][1])
                                try:
                                    seg_nodes = nx_poi.shortest_path(G_main, sn, en, weight="weight")
                                    seg_coords = [[n[0], n[1]] for n in seg_nodes]
                                    seg_km = sum(
                                        G_main[seg_nodes[k]][seg_nodes[k+1]].get("length_km",
                                            haversine(seg_nodes[k], seg_nodes[k+1]))
                                        for k in range(len(seg_nodes) - 1)
                                    )
                                    route_total_km += seg_km
                                    folium.PolyLine(
                                        seg_coords, color="#7C3AED", weight=3, opacity=0.75,
                                        tooltip=f"구간 {wi+1}→{wi+2} ({seg_km:.1f} km)",
                                    ).add_to(route_layer)
                                except Exception:
                                    pass
                            route_layer.add_to(m5)
                            if route_total_km > 0:
                                st.caption(f"자전거 경로 총 거리 (추정): {route_total_km:.1f} km")

                        folium.LayerControl().add_to(m5)
                        st_folium(m5, width="100%", height=520)
        else:
            with col_map:
                # 버튼 누르기 전 기본 지도 (서울 중심)
                if HAS_FOLIUM:
                    m5_default = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
                    # 전체 관광지 히트맵 대신 인기 Top-20 마커 표시
                    from folium.plugins import MarkerCluster
                    cluster = MarkerCluster(name="인기 관광지 Top-20")
                    for i in np.argsort(_pcnt_t5)[::-1][:20]:
                        if np.isnan(_plat_t5[i]) or np.isnan(_plon_t5[i]):
                            continue
                        folium.CircleMarker(
                            location=[float(_plat_t5[i]), float(_plon_t5[i])],
                            radius=6, color="#2563EB", fill=True,
                            fill_opacity=0.7,
                            tooltip=_i2p_t5.get(i, ""),
                        ).add_to(cluster)
                    cluster.add_to(m5_default)
                    folium.LayerControl().add_to(m5_default)
                    st_folium(m5_default, width="100%", height=520)
