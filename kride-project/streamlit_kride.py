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
BASE_DIR      = os.path.dirname(__file__)
MODELS_DIR    = os.path.join(BASE_DIR, "models")
DATA_PATH     = os.path.join(BASE_DIR, "data", "raw_ml", "road_scored.csv")
GRAPH_PATH    = os.path.join(MODELS_DIR, "route_graph.pkl")
FACILITY_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "facility_clean.csv")

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

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_graph():
    """route_graph.pkl 로드 (없으면 None 반환)"""
    if not HAS_NX or not os.path.exists(GRAPH_PATH):
        return None, None
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    return data.get("G_main"), data.get("meta", {})

@st.cache_data
def load_facility():
    if not os.path.exists(FACILITY_PATH):
        return None
    try:
        return pd.read_csv(FACILITY_PATH, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(FACILITY_PATH, encoding="cp949")

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

def nearest_node(graph, lat, lon):
    target = (lat, lon)
    return min(graph.nodes, key=lambda n: haversine(n, target))

clf, reg, scaler, meta = load_models()
df = load_data()
G_main, graph_meta = load_graph()
df_facility = load_facility()

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

    # ── 날씨 섹션 ─────────────────────────────
    st.subheader("🌤️ 현재 날씨")

    KMA_KEY = os.environ.get("KMA_API_KEY", "")

    if not HAS_WEATHER:
        st.info("weather_kma.py 를 찾을 수 없습니다.")
    elif not KMA_KEY:
        st.warning("KMA_API_KEY 환경변수가 없습니다.")
    else:
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
tab1, tab2, tab3, tab4 = st.tabs(["🔍 안전등급 예측", "📍 경로 추천 Top-10", "📊 데이터 탐색", "🗺️ 경로 탐색"])


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
    st.caption(
        f"가중치: 안전 {MODE_WEIGHTS[mode]['safety']*100:.0f}% / "
        f"관광 {MODE_WEIGHTS[mode]['tourism']*100:.0f}%"
    )

    top10 = (
        df.sort_values("route_score", ascending=False)
          .head(10)
          .reset_index(drop=True)
    )
    top10.index += 1  # 1-based rank

    display_cols = {
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
    }

    display_df = top10[list(display_cols.keys())].rename(columns=display_cols)

    # 수치 포맷
    for col in ["추천 점수", "안전 점수", "관광 점수"]:
        display_df[col] = display_df[col].map("{:.4f}".format)

    st.dataframe(display_df, width="stretch")

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
    m4.metric("관광점수 0 비율",
              f"{(df['tourism_score']==0).sum()/len(df)*100:.1f}%")

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
                    st_folium(m, width=700, height=450)
                elif not HAS_FOLIUM:
                    st.info("지도 표시를 위해 folium을 설치하세요: `pip install folium streamlit-folium`")
                    st.write("코스 노드 수:", len(best_course))
