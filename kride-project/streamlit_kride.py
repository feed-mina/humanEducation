# -*- coding: utf-8 -*-
"""
streamlit_kride.py
K-Ride 자전거 경로 안전 분석 & 추천 앱

실행: streamlit run kride-project/streamlit_kride.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",      # Windows 맑은 고딕
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw_ml", "road_scored.csv")

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

clf, reg, scaler, meta = load_models()
df = load_data()

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

def compute_route_score(row, mode: str) -> float:
    w = MODE_WEIGHTS[mode]
    return row["safety_score"] * w["safety"] + row["tourism_score"] * w["tourism"]

df["route_score"] = df.apply(compute_route_score, axis=1, mode=mode)

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 안전등급 예측", "📍 경로 추천 Top-10", "📊 데이터 탐색"])


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
        q33, q66 = meta["q33"], meta["q66"]
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
        ax2.set_title(f"비영(非零) 세그먼트: {len(nonzero):,}개")
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
