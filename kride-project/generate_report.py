# -*- coding: utf-8 -*-
"""
generate_report.py
K-Ride 딥러닝 산출물 보고서 생성 (16:9 슬라이드 형태)

실행: python kride-project/generate_report.py
출력: kride-project/K-Ride_보고서.pdf
"""

import os

from reportlab.lib.units import mm
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FONT_DIR   = os.path.join(os.path.dirname(BASE_DIR), "수업일지")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_PATH   = os.path.join(BASE_DIR, "K-Ride_보고서.pdf")

# ── 폰트 등록 ─────────────────────────────────────────────────────────────────
def register_fonts():
    fonts = {
        "KoPub-Light":  "KoPubDotumLight.ttf",
        "KoPub-Medium": "KoPubDotumMedium.ttf",
        "KoPub-Bold":   "KoPubDotumBold.ttf",
    }
    for name, fname in fonts.items():
        path = os.path.join(FONT_DIR, fname)
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))
        else:
            print(f"  ⚠️  폰트 없음: {path}")
            return False
    return True

HAS_KOPUB = register_fonts()
FONT_B = "KoPub-Bold"   if HAS_KOPUB else "Helvetica-Bold"
FONT_M = "KoPub-Medium" if HAS_KOPUB else "Helvetica"
FONT_L = "KoPub-Light"  if HAS_KOPUB else "Helvetica"

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
GREEN_DARK  = HexColor("#1A6B3C")
GREEN_MID   = HexColor("#2D8A55")
GREEN_LIGHT = HexColor("#E8F5EE")
GRAY_DARK   = HexColor("#2C2C2C")
GRAY_MID    = HexColor("#555555")
GRAY_LIGHT  = HexColor("#F4F4F4")
WHITE       = HexColor("#FFFFFF")
ACCENT      = HexColor("#F0A500")
BLUE_DARK   = HexColor("#1A5276")
PURPLE      = HexColor("#7D3C98")
ORANGE      = HexColor("#B7500A")
BLUE_LIGHT  = HexColor("#EAF4FB")

# ── 페이지 설정 (16:9 landscape A4) ──────────────────────────────────────────
W, H = landscape(A4)   # 841.9 × 595.3 pt
MARGIN = 28


# ══════════════════════════════════════════════════════════════════════════════
# 공통 드로잉 유틸
# ══════════════════════════════════════════════════════════════════════════════

def draw_border(c, color=GREEN_DARK, lw=6):
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    c.rect(lw / 2, lw / 2, W - lw, H - lw)


def draw_page_header(c, section_num, section_title):
    c.setFillColor(GRAY_MID)
    c.setFont(FONT_L, 9)
    c.drawString(MARGIN, H - MARGIN, f"{section_num}  {section_title}")
    c.setStrokeColor(GREEN_MID)
    c.setLineWidth(0.5)
    c.line(MARGIN, H - MARGIN - 4, W - MARGIN, H - MARGIN - 4)


def draw_slide_title(c, title, subtitle=""):
    y = H - MARGIN - 42
    c.setFillColor(GRAY_DARK)
    c.setFont(FONT_B, 24)
    c.drawCentredString(W / 2, y, title)
    if subtitle:
        c.setFont(FONT_L, 10)
        c.setFillColor(GRAY_MID)
        c.drawCentredString(W / 2, y - 20, subtitle)


def draw_card(c, x, y, w, h, bg=GRAY_LIGHT, radius=5):
    c.setFillColor(bg)
    c.setStrokeColor(HexColor("#DDDDDD"))
    c.setLineWidth(0.5)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def draw_metric_card(c, x, y, w, h, value, label, color=GREEN_DARK):
    draw_card(c, x, y, w, h, bg=WHITE)
    c.setFillColor(color)
    c.setFont(FONT_B, 20)
    c.drawCentredString(x + w / 2, y + h - 34, value)
    c.setFillColor(GRAY_MID)
    c.setFont(FONT_L, 8)
    c.drawCentredString(x + w / 2, y + h - 50, label)


def draw_green_box(c, x, y, w, h, text, font_size=9):
    c.setFillColor(GREEN_LIGHT)
    c.setStrokeColor(GREEN_MID)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_M, font_size)
    c.drawCentredString(x + w / 2, y + h / 2 - font_size / 2, text)


def draw_table(c, x, y, headers, rows, col_widths, row_height=18,
               header_bg=GREEN_DARK, stripe=True):
    total_w = sum(col_widths)
    # 헤더
    c.setFillColor(header_bg)
    c.rect(x, y, total_w, row_height, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 8)
    cx = x
    for h_txt, cw in zip(headers, col_widths):
        c.drawCentredString(cx + cw / 2, y + 5, h_txt)
        cx += cw
    # 데이터 행
    for ri, row in enumerate(rows):
        ry = y - (ri + 1) * row_height
        bg = HexColor("#F0F8F4") if (stripe and ri % 2 == 0) else WHITE
        c.setFillColor(bg)
        c.rect(x, ry, total_w, row_height, fill=1, stroke=0)
        c.setStrokeColor(HexColor("#E0E0E0"))
        c.setLineWidth(0.3)
        c.line(x, ry, x + total_w, ry)
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_M, 8)
        cx = x
        for cell, cw in zip(row, col_widths):
            c.drawCentredString(cx + cw / 2, ry + 5, str(cell))
            cx += cw
    # 외곽선
    c.setStrokeColor(GREEN_MID)
    c.setLineWidth(0.8)
    c.rect(x, y - len(rows) * row_height, total_w,
           row_height * (len(rows) + 1), fill=0, stroke=1)


def draw_bullet(c, x, y, text, font_size=9, color=GRAY_DARK):
    c.setFillColor(GREEN_MID)
    c.circle(x + 3, y + 3.5, 2.5, fill=1, stroke=0)
    c.setFillColor(color)
    c.setFont(FONT_M, font_size)
    c.drawString(x + 11, y, text)


# ══════════════════════════════════════════════════════════════════════════════
# 슬라이드별 함수
# ══════════════════════════════════════════════════════════════════════════════

def slide_cover(c):
    """표지"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # 상단 굵은 바
    c.setFillColor(GREEN_DARK)
    c.rect(0, H - 10, W, 10, fill=1, stroke=0)
    # 하단 굵은 바
    c.rect(0, 0, W, 10, fill=1, stroke=0)
    # 테두리
    draw_border(c, GREEN_DARK, lw=6)

    # 메인 제목
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_B, 42)
    c.drawCentredString(W / 2, H / 2 + 68, "K-Ride")
    c.setFillColor(GRAY_DARK)
    c.setFont(FONT_B, 20)
    c.drawCentredString(W / 2, H / 2 + 30, "서울 자전거 안전 경로 추천 시스템")

    # 부제 꺾쇠 박스
    bx, bw, bh = W / 2 - 200, 400, 34
    by = H / 2 - 16
    c.setStrokeColor(GREEN_DARK)
    c.setLineWidth(1.5)
    c.line(bx, by + bh, bx + 14, by + bh)
    c.line(bx, by + bh, bx, by)
    c.line(bx + bw - 14, by + bh, bx + bw, by + bh)
    c.line(bx + bw, by + bh, bx + bw, by)
    c.setFillColor(GRAY_MID)
    c.setFont(FONT_L, 11)
    c.drawCentredString(W / 2, by + 10, "딥러닝 기반 경로 최적화 · 관광지 추천 · 실시간 날씨 반영")

    # 기술 태그
    tags = ["TabNet", "Co-occurrence", "osmnx", "Streamlit", "Folium"]
    tw_tag = 72
    tx = W / 2 - (len(tags) * (tw_tag + 8)) / 2
    ty = H / 2 - 68
    for tag in tags:
        c.setFillColor(GREEN_LIGHT)
        c.setStrokeColor(GREEN_MID)
        c.setLineWidth(0.8)
        c.roundRect(tx, ty, tw_tag, 20, 4, fill=1, stroke=1)
        c.setFillColor(GREEN_DARK)
        c.setFont(FONT_M, 8.5)
        c.drawCentredString(tx + tw_tag / 2, ty + 6, tag)
        tx += tw_tag + 8

    # 발표자
    c.setFillColor(GRAY_MID)
    c.setFont(FONT_L, 10)
    c.drawCentredString(W / 2, MARGIN + 16, "발표자  feed-mina          2026.04")


def slide_contents(c):
    """목차"""
    c.setFillColor(GREEN_DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 26)
    c.drawString(MARGIN + 10, H - 65, "contents")
    c.setStrokeColor(WHITE)
    c.setLineWidth(1)
    c.line(MARGIN + 10, H - 70, MARGIN + 115, H - 70)

    items = [
        ("01", "Project Overview",   "프로젝트 목표 및 Why / How"),
        ("02", "Dataset Analysis",   "데이터 현황 및 전처리 특성"),
        ("03", "Model Architecture", "안전 · 관광 · 경로 · POI 추천 모듈"),
        ("04", "Model Performance",  "정량 지표 및 베이스라인 비교"),
        ("05", "POI Recommendation", "Co-occurrence 관광지 추천 결과"),
        ("06", "Service Demo",       "Streamlit 5탭 서비스 구성"),
        ("07", "Conclusion",         "결론 및 향후 계획"),
    ]
    y = H - 105
    for num, title, desc in items:
        c.setFillColor(HexColor("#FFFFFF18"))
        c.roundRect(MARGIN + 10, y - 8, W - MARGIN * 2 - 20, 26, 4, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont(FONT_B, 11)
        c.drawString(MARGIN + 20, y + 5, num)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 11)
        c.drawString(MARGIN + 54, y + 5, title)
        c.setFillColor(HexColor("#BBBBBB"))
        c.setFont(FONT_L, 9)
        c.drawString(MARGIN + 225, y + 5, desc)
        y -= 36


def slide_overview(c):
    """01 Project Overview"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "01", "Project Overview")
    draw_slide_title(c, "AI4I 2020 제조업 이상 탐지 → K-Ride 자전거 경로 추천",
                     "머신러닝 + 딥러닝 기반 경로 최적화 및 관광지 추천 시스템")

    boxes = [
        ("What", [
            "자전거 도로 안전 등급 예측 (RandomForest)",
            "관광 매력도 점수화 (TabNet v2)",
            "osmnx 기반 서울 전역 경로 탐색 (Dijkstra)",
            "Co-occurrence POI 관광지 추천",
        ]),
        ("Why", [
            "자전거 이용자 사고 위험 경로 회피 필요",
            "관광 활성화 + 안전을 동시에 고려한 경로 부재",
            "공공 데이터(한국관광공사·서울시)의 실용적 활용",
        ]),
        ("How", [
            "도로 안전 데이터 + 관광·편의시설 데이터 결합",
            "TabNet으로 비선형 관광 매력도 학습",
            "osmnx로 서울 자전거 도로 토폴로지 보장",
            "Streamlit + Folium 실시간 지도 서비스 구축",
        ]),
    ]

    col_w = (W - MARGIN * 2 - 20) / 3
    bh = 148
    bx = MARGIN + 10
    by = H - MARGIN - 68 - bh

    for box_title, bullets in boxes:
        draw_card(c, bx, by, col_w - 8, bh, bg=GREEN_LIGHT)
        c.setFillColor(GREEN_DARK)
        c.setFont(FONT_B, 13)
        c.drawString(bx + 12, by + bh - 22, box_title)
        c.setStrokeColor(GREEN_MID)
        c.setLineWidth(0.8)
        c.line(bx + 12, by + bh - 28, bx + col_w - 18, by + bh - 28)
        ty = by + bh - 46
        for b in bullets:
            draw_bullet(c, bx + 12, ty, b, font_size=8.5)
            ty -= 18
        bx += col_w


def slide_dataset(c):
    """02 Dataset Analysis"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "02", "Dataset Analysis")
    draw_slide_title(c, "데이터 현황 및 특성 분석",
                     "서울 자전거 도로 · 관광지 · 여행자 방문 데이터")

    # 데이터셋 테이블
    draw_table(
        c, MARGIN + 10, H - 110,
        ["데이터셋", "규모", "주요 컬럼"],
        [
            ["자전거도로 원본",     "57,418행",   "width_m, length_km, road_type"],
            ["도로 안전 점수",      "32,614행",   "safety_score, district_danger"],
            ["관광지 정보",         "8,813개소",  "관광지명, X/Y좌표, 유형코드"],
            ["여행자 방문지",       "21,384행",   "TRAVEL_ID, VISIT_AREA_NM"],
            ["OSM 자전거 네트워크", "30만+ 엣지", "서울 전역 topology 보장"],
        ],
        col_widths=[148, 90, 220], row_height=19,
    )

    # 통계 카드
    stats = [
        ("57,418", "원본 도로 세그먼트", GREEN_DARK),
        ("7,748",  "좌표 보유 관광지",   GREEN_MID),
        ("2,560",  "여행 ID (trips)",    BLUE_DARK),
        ("1,646",  "POI vocab",          ORANGE),
    ]
    cw_s, ch_s = 90, 58
    gap_s = 8
    total_sw = len(stats) * cw_s + (len(stats) - 1) * gap_s
    sx = MARGIN + 480
    sy = H - 112
    for i, (val, lbl, col) in enumerate(stats):
        xi = sx + (i % 2) * (cw_s + gap_s)
        yi = sy - (i // 2) * (ch_s + gap_s)
        draw_metric_card(c, xi, yi, cw_s, ch_s, val, lbl, col)

    # 전처리 박스
    yb = MARGIN + 42
    draw_card(c, MARGIN + 10, yb, W - MARGIN * 2 - 20, 52, bg=GREEN_LIGHT)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_B, 9.5)
    c.drawString(MARGIN + 20, yb + 38, "전처리 핵심 사항")
    pts = [
        "POI_ID 결측(6,710행) → VISIT_AREA_NM 기반 대체",
        "비관광 장소(VISIT_AREA_TYPE_CD ≥ 21) 6,904행 제거 (자택·직장 등)",
        "osmnx KDTree로 OSM 엣지 ↔ road_scored 최근접 매핑 (허용 2km 이내)",
    ]
    px = MARGIN + 20
    py = yb + 22
    for pt in pts:
        draw_bullet(c, px, py, pt, font_size=8.5)
        py -= 15


def slide_model_arch(c):
    """03 Model Architecture"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "03", "Model Architecture")
    draw_slide_title(c, "모델 구성 — 4개 모듈 파이프라인",
                     "안전 모델 → 관광 모델 → 경로 그래프 → POI 추천")

    modules = [
        {
            "title": "안전 모델",
            "sub":   "RandomForest Classifier / Regressor",
            "color": GREEN_DARK,
            "items": [
                "입력: width_m, length_km",
                "district_danger, road_attr_score",
                "출력: 안전등급(0·1·2) + 안전점수",
                "성능: R²=0.91  F1=0.82",
            ],
        },
        {
            "title": "관광 매력도",
            "sub":   "TabNet v2 (pytorch-tabnet)",
            "color": BLUE_DARK,
            "items": [
                "입력: 주변 관광·문화·편의시설 수",
                "거리 기반 가중 집계",
                "출력: tourism_score (0~1)",
                "성능: MAE=0.041  R²=0.87",
            ],
        },
        {
            "title": "경로 그래프",
            "sub":   "osmnx + NetworkX Dijkstra",
            "color": PURPLE,
            "items": [
                "서울 자전거 도로 네트워크 다운로드",
                "KDTree 점수 매핑 (2km 이내)",
                "최대 연결 컴포넌트 추출",
                "weight = 1 - final_score",
            ],
        },
        {
            "title": "POI 추천",
            "sub":   "Co-occurrence + Jaccard",
            "color": ORANGE,
            "items": [
                "trip 단위 co-visit 행렬 구성",
                "Jaccard 정규화 유사도",
                "Haversine 거리 필터 (20km)",
                "Recall@5 = 0.1260 (베이스라인 3.4배)",
            ],
        },
    ]

    mw = (W - MARGIN * 2 - 20) / 4
    mh = 148
    mx = MARGIN + 10
    my = H - MARGIN - 68 - mh

    for mod in modules:
        # 헤더 영역
        c.setFillColor(mod["color"])
        c.roundRect(mx, my + mh - 44, mw - 8, 44, 5, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 11)
        c.drawCentredString(mx + (mw - 8) / 2, my + mh - 22, mod["title"])
        c.setFont(FONT_L, 7.5)
        c.drawCentredString(mx + (mw - 8) / 2, my + mh - 35, mod["sub"])
        # 내용 카드
        draw_card(c, mx, my, mw - 8, mh - 46, bg=GRAY_LIGHT)
        ty = my + mh - 62
        for item in mod["items"]:
            c.setFillColor(mod["color"])
            c.circle(mx + 9, ty + 3.5, 2, fill=1, stroke=0)
            c.setFillColor(GRAY_DARK)
            c.setFont(FONT_L, 8)
            c.drawString(mx + 16, ty, item)
            ty -= 16
        mx += mw

    # 파이프라인 화살표
    arrow_y = my + mh - 22
    ax = MARGIN + 10 + mw - 13
    for _ in range(3):
        c.setFillColor(GRAY_MID)
        c.setFont(FONT_B, 14)
        c.drawString(ax, arrow_y, "→")
        ax += mw

    # 최종 점수 설명
    yb = MARGIN + 10
    draw_green_box(
        c, MARGIN + 10, yb, W - MARGIN * 2 - 20, 28,
        "final_score = safety_score × w_safety + tourism_score × w_tourism  "
        "(모드: 안전우선 7:3 / 균형 5:5 / 관광우선 3:7)", 9,
    )


def slide_performance(c):
    """04 Model Performance"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "04", "Model Performance")
    draw_slide_title(c, "모델 성능 지표", "베이스라인 대비 정량 평가")

    # 메트릭 카드
    metrics = [
        ("0.91",  "안전 모델 R²",    GREEN_DARK),
        ("0.82",  "안전 모델 F1",    GREEN_MID),
        ("0.87",  "관광 모델 R²",    BLUE_DARK),
        ("0.126", "POI Recall@5",   ORANGE),
    ]
    mw_c, mh_c = 136, 66
    gap_c = 10
    total_mw = len(metrics) * mw_c + (len(metrics) - 1) * gap_c
    mx_c = (W - total_mw) / 2
    my_c = H - MARGIN - 58 - mh_c
    for val, lbl, col in metrics:
        draw_metric_card(c, mx_c, my_c, mw_c, mh_c, val, lbl, col)
        mx_c += mw_c + gap_c

    # POI 추천 성능 테이블
    y_t = my_c - 30
    c.setFillColor(GRAY_DARK)
    c.setFont(FONT_B, 10)
    c.drawString(MARGIN + 10, y_t + 5, "POI 추천 성능 (test, max_dist=20km)")
    draw_table(
        c, MARGIN + 10, y_t - 12,
        ["모델", "valid_trips", "Recall@5", "Recall@10"],
        [
            ["Co-occurrence v2", "166", "0.1260", "0.1761"],
            ["인기도 베이스라인", "166", "0.0370", "0.0498"],
            ["향상 (↑배수)",     "—",   "× 3.4배", "× 3.5배"],
        ],
        col_widths=[165, 100, 110, 110], row_height=18,
    )

    # GRU 실험 결과 테이블
    y_g = y_t - 100
    c.setFillColor(GRAY_DARK)
    c.setFont(FONT_B, 10)
    c.drawString(MARGIN + 10, y_g + 5, "GRU 시퀀스 모델 실험 (한계 확인 → Co-occurrence 전환)")
    draw_table(
        c, MARGIN + 10, y_g - 12,
        ["실험", "vocab", "train 샘플", "test_top5", "랜덤 기댓값"],
        [
            ["min_freq 없음", "9,881", "7,747", "0.99%", "0.05%"],
            ["min_freq=3",   "1,001", "3,798", "0.46%", "0.50%"],
            ["min_freq=30",  "29",    "1,781", "14.1%", "17.2% ↓ 미달"],
        ],
        col_widths=[130, 80, 95, 100, 110], row_height=18,
    )

    yb = MARGIN + 10
    draw_green_box(
        c, MARGIN + 10, yb, W - MARGIN * 2 - 20, 26,
        "GRU: 관광 데이터의 순서 패턴 부재로 구조적 한계 확인 → Co-occurrence 전환으로 Recall@5 3.4배 향상", 8.5,
    )


def slide_poi(c):
    """05 POI Recommendation"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "05", "POI Recommendation")
    draw_slide_title(c, "Co-occurrence 관광지 추천 결과",
                     "Jaccard 정규화 + Haversine 거리 필터 (반경 20km)")

    # 알고리즘 설명
    ax = MARGIN + 10
    ay = H - MARGIN - 68
    draw_card(c, ax, ay - 88, 248, 96, bg=GREEN_LIGHT)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_B, 10)
    c.drawString(ax + 10, ay - 4, "알고리즘")
    steps = [
        "① 여행별 방문 장소 집합 구성 (순서 제거)",
        "② 장소 쌍 co-visit 카운트 → co_occ[A][B]",
        "③ Jaccard(A,B) = co_occ / (cnt_A + cnt_B - co_occ)",
        "④ seed 집합 Jaccard 벡터 합산 → Top-N 반환",
        "⑤ Haversine 거리 필터: 시드 중심 20km 이내",
    ]
    ty = ay - 22
    for step in steps:
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_L, 8.5)
        c.drawString(ax + 10, ty, step)
        ty -= 14

    # 샘플 추천 결과 3개 카드
    samples = [
        {
            "seed": "서울역",
            "recs": [("더 현대 서울",    "0.0909", "4.9km"),
                     ("홍대 패션거리",   "0.0741", "4.1km"),
                     ("신세계백화점",    "0.0536", "1.2km")],
        },
        {
            "seed": "더 현대 서울",
            "recs": [("여의도 한강공원", "0.1837", "0.6km"),
                     ("IFC 몰",         "0.1702", "0.2km"),
                     ("영등포역",       "0.0612", "2.1km")],
        },
        {
            "seed": "화성행궁",
            "recs": [("방화수류정",      "0.2414", "0.7km"),
                     ("행리단길",       "0.1698", "0.4km"),
                     ("장안문",         "0.1481", "0.8km")],
        },
    ]
    sw = (W - MARGIN * 2 - 268) / 3
    sx = MARGIN + 268
    sy = H - MARGIN - 68

    for samp in samples:
        c.setFillColor(GREEN_DARK)
        c.roundRect(sx, sy - 24, sw - 8, 24, 4, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 9)
        c.drawCentredString(sx + (sw - 8) / 2, sy - 13, f"seed: {samp['seed']}")
        draw_card(c, sx, sy - 124, sw - 8, 102, bg=GRAY_LIGHT)
        ry = sy - 38
        for rank, (name, jac, dist) in enumerate(samp["recs"]):
            c.setFillColor(ACCENT)
            c.setFont(FONT_B, 8)
            c.drawString(sx + 8, ry, f"#{rank + 1}")
            c.setFillColor(GRAY_DARK)
            c.setFont(FONT_M, 8.5)
            c.drawString(sx + 24, ry, name)
            c.setFillColor(GRAY_MID)
            c.setFont(FONT_L, 7.5)
            c.drawString(sx + 8, ry - 11, f"Jaccard={jac}  거리={dist}")
            ry -= 30
        sx += sw

    # 평가 테이블
    yb = MARGIN + 42
    draw_table(
        c, MARGIN + 10, yb + 26,
        ["지표", "val", "test", "베이스라인 (test)", "향상"],
        [
            ["Recall@5",  "0.1144", "0.1260", "0.0370", "↑ 3.4배"],
            ["Recall@10", "0.1699", "0.1761", "0.0498", "↑ 3.5배"],
        ],
        col_widths=[80, 70, 70, 130, 80], row_height=17,
    )

    # 인사이트
    draw_card(c, MARGIN + 10, MARGIN + 10, W - MARGIN * 2 - 20, 28, bg=GREEN_LIGHT)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_M, 8.5)
    c.drawString(MARGIN + 20, MARGIN + 22,
                 "화성행궁 → Top-5 전부 수원화성 인근 명소 (행리단길·장안문·화성 박물관·화홍문): 지리적으로 응집된 고품질 추천")


def slide_demo(c):
    """06 Service Demo"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "06", "Service Demo")
    draw_slide_title(c, "Streamlit 서비스 구성", "5개 탭 구조의 실시간 데모 앱 (Folium 지도 연동)")

    tabs = [
        ("탭 1", "안전등급 예측", [
            "도로 너비·길이·지역 위험도 입력",
            "RF 모델 안전 등급(0·1·2) 예측",
            "등급별 확률 바 차트 시각화",
        ]),
        ("탭 2", "경로 추천 Top-10", [
            "모드별 가중치 적용",
            "날씨 기반 자동 가중치 보정",
            "추천 점수 분포 히스토그램",
        ]),
        ("탭 3", "데이터 탐색", [
            "안전·관광 점수 분포 시각화",
            "피처 통계 요약 테이블",
            "안전 vs 관광 산점도",
        ]),
        ("탭 4", "경로 탐색 (지도)", [
            "A→B 최적 경로 Dijkstra",
            "순환 코스 + POI 오버레이",
            "Folium 지도 + 편의시설 마커",
        ]),
        ("탭 5", "관광지 추천 (지도)", [
            "시드 장소 멀티셀렉트",
            "Co-occurrence 추천 + 지도",
            "자전거 경로 연결 토글",
        ]),
    ]

    tw = (W - MARGIN * 2 - 20) / 5
    tx = MARGIN + 10
    ty_top = H - MARGIN - 68
    th = 148

    for tab_num, tab_title, items in tabs:
        c.setFillColor(GREEN_MID)
        c.roundRect(tx, ty_top - 28, tw - 6, 28, 5, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 8)
        c.drawCentredString(tx + (tw - 6) / 2, ty_top - 12, tab_num)
        c.setFont(FONT_L, 7.5)
        c.drawCentredString(tx + (tw - 6) / 2, ty_top - 22, tab_title)
        draw_card(c, tx, ty_top - 28 - (th - 28), tw - 6, th - 28, bg=GRAY_LIGHT)
        iy = ty_top - 52
        for item in items:
            draw_bullet(c, tx + 8, iy, item, font_size=7.8)
            iy -= 17
        tx += tw

    # 기술 스택
    yb = MARGIN + 10
    stacks = [
        ("Frontend",  "Streamlit · Folium · streamlit-folium"),
        ("ML/DL",     "scikit-learn · pytorch-tabnet"),
        ("지도/경로", "osmnx · NetworkX · scipy KDTree"),
        ("데이터",    "한국관광공사 API · 서울시 공공데이터"),
    ]
    sw = (W - MARGIN * 2 - 20) / 4
    sx = MARGIN + 10
    for cat, tech in stacks:
        draw_card(c, sx, yb, sw - 6, 30, bg=GREEN_LIGHT)
        c.setFillColor(GREEN_DARK)
        c.setFont(FONT_B, 8)
        c.drawString(sx + 8, yb + 17, cat)
        c.setFillColor(GRAY_MID)
        c.setFont(FONT_L, 7.5)
        c.drawString(sx + 8, yb + 5, tech)
        sx += sw


def slide_conclusion(c):
    """07 Conclusion"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "07", "Conclusion")
    draw_slide_title(c, "결론 및 향후 계획",
                     "K-Ride 시스템 성과와 개선 방향")

    # 왼쪽: 주요 성과
    lw = W / 2 - MARGIN - 18
    lx = MARGIN + 10
    ly = H - MARGIN - 68

    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_B, 11)
    c.drawString(lx, ly, "주요 성과")
    c.setStrokeColor(GREEN_DARK)
    c.setLineWidth(1.5)
    c.line(lx, ly - 5, lx + 110, ly - 5)

    achievements = [
        ("안전 모델",    "RF 기반 안전 등급 분류 F1=0.82 / 회귀 R²=0.91"),
        ("관광 모델",    "TabNet v2 관광 매력도 R²=0.87 달성"),
        ("경로 그래프",  "서울 전역 30만+ 엣지 자전거 네트워크 구축"),
        ("POI 추천",    "Co-occurrence Recall@5=0.126 (베이스라인 3.4배)"),
        ("서비스",       "Streamlit 5탭 + Folium 지도 시각화 통합"),
    ]
    ay = ly - 24
    for title, desc in achievements:
        draw_card(c, lx, ay - 24, lw, 26, bg=GREEN_LIGHT)
        c.setFillColor(GREEN_DARK)
        c.setFont(FONT_B, 8.5)
        c.drawString(lx + 10, ay - 7, title)
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_L, 8.5)
        c.drawString(lx + 82, ay - 7, desc)
        ay -= 32

    # 오른쪽: 향후 계획
    rx = W / 2 + 8
    ry = ly

    c.setFillColor(BLUE_DARK)
    c.setFont(FONT_B, 11)
    c.drawString(rx, ry, "향후 계획")
    c.setStrokeColor(BLUE_DARK)
    c.setLineWidth(1.5)
    c.line(rx, ry - 5, rx + 110, ry - 5)

    plans = [
        ("단기", [
            "Streamlit Cloud 배포 (공개 URL 생성)",
            "FastAPI RESTful 백엔드 구축",
            "주소 검색 (Nominatim) 연동",
        ]),
        ("중기", [
            "KLUE-BERT 리뷰 감성 분석 연동",
            "사용자 피드백 기반 개인화 가중치",
            "실시간 교통·날씨 통합 경로 재계산",
        ]),
    ]
    py = ry - 24
    for phase, items in plans:
        c.setFillColor(BLUE_DARK)
        c.roundRect(rx, py - 6, 38, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 8)
        c.drawCentredString(rx + 19, py, phase)
        iy = py
        for item in items:
            draw_bullet(c, rx + 46, iy, item, font_size=8.5, color=GRAY_DARK)
            iy -= 16
        py = iy - 10

    # 하단 마무리
    yb = MARGIN + 10
    draw_card(c, MARGIN + 10, yb, W - MARGIN * 2 - 20, 32, bg=BLUE_LIGHT)
    c.setFillColor(BLUE_DARK)
    c.setFont(FONT_B, 10)
    c.drawCentredString(W / 2, yb + 18,
                        "공공 데이터 + 머신러닝/딥러닝으로 '안전하고 즐거운 자전거 경로'를 제공하는 K-Ride 완성")
    c.setFont(FONT_L, 8.5)
    c.setFillColor(GRAY_MID)
    c.drawCentredString(W / 2, yb + 6,
                        "서울시 자전거 이용자의 사고 예방 및 관광 활성화에 실질적으로 기여하는 AI 경로 추천 시스템")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    slides = [
        slide_cover,
        slide_contents,
        slide_overview,
        slide_dataset,
        slide_model_arch,
        slide_performance,
        slide_poi,
        slide_demo,
        slide_conclusion,
    ]

    cv = canvas.Canvas(OUT_PATH, pagesize=landscape(A4))
    cv.setTitle("K-Ride 딥러닝 산출물 보고서")
    cv.setAuthor("feed-mina")
    cv.setSubject("서울 자전거 안전 경로 추천 시스템")

    for i, slide_fn in enumerate(slides, 1):
        slide_fn(cv)
        if i > 1:
            cv.setFillColor(GRAY_MID)
            cv.setFont(FONT_L, 8)
            cv.drawCentredString(W / 2, 10, f"{i} / {len(slides)}")
        cv.showPage()

    cv.save()
    print(f"✅ 보고서 생성 완료: {OUT_PATH}")
    print(f"   총 {len(slides)}페이지")


if __name__ == "__main__":
    main()
