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


def draw_metric_card(c, x, y, w, h, value, label, color=GREEN_DARK, font_b=20, font_l=8):
    draw_card(c, x, y, w, h, bg=WHITE)
    c.setFillColor(color)
    c.setFont(FONT_B, font_b)
    c.drawCentredString(x + w / 2, y + h - 34 - (font_b - 20)/2, value)
    c.setFillColor(GRAY_MID)
    c.setFont(FONT_L, font_l)
    c.drawCentredString(x + w / 2, y + h - 50 - (font_b - 20)/2 - (font_l - 8)/2, label)


def draw_green_box(c, x, y, w, h, text, font_size=9):
    c.setFillColor(GREEN_LIGHT)
    c.setStrokeColor(GREEN_MID)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_M, font_size)
    c.drawCentredString(x + w / 2, y + h / 2 - font_size / 2, text)


def draw_table(c, x, y, headers, rows, col_widths, row_height=18,
               header_bg=GREEN_DARK, stripe=True, font_size_h=10, font_size_d=9.5):
    total_w = sum(col_widths)
    # 헤더
    c.setFillColor(header_bg)
    c.rect(x, y, total_w, row_height, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, font_size_h)
    cx = x
    for h_txt, cw in zip(headers, col_widths):
        c.drawCentredString(cx + cw / 2, y + (row_height - font_size_h)/2 + 1, h_txt)
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
        c.setFont(FONT_M, font_size_d)
        cx = x
        for cell, cw in zip(row, col_widths):
            c.drawCentredString(cx + cw / 2, ry + (row_height - font_size_d)/2 + 1, str(cell))
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
        ("03", "Model Architecture", "안전 · 경로 · POI 추천 모듈 구조"),
        ("04", "Model Performance",  "정량 지표 및 베이스라인 비교"),
        ("05", "Step 1: Weather",    "LSTM 기반 3분류 및 안전 패널티 (Data/Perf)"),
        ("06", "Step 2: TabNet",     "비선형 관광 매력도 예측 (Data/Perf/Map)"),
        ("07", "POI Recommendation", "Co-occurrence 기반 추천 결과"),
        ("08", "Service Demo",       "지도 웹 서비스 시연 데모"),
        ("09", "Lessons Learned",    "개발 간 배운 점 및 실무 인사이트"),
        ("10", "Future Work",        "한계점 및 개인화 추천 프레임워크 향후 과제"),
    ]
    y = H - 95
    for num, title, desc in items:
        c.setFillColor(GREEN_LIGHT)
        c.roundRect(MARGIN + 10, y - 6, W - MARGIN * 2 - 20, 24, 4, fill=1, stroke=0)
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_B, 10.5)
        c.drawString(MARGIN + 20, y + 5, num)
        c.setFont(FONT_B, 10.5)
        c.drawString(MARGIN + 54, y + 5, title)
        c.setFont(FONT_L, 9.5)
        c.drawString(MARGIN + 240, y + 5, desc)
        y -= 30


def slide_overview(c):
    """01 Project Overview"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "01", "Project Overview")
    draw_slide_title(c, "K-Ride 자전거 안전 경로 추천 시스템",
                     "머신러닝 + 딥러닝 기반 경로 최적화 및 관광지 추천 파이프라인")

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
    by = (H - 68) / 2 - (bh / 2) + 15

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
        c, MARGIN + 10, H - 240,
        ["데이터셋", "규모", "주요 컬럼"],
        [
            ["자전거도로 원본",     "57,418행",   "width_m, length_km, road_type"],
            ["도로 안전 점수",      "32,614행",   "safety_score, district_danger"],
            ["관광지 정보",         "8,813개소",  "관광지명, X/Y좌표, 유형코드"],
            ["여행자 방문지",       "21,384행",   "TRAVEL_ID, VISIT_AREA_NM"],
            ["OSM 자전거 네트워크", "23만+ 엣지", "서울 전역 topology 보장"],
        ],
        col_widths=[140, 90, 210], row_height=26, font_size_h=12, font_size_d=11
    )

    # 통계 카드
    stats = [
        ("57,418", "원본 도로 세그먼트", GREEN_DARK),
        ("7,748",  "좌표 보유 관광지",   GREEN_MID),
        ("2,560",  "여행 ID (trips)",    BLUE_DARK),
        ("1,646",  "POI vocab",          ORANGE),
    ]
    cw_s, ch_s = 100, 72
    gap_s = 10
    total_sw = len(stats) * cw_s + (len(stats) - 1) * gap_s
    sx = MARGIN + 480
    sy = H - 280
    for i, (val, lbl, col) in enumerate(stats):
        xi = sx + (i % 2) * (cw_s + gap_s)
        yi = sy - (i // 2) * (ch_s + gap_s)
        draw_metric_card(c, xi, yi, cw_s, ch_s, val, lbl, col, font_b=28, font_l=11)

    # 전처리 박스
    yb = MARGIN + 42
    draw_card(c, MARGIN + 10, yb, W - MARGIN * 2 - 20, 80, bg=GREEN_LIGHT)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_B, 11)
    c.drawString(MARGIN + 20, yb + 60, "전처리 핵심 사항")
    pts = [
        "POI_ID 결측(6,710행) → VISIT_AREA_NM 기반 대체",
        "비관광 장소(VISIT_AREA_TYPE_CD ≥ 21) 6,904행 제거 (자택·직장 등)",
        "osmnx KDTree로 OSM 엣지 ↔ road_scored 최근접 매핑 (허용 2km 이내)",
    ]
    px = MARGIN + 20
    py = yb + 40
    for pt in pts:
        draw_bullet(c, px, py, pt, font_size=11)
        py -= 18

    c.setFillColor(ORANGE)
    c.roundRect(MARGIN + 10, MARGIN + 10, W - MARGIN * 2 - 20, 24, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 10)
    c.drawString(MARGIN + 20, MARGIN + 18, "[개념 💡] KDTree: 무수히 많은 주변 도로 거리를 빛의 속도로 찾아주는 공간 탐색 기법  |  NetworkX: 복잡한 도로망을 점과 선으로 구현해주는 파이썬 수학 라이브러리")


def slide_model_arch(c):
    """03 Model Architecture"""
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "03", "Model Architecture")
    draw_slide_title(c, "모델 구성 - 4개 모듈 파이프라인",
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
                "성능: MAE=0.6558  R²=0.0662",
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
    mh = 178
    mx = MARGIN + 10
    my = H - MARGIN - 168 - mh

    for mod in modules:
        # 헤더 영역
        c.setFillColor(mod["color"])
        c.roundRect(mx, my + mh - 44, mw - 8, 44, 5, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 13)
        c.drawCentredString(mx + (mw - 8) / 2, my + mh - 20, mod["title"])
        c.setFont(FONT_L, 9.5)
        c.drawCentredString(mx + (mw - 8) / 2, my + mh - 36, mod["sub"])
        # 내용 카드
        draw_card(c, mx, my, mw - 8, mh - 46, bg=GRAY_LIGHT)
        ty = my + mh - 66
        for item in mod["items"]:
            c.setFillColor(mod["color"])
            c.circle(mx + 9, ty + 4.5, 2.5, fill=1, stroke=0)
            c.setFillColor(GRAY_DARK)
            c.setFont(FONT_M, 10.5)
            c.drawString(mx + 16, ty, item)
            ty -= 20
        mx += mw

    # 파이프라인 화살표
    arrow_y = my + mh - 22
    ax = MARGIN + 10 + mw - 13
    for _ in range(3):
        c.setFillColor(GRAY_MID)
        c.setFont(FONT_B, 20)
        c.drawString(ax, arrow_y - 2, "→")
        ax += mw

    c.setFillColor(ORANGE)
    c.roundRect(MARGIN + 10, MARGIN + 10, W - MARGIN * 2 - 20, 24, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 10)
    c.drawString(MARGIN + 20, MARGIN + 18, "[개념 💡] Dijkstra (다익스트라): 모든 갈래길의 비용(안전 및 관광점수)을 더해보고 비교하여, 시작부터 끝까지 가장 점수가 높은 최적의 경로를 찾아내는 길찾기 알고리즘")

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
        ("0.066", "관광 모델 R²",    BLUE_DARK),
        ("0.126", "POI Recall@5",   ORANGE),
    ]
    aw_c, ah_c = 136, 68
    gap_c = 10
    total_aw = len(metrics) * aw_c + (len(metrics) - 1) * gap_c
    ax_c = (W - total_aw) / 2
    ay_c = H - MARGIN - 68 - ah_c
    for val, lbl, col in metrics:
        draw_metric_card(c, ax_c, ay_c, aw_c, ah_c, val, lbl, col, font_b=26, font_l=11)
        ax_c += aw_c + gap_c

    # POI 추천 성능 테이블
    y_t = ay_c - 40
    c.setFillColor(GRAY_DARK)
    c.setFont(FONT_B, 12)
    c.drawString(MARGIN + 10, y_t + 8, "POI 추천 성능 (test, max_dist=20km)")
    draw_table(
        c, MARGIN + 10, y_t - 28,
        ["모델", "valid_trips", "Recall@5", "Recall@10"],
        [
            ["Co-occurrence v2", "166", "0.1260", "0.1761"],
            ["인기도 베이스라인", "166", "0.0370", "0.0498"],
            ["향상 (↑배수)",     "—",   "× 3.4배", "× 3.5배"],
        ],
        col_widths=[165, 100, 110, 110], row_height=22, font_size_h=12, font_size_d=11
    )

    # GRU 실험 결과 테이블
    y_g = y_t - 135
    c.setFillColor(GRAY_DARK)
    c.setFont(FONT_B, 12)
    c.drawString(MARGIN + 10, y_g + 8, "GRU 시퀀스 모델 실험 (한계 확인 → Co-occurrence 전환)")
    draw_table(
        c, MARGIN + 10, y_g - 28,
        ["실험", "vocab", "train 샘플", "test_top5", "랜덤 기댓값"],
        [
            ["min_freq 없음", "9,881", "7,747", "0.99%", "0.05%"],
            ["min_freq=3",   "1,001", "3,798", "0.46%", "0.50%"],
            ["min_freq=30",  "29",    "1,781", "14.1%", "17.2% ↓ 미달"],
        ],
        col_widths=[130, 80, 95, 100, 110], row_height=22, font_size_h=11, font_size_d=10
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
    ay = H - MARGIN - 168
    draw_card(c, ax, ay - 110, 248, 120, bg=GREEN_LIGHT)
    c.setFillColor(GREEN_DARK)
    c.setFont(FONT_B, 13)
    c.drawString(ax + 10, ay - 2, "알고리즘")
    steps = [
        "① 여행별 방문 장소 집합 구성 (순서 제거)",
        "② 장소 쌍 co-visit 카운트 → co_occ[A][B]",
        "③ Jaccard(A,B) = co_occ / (cnt_A + cnt_B - co_occ)",
        "④ seed 집합 Jaccard 벡터 합산 → Top-N 반환",
        "⑤ Haversine 거리 필터: 시드 중심 20km 이내",
    ]
    ty = ay - 26
    for step in steps:
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_M, 10.5)
        c.drawString(ax + 10, ty, step)
        ty -= 18

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
    sy = H - MARGIN - 168

    for samp in samples:
        c.setFillColor(GREEN_DARK)
        c.roundRect(sx, sy - 30, sw - 8, 30, 4, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 11)
        c.drawCentredString(sx + (sw - 8) / 2, sy - 16, f"seed: {samp['seed']}")
        draw_card(c, sx, sy - 150, sw - 8, 122, bg=GRAY_LIGHT)
        ry = sy - 48
        for rank, (name, jac, dist) in enumerate(samp["recs"]):
            c.setFillColor(ACCENT)
            c.setFont(FONT_B, 10)
            c.drawString(sx + 8, ry, f"#{rank + 1}")
            c.setFillColor(GRAY_DARK)
            c.setFont(FONT_B, 10.5)
            c.drawString(sx + 28, ry, name)
            c.setFillColor(GRAY_MID)
            c.setFont(FONT_M, 8.5)
            c.drawString(sx + 8, ry - 13, f"Jaccard={jac}  거리={dist}")
            ry -= 36
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



def slide_step1_weather_arch(c):
    from reportlab.lib.utils import ImageReader
    import os
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "05", "Step 1: Weather LSTM (Data & Arch)")
    draw_slide_title(c, "데이터 불균형 분석 및 아키텍처",
                     "기상청 ASOS 14일 시퀀스 기반 날씨 3분류 분석")

    draw_card(c, MARGIN + 10, MARGIN + 10, W/2 - MARGIN - 20, H - MARGIN * 2 - 80, bg=GREEN_LIGHT)
    c.setFillColor(BLUE_DARK)
    c.setFont(FONT_B, 13)
    c.drawString(MARGIN + 30, H - 120, "1-1. 데이터 분포와 모델 구조")
    
    pts = [
        "분석 원천: 10년간 기상청 ASOS 일별/시간별 관측 데이터",
        "극심한 불균형: 전체 중 맑음(70%), 비눈(23%), 흐림(7%)만 점유",
        "단순 편향: 기존 다수 클래스 분류 모델은 흐림을 100% 무시함",
        "LSTM 시계열 피처: 평균기온, 강수량, 상대습도 등 8개 연속 피처",
        "Time Steps: 최근 14일치 기상 상태를 Look-back window로 사용",
    ]
    py = H - 170
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=12)
        py -= 32

    right_x = W/2 + 20
    im1 = os.path.join(BASE_DIR, "report", "charts", "01_data_distribution.png")
    im2 = os.path.join(BASE_DIR, "report", "charts", "06_model_architecture.png")
    if os.path.exists(im1):
        c.drawImage(ImageReader(im1), right_x, H/2 + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')
    if os.path.exists(im2):
        c.drawImage(ImageReader(im2), right_x, MARGIN + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')

def slide_step1_weather_res(c):
    from reportlab.lib.utils import ImageReader
    import os
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "05", "Step 1: Weather LSTM (Results)")
    draw_slide_title(c, "클래스 가중치 최적화 결과 및 노면 안전 패널티",
                     "성능 지표 한계 극복 및 Safety Score에의 기상 보정 패널티 환류")

    draw_card(c, MARGIN + 10, MARGIN + 10, W/2 - MARGIN - 20, H - MARGIN * 2 - 80, bg=GREEN_LIGHT)
    c.setFillColor(BLUE_DARK)
    c.setFont(FONT_B, 13)
    c.drawString(MARGIN + 30, H - 120, "1-2. 성능 검증 및 패널티 통합")
    
    pts = [
        "해결책 적용: 역비율 클래스 가중치(흐림에 가중치 4.41배) 적용",
        "극복 지표: 흐림(1) F1=0.00 에서 0.08로 유의미성 획득 (가중치 동작 검증)",
        "최종 Acc: 검증 73.2%, 테스트 64.9%",
        "모델 응용: 내일의 기상 예측 라벨에 따라 도로 안전에 즉각 패널티 발동",
        "패널티 부과 규칙:",
        " -> [맑음]: 보정값 없음",
        " -> [흐림/미세비]: 안전점수 -0.05 하향 보정 (노면 적응 단계)",
        " -> [비/눈/얼음]: 안전점수 최대 -0.20 하향 (자전거 기피 경로 형성)"
    ]
    py = H - 170
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=12)
        py -= 32

    right_x = W/2 + 20
    im1 = os.path.join(BASE_DIR, "report", "charts", "03_learning_curve.png")
    im2 = os.path.join(BASE_DIR, "report", "charts", "04_confusion_matrix.png")
    if os.path.exists(im1):
        c.drawImage(ImageReader(im1), right_x, H/2 + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')
    if os.path.exists(im2):
        c.drawImage(ImageReader(im2), right_x, MARGIN + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')

def slide_step2_poi_arch(c):
    from reportlab.lib.utils import ImageReader
    import os
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "06", "Step 2: TabNet (Data & Arch)")
    draw_slide_title(c, "딥러닝 기반 관광 매력도 예측망",
                     "AI Hub 국내 여행로그 기반 데이터 구축 및 TabNet 비선형 어텐션 학습기")

    draw_card(c, MARGIN + 10, MARGIN + 10, W/2 - MARGIN - 20, H - MARGIN * 2 - 80, bg=BLUE_LIGHT)
    c.setFillColor(ORANGE)
    c.setFont(FONT_B, 13)
    c.drawString(MARGIN + 30, H - 120, "2-1. 타겟 설계 및 Attentive Architecture")
    
    pts = [
        "학습 근거: AI Hub 내 수도권(경기/인천/서울) 사용자 여행 로그",
        "타겟(Tourism Score): 관광명소 고유 체류빈도와 만족도를 스케일링(0~1)",
        "매칭: 1,647개 OSM 도로 세그먼트와 반경 2km 이내의 POI를 조인 반영",
        "TabNet 도입: 정형 데이터(위경도, 밀도 등)에서 비선형성을 갖춘 피처 발굴",
        "모델 특성: Attention Mechanism을 이용해 해석 가능한 피처 중요도 도출"
    ]
    py = H - 170
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=12, color=GRAY_DARK)
        py -= 32

    c.setFillColor(ORANGE)
    c.roundRect(MARGIN + 30, 60, W/2 - MARGIN - 40, 60, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 12)
    c.drawString(MARGIN + 45, 102, "[개념 💡] POI TabNet 구조란?")
    c.setFont(FONT_M, 11)
    c.drawString(MARGIN + 45, 84, "정형 데이터의 규칙(Tree)과 신경망(DNN)의 장점을 결합하여")
    c.drawString(MARGIN + 45, 70, "가장 똑똑하고 신속하게 관광지 매력도를 예측하는 AI")

    right_x = W/2 + 20
    im1 = os.path.join(BASE_DIR, "report", "charts", "08_poi_target_distribution.png")
    im2 = os.path.join(BASE_DIR, "report", "charts", "09_tabnet_architecture.png")
    if os.path.exists(im1):
        c.drawImage(ImageReader(im1), right_x, H/2 + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')
    if os.path.exists(im2):
        c.drawImage(ImageReader(im2), right_x, MARGIN + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')

def slide_step2_poi_res(c):
    from reportlab.lib.utils import ImageReader
    import os
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "06", "Step 2: TabNet (Result & Limitation)")
    draw_slide_title(c, "POI 예측 성능 검증 및 한계돌파 단서 도출",
                     "MAE, R² 수치 및 Feature Importance 피처 기여도 해석")

    draw_card(c, MARGIN + 10, MARGIN + 10, W/2 - MARGIN - 20, H - MARGIN * 2 - 80, bg=BLUE_LIGHT)
    c.setFillColor(ORANGE)
    c.setFont(FONT_B, 13)
    c.drawString(MARGIN + 30, H - 120, "2-2. 성과 및 소비패턴 피처의 영향도 분석")
    
    pts = [
        "성과 지표: MAE 0.6558, R² 0.0662. 타겟 변동률 대비 설명력이 다소 낮음",
        "오차 요인: 소비/방문이라는 인간의 패턴의 큰 분산값 및 외부 환경 피처 결여",
        "Feature Importance 해석(우측 맵핑):",
        " - 주변 장소와의 위상(Latitude, Longitude)이 가장 높은 가중치를 받음",
        " - 모델이 '지역 응집(Clustering)'이 밀집된 곳을 매력적인 관광지로 유추",
        "개선 연계: 딥러닝 타겟 다변화 및 개인화 파생 피처 엔지니어링 집중 필요"
    ]
    py = H - 170
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=12, color=GRAY_DARK)
        py -= 32

    right_x = W/2 + 20
    im1 = os.path.join(BASE_DIR, "report", "charts", "10_tabnet_learning_curve.png")
    im2 = os.path.join(BASE_DIR, "report", "charts", "11_tabnet_feature_importance.png")
    if os.path.exists(im1):
        c.drawImage(ImageReader(im1), right_x, H/2 + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')
    if os.path.exists(im2):
        c.drawImage(ImageReader(im2), right_x, MARGIN + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')

def slide_step2_poi_map(c):
    from reportlab.lib.utils import ImageReader
    import os
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "06", "Step 2: TabNet (Integration Map)")
    draw_slide_title(c, "경로상 평가 맵핑 및 가중합(Weighted Sum)",
                     "Dijkstra 그래프에 주입될 안전(Safety) + 매력도(Tourism)의 통합")

    draw_card(c, MARGIN + 10, MARGIN + 10, W/2 - MARGIN - 20, H - MARGIN * 2 - 80, bg=BLUE_LIGHT)
    c.setFillColor(ORANGE)
    c.setFont(FONT_B, 13)
    c.drawString(MARGIN + 30, H - 120, "2-3. Final Score 계산 및 시각화")
    
    pts = [
        "산출식: Final Cost = (Safety × 0.6) + (Tourism Score × 0.4)",
        "결과 분포 변화: 안전만 고려했을 때 우회하던 코스나 빈약했던 경로들이",
        "관광지 밀집도가 높은 도심 구역에서 가중치 보너스를 받아 최적 경로로 선택됨",
        "시각화 검토(우측 차트): 수치 산포 비교 및 Folium 기반의 노드 색상 맵핑 시",
        "의도했던 바와 같이 명확한 Tourism 핫스팟이 두껍게 표현됨을 확인",
        "이후 route_graph.pkl 빌드(17만 노드) 완료 후 Streamlit 서빙으로 전환"
    ]
    py = H - 170
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=12, color=GRAY_DARK)
        py -= 32

    right_x = W/2 + 20
    im1 = os.path.join(BASE_DIR, "report", "charts", "14_poi_map.png")
    im2 = os.path.join(BASE_DIR, "report", "charts", "15_tourism_score_comparison.png")
    if os.path.exists(im1):
        c.drawImage(ImageReader(im1), right_x, H/2 + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')
    if os.path.exists(im2):
        c.drawImage(ImageReader(im2), right_x, MARGIN + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')


def draw_icon(c, cx, cy, kind):
    def _poly(pts, fill=1, stroke=0):
        p = c.beginPath()
        p.moveTo(pts[0], pts[1])
        for i in range(2, len(pts), 2):
            p.lineTo(pts[i], pts[i+1])
        p.close()
        c.drawPath(p, fill=fill, stroke=stroke)

    # Vector pictograms
    c.setLineWidth(2)
    if kind == "weather":
        c.setFillColor(HexColor("#3498DB")); c.setStrokeColor(WHITE)
        c.circle(cx-8, cy-2, 10, fill=1, stroke=0)
        c.circle(cx+8, cy-2, 8, fill=1, stroke=0)
        c.circle(cx, cy+6, 12, fill=1, stroke=0)
    elif kind == "safety":
        c.setFillColor(HexColor("#2ECC71")); c.setStrokeColor(WHITE)
        _poly([cx-10, cy+10, cx+10, cy+10, cx+10, cy-5, cx, cy-15, cx-10, cy-5])
    elif kind == "tourism":
        c.setFillColor(HexColor("#F1C40F")); c.setStrokeColor(WHITE)
        c.circle(cx, cy, 12, fill=1, stroke=0); c.setFillColor(WHITE); c.circle(cx, cy, 6, fill=1, stroke=0)
    elif kind == "routing":
        c.setStrokeColor(HexColor("#E74C3C")); c.setLineWidth(3)
        c.line(cx-10, cy-10, cx-4, cy+5); c.line(cx-4, cy+5, cx+4, cy-5); c.line(cx+4, cy-5, cx+10, cy+10)
        c.setFillColor(HexColor("#E74C3C")); c.circle(cx+10, cy+10, 3, fill=1, stroke=0)
    elif kind == "poi":
        c.setFillColor(HexColor("#9B59B6")); c.setStrokeColor(WHITE)
        c.circle(cx, cy+5, 8, fill=1, stroke=0); _poly([cx-8, cy+5, cx+8, cy+5, cx, cy-12])
        c.setFillColor(WHITE); c.circle(cx, cy+5, 3, fill=1, stroke=0)
    elif kind == "vision":
        c.setFillColor(HexColor("#F39C12"))
        c.rect(cx-12, cy-8, 24, 16, fill=1, stroke=0); c.setFillColor(WHITE); c.circle(cx, cy, 5, fill=1, stroke=0)
    elif kind == "data":
        c.setFillColor(HexColor("#34495E"))
        c.rect(cx-10, cy-10, 20, 4, fill=1, stroke=0)
        c.rect(cx-10, cy-2, 20, 4, fill=1, stroke=0)
        c.rect(cx-10, cy+6, 20, 4, fill=1, stroke=0)
    elif kind == "user":
        c.setFillColor(HexColor("#1ABC9C"))
        c.circle(cx, cy+4, 6, fill=1, stroke=0); _poly([cx-10, cy-10, cx+10, cy-10, cx+6, cy-2, cx-6, cy-2])
    elif kind == "theme":
        c.setFillColor(HexColor("#E67E22"))
        c.circle(cx-5, cy-5, 7, fill=1, stroke=0); c.circle(cx+5, cy+5, 7, fill=1, stroke=0)
    elif kind == "scale":
        c.setFillColor(HexColor("#ECF0F1")); c.setStrokeColor(HexColor("#3498DB")); c.setLineWidth(2)
        c.rect(cx-8, cy-8, 16, 16, fill=1, stroke=1)
        c.circle(cx, cy, 4, fill=1, stroke=0)
    elif kind == "loop":
        c.setStrokeColor(HexColor("#8E44AD")); c.setLineWidth(3)
        c.line(cx-8, cy-5, cx+8, cy+5); c.line(cx+8, cy-5, cx-8, cy+5)

def slide_demo(c):
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "08", "Service Demo")
    draw_slide_title(c, "Streamlit 서비스 구성 및 데모 화면", "5개 탭 구조의 실시간 데모 앱 (Folium 환경 구축 및 카카오/네이버 맵 한계 극복)")

    tabs = [
        ("1. 날씨 연동", "요일/기상 조회\n비/눈 감지 패널티", "weather"),
        ("2. 자전거 안전", "[Tab 1] 안전 등급\n노드 색상 그라데이션", "safety"),
        ("3. 매력도 추천", "[Tab 2, 4] 핫스팟 렌더링\n핵심 POI 오버레이", "tourism"),
        ("4. 최적 경로", "[Tab 3] A출발 → B도착\nDijkstra 안전 경로", "routing"),
        ("5. 명소 연계", "[Tab 5] 목적지 주변\n상호 연관 명소 추천", "poi")
    ]
    cw = (W - MARGIN * 2 - 40) / 5
    ch = 140
    cx = MARGIN + 10
    cy = H/2 - 20 - ch/2

    for title, desc, icon in tabs:
        draw_card(c, cx, cy, cw, ch, bg=GRAY_LIGHT)
        # title
        c.setFillColor(GREEN_DARK)
        c.setFont(FONT_B, 12)
        c.drawCentredString(cx + cw/2, cy + ch - 30, title)
        # line
        c.setStrokeColor(HexColor("#D0D0D0"))
        c.setLineWidth(1)
        c.line(cx+10, cy+ch-40, cx+cw-10, cy+ch-40)
        # Two-line desc
        c.setFillColor(GRAY_MID)
        c.setFont(FONT_M, 10.5)
        ty = cy + ch - 60
        for line in desc.split('\n'):
            c.drawCentredString(cx + cw/2, ty, line)
            ty -= 16
        # Pictogram at bottom
        draw_icon(c, cx + cw/2, cy + 30, icon)
        
        cx += cw + 10

def slide_insights(c):
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "09", "Lessons Learned")
    draw_slide_title(c, "프로젝트 리뷰: 인사이트 및 배운 점", "개발 파이프라인 전 과정에서 얻은 본질적 깨달음")

    draw_card(c, MARGIN + 10, H/2 - 90, W - MARGIN * 2 - 20, 260, bg=GREEN_LIGHT)
    c.setFillColor(GREEN_DARK)
    
    pts = [
        ("서비스 본질 중심의 기획:", "엔지니어링 기술도 중요하나 '서비스의 최종 목적과 기획 의도'를 항상 최우선으로 두어야 프로젝트가 궤도를 잃지 않는다는 원칙 체감."),
        ("알고리즘의 확장 한계 체감:", "단순 유클리디안 거리 기반 알고리즘을 넘어 네트워크 토폴로지를 적용한 그래프 탐색(Dijkstra) 알고리즘의 복잡성과 최적화(Caching) 성능 체감."),
        ("데이터 핏(Data-Fit) 모델 탐구:", "딥러닝이라도 무조건 복잡한 구조(GRU 시계열)보다, 서비스 성격 및 추천 범위를 관통하는 적절한 방법론(Co-occurrence) 도입이 더 실용적일 수 있음을 확인."),
        ("상용 실무와 유사한 지도 인터페이스 연동:", "단순 CSV 출력을 넘어 네이버/카카오 API 맵에 준하는 경로 지도를 Folium, Streamlit과 연계하며 공간정보(GeoJSON) 데이터의 프론트엔드 서빙 경험 확보.")
    ]
    py = H/2 + 130
    for title, desc in pts:
        c.setFont(FONT_B, 13)
        c.setFillColor(BLUE_DARK)
        c.drawString(MARGIN + 30, py, "▪ " + title)
        c.setFont(FONT_M, 11.5)
        c.setFillColor(GRAY_DARK)
        c.drawString(MARGIN + 30, py - 20, "   " + desc)
        py -= 55

def slide_future(c):
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "10", "Future Work")
    draw_slide_title(c, "아쉬운 점 및 향후 개선 과제", "스케일업 및 고도화 파이프라인 전망치")

    # 3x2 Grid View
    cards = [
        ("비전 딥러닝 확대", "하늘/구름 실시간 Vision CNN 추론을\n날씨 예측에 융합하여 기상 시너지 극대화", "vision"),
        ("타겟 분산성 교정", "소비/매력도 예측 정확도 한계 돌파를 위한\n파생 피처 발굴 및 데이터 수집채널 확대", "data"),
        ("초개인화 경로 로직", "성별·나이·BMI 등 신체 지표를 반영하여\n사용자 수준별 맞춤 경로 난이도 자동 조절", "user"),
        ("멀티테마 관광망", "단순화된 관광지 점수에서 벗어나\n자연·역사·액티비티 등 취향 테마별 경로 추출", "theme"),
        ("지역 스케일업", "서울권역 중심의 프로토타입을 넘어\n경기·인천 메가시티 및 전국 자전거망 확대", "scale"),
        ("피드백 MLOps 구축", "유저 만족도 피드백 및 사고 위험 제보를\n가중치 재학습 파이프라인에 즉각 편입", "loop"),
    ]

    card_w = (W - MARGIN * 2 - 40) / 3
    card_h = 130
    start_x = MARGIN + 10
    start_y = H - 160

    for i, (title, desc, icon) in enumerate(cards):
        col = i % 3
        row = i // 3
        cx = start_x + (col * (card_w + 20))
        cy = start_y - (row * (card_h + 30))

        # Box background
        draw_card(c, cx, cy - card_h, card_w, card_h, bg=HexColor("#F8F9F9"))
        
        # Header strip
        c.setFillColor(HexColor("#EBF5FB"))
        c.roundRect(cx, cy - 40, card_w, 40, 6, fill=1, stroke=0)
        
        # Corner icon
        draw_icon(c, cx + 25, cy - 20, icon)
        
        # Title
        c.setFillColor(BLUE_DARK)
        c.setFont(FONT_B, 13)
        c.drawString(cx + 50, cy - 26, f"{i+1}. {title}")
        
        # Content body
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_M, 11.5)
        ty = cy - 75
        for line in desc.split('\n'):
            c.drawString(cx + 20, ty, line)
            ty -= 22


def main():
    slides = [
        slide_cover,
        slide_contents,
        slide_overview,
        slide_dataset,
        slide_model_arch,
        slide_performance,
        slide_poi,
        slide_step1_weather_arch,
        slide_step1_weather_res,
        slide_step2_poi_arch,
        slide_step2_poi_res,
        slide_step2_poi_map,
        slide_demo,
        slide_insights,
        slide_future
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
    print(f"[SUCCESS] 보고서 생성 완료: {OUT_PATH}")
    print(f"   총 {len(slides)}페이지")

if __name__ == "__main__":
    main()
