import os

FILE_PATH = r"c:\Users\human-32\OneDrive\ドキュメント\yerinMin\humaneducation\kride-project\generate_report.py"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Replace slide_contents items array
old_items = """    items = [
        ("01", "Project Overview",   "프로젝트 목표 및 Why / How"),
        ("02", "Dataset Analysis",   "데이터 현황 및 전처리 특성"),
        ("03", "Model Architecture", "안전 · 관광 · 경로 · POI 추천 모듈"),
        ("04", "Model Performance",  "정량 지표 및 베이스라인 비교"),
        ("05", "POI Recommendation", "Co-occurrence 관광지 추천 결과"),
        ("06", "Step 1: Weather",    "날씨 3분류 예측 및 가중치 패널티"),
        ("07", "Step 2: POI TabNet", "방문지 매력도 딥러닝 예측"),
        ("08", "Service Demo",       "Streamlit 5탭 서비스 구성"),
        ("09", "Conclusion",         "결론 및 향후 계획 (Step 3 모델 통합)"),
    ]
    y = H - 105
    for num, title, desc in items:
        c.setFillColor(GREEN_LIGHT)
        c.roundRect(MARGIN + 10, y - 8, W - MARGIN * 2 - 20, 26, 4, fill=1, stroke=0)
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_B, 11)
        c.drawString(MARGIN + 20, y + 5, num)
        c.setFont(FONT_B, 11)
        c.drawString(MARGIN + 54, y + 5, title)
        c.setFont(FONT_L, 9)
        c.drawString(MARGIN + 225, y + 5, desc)
        y -= 36"""

new_items = """    items = [
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
        y -= 30"""

text = text.replace(old_items, new_items)

# Now segment out the slide_step1_weather down to slide_conclusion, and replace it entirely.
parts = text.split('def slide_step1_weather(c):')
before_text = parts[0]

new_slides_code = """
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
    py = H - 150
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=10.5)
        py -= 28

    right_x = W/2 + 20
    im1 = os.path.join("report", "charts", "01_data_distribution.png")
    im2 = os.path.join("report", "charts", "06_model_architecture.png")
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
    py = H - 150
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=10)
        py -= 24

    right_x = W/2 + 20
    im1 = os.path.join("report", "charts", "03_learning_curve.png")
    im2 = os.path.join("report", "charts", "04_confusion_matrix.png")
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
    py = H - 150
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=10.2, color=GRAY_DARK)
        py -= 28

    right_x = W/2 + 20
    im1 = os.path.join("report", "charts", "08_poi_target_distribution.png")
    im2 = os.path.join("report", "charts", "09_tabnet_architecture.png")
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
    py = H - 150
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=10, color=GRAY_DARK)
        py -= 28

    right_x = W/2 + 20
    im1 = os.path.join("report", "charts", "10_tabnet_learning_curve.png")
    im2 = os.path.join("report", "charts", "11_tabnet_feature_importance.png")
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
    py = H - 150
    for pt in pts:
        draw_bullet(c, MARGIN + 30, py, pt, font_size=10, color=GRAY_DARK)
        py -= 28

    right_x = W/2 + 20
    im1 = os.path.join("report", "charts", "14_poi_map.png")
    im2 = os.path.join("report", "charts", "15_tourism_score_comparison.png")
    if os.path.exists(im1):
        c.drawImage(ImageReader(im1), right_x, H/2 + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')
    if os.path.exists(im2):
        c.drawImage(ImageReader(im2), right_x, MARGIN + 10, width=320, height=180, preserveAspectRatio=True, anchor='c')


def slide_demo(c):
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_border(c, GREEN_DARK, lw=4)
    draw_page_header(c, "08", "Service Demo")
    draw_slide_title(c, "Streamlit 서비스 구성 및 데모 화면", "5개 탭 구조의 실시간 데모 앱 (Folium 환경 구축 및 카카오/네이버 맵 한계 극복)")

    tabs = [
        ("1. 실시간 날씨 및 기상 연동", "사이드바에 접속 요일, 강수 유무 표시\n(비/눈 감지 시 경로 패널티 반영)"),
        ("2. 자전거 안전 도로 분석", "[Tab 1] 안전 등급 회귀 점수(0~1)를\n초록-빨강 그라데이션으로 시각화"),
        ("3. 관광 매력도 및 추천", "[Tab 2, 4] TabNet 기반 관광 매력도를\n도로 위에 렌더링, 핵심 POI 마커 오버레이"),
        ("4. 자동 최적 경로 제공", "[Tab 3] A출발점 -> B도착점 입력 시\nDijkstra 기반 안전+관광 결합 최적선 표시"),
        ("5. 추천 명소 리스트", "[Tab 5] Co-occurrence 모델 탑재\n목적지 주변 상위 추천 지역/행사 연계 제시")
    ]
    cw = (W - MARGIN * 2 - 40) / 5
    ch = 180
    cx = MARGIN + 10
    cy = H/2 - 20 - ch/2

    for title, desc in tabs:
        draw_card(c, cx, cy, cw, ch, bg=GRAY_LIGHT)
        c.setFillColor(GREEN_DARK)
        c.setFont(FONT_B, 11)
        c.drawString(cx + 10, cy + ch - 25, title)
        c.setFillColor(GRAY_MID)
        c.setFont(FONT_M, 8.5)
        tx = cx + 10
        ty = cy + ch - 45
        for line in desc.split('\\n'):
            c.drawString(tx, ty, line)
            ty -= 12
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

    draw_card(c, MARGIN + 10, 40, W - MARGIN * 2 - 20, H - 150, bg=BLUE_LIGHT)
    
    pts = [
        "1. 비전 딥러닝 확대: 사용자가 하늘/구름 사진을 찍으면 실시간 Vision CNN 추론을 날씨 예측 모델에 추가하여 정확도 극대화",
        "2. 소비예측/관광매력도의 타겟 분산 교정: 모델 예측 정확도 한계 극복을 위한 파생 피처 엔지니어링 및 데이터 확대 수집 필수",
        "3. 초개인화(Hyper-Personalization) 경로 로직 도입: 성별, 나이, 체질량지수(BMI) 등 사용자 신체 지표를 입력받아 경로 난이도를 자동 조절",
        "4. 멀티테마 관광망: 단순 '관광점수'에서 벗어나 선호도(자연친화/역사탐방/스포츠/문화전시 등)를 고려한 테마 경로 추출 로직 고도화",
        "5. 지역 스케일업(Nationwide): 서울 중심 서비스를 경기, 인천 등 메가시티로 확대하고 전국 데이터 구축으로 인프라 확장",
        "6. 피드백 루프(Continual Learning): 소비자의 '실제 경유 만족도' 및 '사고 위험지역 제보'를 즉각적으로 재학습 파이프라인에 편입하는 선순환 MLOps 구축"
    ]
    py = H - 90
    for pt in pts:
        c.setFillColor(ORANGE)
        c.setFont(FONT_B, 12)
        c.drawString(MARGIN + 30, py, pt[:pt.index(':')+1] if ':' in pt else pt)
        c.setFillColor(GRAY_DARK)
        c.setFont(FONT_M, 11)
        # Handle wrap
        desc = pt[pt.index(':')+1:].strip() if ':' in pt else ""
        if desc:
            c.drawString(MARGIN + 50, py - 18, desc)
            py -= 48
        else:
            py -= 35

"""

def_main = """
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
"""

new_text = before_text + new_slides_code + def_main

with open(FILE_PATH, "w", encoding="utf-8") as f:
    f.write(new_text)

print("INJECTION COMPLETE")
