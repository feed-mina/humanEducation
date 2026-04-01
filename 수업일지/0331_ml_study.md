
회귀모델
선형회귀
LinearRegression

다중선형회귀
독립변수가 여러개이다

다항선형회귀 (곡선)
PolynomialFeatures

SGDRegressor
경사하강법

분류모델

로직스틱회귀
벚꽃모양 테스트  knn

랜덤포레스트

하이퍼파라미터 튜닝 중요 어떤 값을 잡을꺼야 ??

---

## 2026-03-31 데이터 전처리 실습 (K-Ride ML)

### Step 1: 편의시설 데이터 컬럼 추출

#### 왜 컬럼을 줄이는가?
- 원본: 43컬럼 → 유효 피처: 6개
- 불필요한 차원 제거 = 메모리 절약 + 연산 속도 향상
- ML 모델이 노이즈가 많은 컬럼을 학습하면 성능 저하

#### 핵심 개념: Pandas 컬럼 선택

```python
# 방법 1: 리스트 인덱싱으로 원하는 컬럼만 선택
KEEP_COLS = ["x 좌표", "y 좌표", "거리", "상세정보 값 4"]
df_clean = df[KEEP_COLS].copy()
# .copy() → 원본 df를 건드리지 않고 독립된 새 DataFrame 생성
```

#### 핵심 개념: index=False 로 CSV 저장

```python
df_clean.to_csv("facility_clean.csv", index=False, encoding="utf-8-sig")
# index=False  → 0, 1, 2... 행 번호를 파일에 쓰지 않음
#              → 쓰면 나중에 불필요한 'Unnamed: 0' 컬럼이 생겨버림
# utf-8-sig    → BOM 포함 UTF-8, Excel에서 한글 깨짐 방지
```

#### 파생 피처 생성 (Feature Engineering)

```python
# is_24h: "24시간"이라는 텍스트가 상세정보 어딘가에 있으면 True
df_clean["is_24h"] = df[detail_cols].apply(check_24h, axis=1)
# axis=1 → 행 단위로 함수 적용 (axis=0이면 열 단위)

# has_restricted_hours: 예약/제한 키워드 포함 여부
df_clean["has_restricted_hours"] = df[detail_cols].apply(check_restricted, axis=1)
```

#### 실행 방법

```bash
# kride-project 폴더에서 실행
python step1_facility_clean.py
```

#### 체크포인트

- [ ] `facility_clean.csv` 생성됐는지 확인
- [ ] shape 출력에서 6컬럼인지 확인
- [ ] `is_24h` 분포 확인 (91% 이상이 False면 정상 - research.md 기록과 일치)
- [ ] 결측값 현황 확인 → `install_type`(상세정보 값 4)이 70.9% 결측 예상

#### 다음 단계 (Step 2)
- x/y 좌표 → WGS84(위경도) 좌표계 변환 확인 (EPSG:5179인지 체크)
