# Designer — Research

> 이 파일은 UI/UX 설계를 위한 현재 스타일 시스템 분석 결과를 기록한다.

---

## 현재 스타일 시스템 분석 (2026-02-28 기준)

### 기술 스택

| 항목 | 기술 |
|------|------|
| CSS 프레임워크 | Tailwind CSS 4.0.0 |
| 커스텀 클래스 | globals.css + 컴포넌트별 CSS |
| 클래스 유틸 | clsx 2.1.1 (`cn.tsx`) |
| 반응형 감지 | `useDeviceType` 훅 (mobile/pc) |
| 아이콘 | 별도 라이브러리 없음 — emoji 직접 사용 (📔, → 등) |

---

### SDUI 레이아웃 시스템

#### 최외곽 컨테이너 (항상 고정)
```
engine-container {deviceClass}
  deviceClass: "mobile" | "pc"
```

#### 그룹 레이아웃 클래스
```
group_direction=ROW    → .flex-row-layout
group_direction=COLUMN → .flex-col-layout
```

#### 컴포넌트별 확인된 CSS 클래스
| 컴포넌트 | CSS 클래스 |
|----------|-----------|
| InputField | `.inputfield-core`, `.readonly-style` |
| ButtonField | (동적 css_class) |
| Modal | (동적 css_class) |
| DynamicEngine | `.engine-container` |

---

### 반응형 설계 현황

#### useDeviceType 훅
- 반환값: `"mobile"` | `"pc"`
- 사용처: `DynamicEngine` className 분기, `Header.tsx` 레이아웃 분기

#### Header 반응형
- 모바일: 햄버거 메뉴 (모바일 전용 스타일)
- PC: 사이드바 네비게이션 (`Sidebar.tsx`)
- `Header.tsx`: 최근 커밋(70db7f4)에서 모바일 대응 로직 개선됨

#### AppShell 구조
```
AppShell
  ├── Header (top: mobile hamburger / pc sidebar)
  └── main content area (DynamicEngine)
```

---

### 로딩 상태 시스템
| 컴포넌트 | 용도 |
|----------|------|
| `Skeleton.tsx` | 기본 스켈레톤 블록 |
| `SkeletonLoader.tsx` | 화면별 스켈레톤 조합 |

---

### 현재 화면별 레이아웃 패턴

#### MAIN_PAGE (벤토 그리드 — 2026-03-06 V8 전환)
- 3열 grid 레이아웃 (모바일: 1열)
- USER: 약속위젯(col 1-2) + 콘텐츠쓰기(col 3) + 콘텐츠보기(full)
- GUEST: 목표미설정위젯(col 1-2) + 로그인유도(col 3) + 튜토리얼(full)

```
.main-bento             → 3열 grid 컨테이너
.bento-card             → 공통 카드 (overflow: visible)
.col-span-2 / .col-span-3 → 그리드 점유
.bento-card-appointment / no-goal / diary / login / dark → 타입별 배경
.bento-card-body/icon/title/desc/arrow/tag → 카드 내부
```

> 파일 위치: `app/styles/pages.css` (line 1715~)

#### CONTENT_LIST (기존 DIARY_LIST)
- 카드 목록 레이아웃 (COLUMN 방향)
- 필터 토글 (`FilterToggle.tsx`)
- 페이지네이션 (`Pagination.tsx`)
- 모바일: 스크롤 단일 컬럼 / PC: 그리드 레이아웃

#### CONTENT_WRITE / CONTENT_MODIFY (기존 DIARY_WRITE / DIARY_MODIFY)
- 폼 레이아웃 (COLUMN 방향)
- EmotionSelectField (감정 선택 UI)
- TimeSlotRecord (시간 슬롯 기록)
- ADDRESS_SEARCH_GROUP (주소 검색 — 다음 우편번호 API)

#### LOGIN_PAGE
- 중앙 정렬 카드 레이아웃
- SNS_BUTTON (카카오 로그인 — 노란색)

#### SET_TIME_PAGE
- 타임 휠 피커 (DateTimePicker, TimeSelect)

---

### 갭(Gap) 분석 — 디자인 관점 (2026-03-06 업데이트)

1. **css_class 사전 없음**: 사용 가능한 클래스 목록이 문서화되지 않음 → 개발자마다 클래스명 혼용 가능성
2. **디자인 토큰**: Tailwind 기본값 사용 — `globals.css`에 CSS 변수 + Tailwind import만 존재, 커스텀 토큰 없음 확인
3. **다크 모드**: 현재 미구현 (향후 DB의 css_class로 제어 가능)
4. **애니메이션**: 페이지 전환, 모달 열림/닫힘 애니메이션 정의 없음

---

## 분석 히스토리

| 날짜 | 분석 내용 | 결론 |
|------|-----------|------|
| 2026-02-28 | 전체 스타일 시스템 초기 분석 | 위 내용 도출 |
| 2026-03-06 | 아이콘/디자인토큰 확인, MAIN_PAGE 벤토 그리드 반영 | emoji 직접 사용 확인, Tailwind 기본값 확인, 벤토 CSS 클래스 추가 |
