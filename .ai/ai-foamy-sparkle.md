# K-Ride PWA 프론트엔드 구현 계획 (SDUI 통합)

## Context
Netflix 클론(Next.js 14, App Router, TypeScript, Tailwind, DaisyUI)을 K-Culture 여행 추천 PWA로 전환한다.
기존 SDUI 프로젝트의 DynamicEngine + componentMap 패턴을 도입하여, 아티스트/지역 목록 같은 동적 콘텐츠를
FastAPI → Redis → Supabase 파이프라인으로 서버에서 제공한다.
로그인/회원가입은 그대로 유지하고, 5단계 온보딩 → AI 추천 → 지도 일정 화면으로 이어진다.

---

## 메모 답변 (먼저 읽기)

### Zustand persist 미들웨어란?
Zustand의 `persist` 미들웨어는 스토어 상태를 자동으로 `localStorage`에 직렬화해서 저장한다.
페이지 새로고침해도 온보딩 선택값(여행기간, 아티스트 등)이 사라지지 않는다.
```typescript
import { persist } from 'zustand/middleware'
const useStore = create(persist(
  (set) => ({ duration: null, setDuration: (d) => set({ duration: d }) }),
  { name: 'kride-onboarding' }  // localStorage key 이름
))
```

### React Query + Redis 도입 전략
- **React Query** (이미 설치됨): 서버 상태(FastAPI API 응답) 관리. SDUI 메타데이터 fetch에 사용
- **Redis**: FastAPI 백엔드에서 UI 메타데이터를 1시간 캐싱 → SDUI 프로젝트와 동일한 패턴
- **Zustand**: 클라이언트 온보딩 상태(사용자 선택값)만 관리. 서버 상태는 React Query가 담당

```
[분리 원칙]
Zustand persist  → 온보딩 선택값 (로컬, localStorage)
React Query      → API 데이터 (서버: 아티스트/지역 목록, AI 추천 결과)
Redis (FastAPI)  → UI 메타데이터 + 아티스트/지역 데이터 캐시 (TTL 1시간)
```

### Focus 사이드패널 일정 구조 (아코디언)
```
당일치기:
  └─ 하루
      ├─ [오전] ▼ (접기/펼치기)
      │    ├─ 장소1 ── 장소2 ── 장소3
      └─ [오후] ▼
           ├─ 장소4 ── 장소5

1박 2일:
  ├─ 첫째날
  │   ├─ [오전] ▼
  │   └─ [오후] ▼
  └─ 둘째날
      ├─ [오전] ▼
      └─ [오후] ▼

2박 3일: 동일 패턴, 셋째날 추가
```
각 날/시간대는 DaisyUI `collapse` 컴포넌트로 구현 (아코디언, 접었다 펼치기 가능)

### Jest 테스트
각 화면 구현 후 Jest 단위 테스트 작성. 각 Phase 완료 후 진행.

---
### [메모] 기존의 SDUI 프로젝트에 이어서 작업할 예정입니다. 따라서 DB추가, 서버는 MSA 로 진행될것같습니다. SDUI 의 법칙은 select, section, input, text 등등 웹이나 앱에서 사용하는 가장 작은 단위의 요소를 만들고 그걸 블록처럼 쌓아서 페이지를 만드는데 매력이 있습니다.  프로젝트에 사용하는 가장 작은 단위 요소를 사용하여 페이지를 구성하는데 있고 (DOM 트리모드), 그리고 screen_id로 관리하는 ui_metadata에서 테이블에서 db값을 바꾸면 작은단위 요소에 적용이 될 수 있고 이렇게 쉽게 페이지 수정이 가능할수 있도록 하는 부분이 핵심이라고 생각합니다. 추가적으로 기본단위가 필요한 부분이 있다면 만들어야 하고 그걸 블록처럼 쌓아서 페이지를 만들수 있도록 해야 합니다. 


## SDUI 통합 전략 (K-Ride 적합 하이브리드)

SDUI 프로젝트의 전체 Spring Boot 서버를 이식하지 않고, 핵심 패턴만 Python/FastAPI로 경량 구현한다.


| 항목 | SDUI 원본 | K-Ride 적용 |
|------|-----------|-------------|
| 백엔드 | Spring Boot + Java | FastAPI + Python (기존 프로젝트) |
| DB | PostgreSQL (별도) | Supabase (기존) |
| 캐시 | Redis (Docker) | Redis (FastAPI) |
| 렌더링 엔진 | DynamicEngine (전체) | 경량 DynamicEngine (인트로 화면 한정) |
| componentMap | 18개 타입 | K-Ride 전용 6개 타입 |
| RBAC | allowed_roles 필터 | 현재 미적용 (MVP) |
| 동적 데이터 | ui_metadata + query_master | 아티스트/지역만 DB화 (나머지 하드코딩) |

### 무엇을 SDUI로 만드나?
- **동적 (DB에서 가져오기)**: 아티스트 목록, 지역 목록 → Neo4j / Supabase에서 서버가 제공
- **정적 (하드코딩 유지)**: 인트로 화면 레이아웃, 여행 목적 체크박스 6개, 버튼 텍스트

이유: 아티스트 20개, 지역 10개는 자주 바뀔 수 있는 데이터. 배포 없이 DB만 수정하면 반영되도록.

---

## 화면 흐름

```
/ (랜딩) → /login → /signup
                       ↓ 로그인 완료
/browse (Intro1) → /movies (Intro2) → /latest (Intro3) → /intro4 → /intro5 → /my-list → /focus
```

---

## Phase 0: 패키지 설치 (사용자가 직접 실행)

```bash
cd netflix-clone-main
npm install zustand leaflet react-leaflet @types/leaflet next-pwa
# (react-query, axios는 이미 설치됨 확인)
```

FastAPI 쪽 (kride-project 루트):
```bash
pip install redis
```

---

## Phase 1: 기반 작업

### 1-1. Supabase `ui_content` 테이블 추가
기존 SDUI의 `ui_metadata`를 단순화. Supabase SQL Editor에서 실행:
```sql
CREATE TABLE kride_content (
  id          SERIAL PRIMARY KEY,
  content_type VARCHAR(20),   -- 'artist' | 'region'
  name        VARCHAR(100),
  image_url   VARCHAR(500),
  tags        TEXT[],          -- 아티스트: ['idol', 'kpop'], 지역: ['자연', '해변']
  is_active   BOOLEAN DEFAULT true,
  sort_order  INT DEFAULT 0
);
-- 초기 데이터 INSERT는 Phase 2에서
```

### 1-2. FastAPI 엔드포인트 추가 (기존 FastAPI 서버에 추가)
**파일:** `kride-project/api/content_router.py` (신규)
```python
# GET /api/content/{content_type}
# Redis 캐시 조회 → 없으면 Supabase 쿼리 → Redis 저장(TTL 3600s)
# 반환: [{ id, name, image_url, tags }]
```

### 1-3. Zustand 온보딩 스토어
**신규:** `src/store/onboarding-store.ts`
```typescript
// persist 미들웨어로 localStorage 동기화
type TravelDuration = 'day' | 'onenight' | 'twonight'
type TravelPurpose = 'food' | 'kculture' | 'nature' | 'history' | 'shopping' | 'rest'
interface BudgetRange { min: number; max: number }
interface ContentItem { id: number; name: string; imageUrl: string }

interface OnboardingState {
  duration: TravelDuration | null
  selectedArtists: ContentItem[]
  selectedRegions: ContentItem[]
  purposes: TravelPurpose[]
  budget: BudgetRange
  // actions
  setDuration, toggleArtist, toggleRegion, togglePurpose, setBudget, reset
}
```

### 1-4. K-Ride componentMap (SDUI 패턴 경량 적용)
**신규:** `src/components/kride/componentMap.ts`
```typescript
export const componentMap = {
  SELECTION_CARD: SelectionCard,    // 아티스트/지역 카드 (circle/square)
  PURPOSE_CARD: PurposeCard,        // 여행목적 체크 카드
  BUDGET_SLIDER: BudgetSlider,      // 듀얼 range slider
  NEXT_BUTTON: NextArrowButton,     // 화살표 다음 버튼
  DURATION_BUTTON: DurationButton,  // 당일치기/1박2일/2박3일
  ITINERARY_NODE: ItineraryNode,    // 일정 노드 (아코디언 내부)
}
```

### 1-5. 헤더 조건부 숨김
**신규:** `src/app/(afterLogin)/_component/conditional-header.tsx` (Client Component)
```typescript
"use client";
const HIDE_PATHS = ['/browse', '/movies', '/latest', '/intro4', '/intro5'];
// usePathname() 정확 일치 시 null 반환
```
**수정:** `src/app/(afterLogin)/layout.tsx` → `<Header />` → `<ConditionalHeader />`

---

## Phase 2: 인트로 5단계 페이지

### Intro1 — `/browse/page.tsx` (교체)
- 검은 배경, 중앙 정렬
- "어떤 여행을 떠나실 건가요?"
- 버튼 3개: 당일치기 / 1박 2일 / 2박 3일
  - DaisyUI `btn btn-outline btn-lg` + Netflix 레드(`border-red-600 hover:bg-red-600`)
- 클릭 → `setDuration()` → `router.push('/movies')`
- **Jest 테스트**: 버튼 3개 렌더링, 클릭 시 duration 설정 확인

### Intro2 — `/movies/page.tsx` (신규, `/movies/[id]/` 유지)
- React Query로 `GET /api/content/artist` fetch
- 로딩 중: skeleton grid 표시
- "좋아하는 아이돌/배우를 선택하세요" + 카운트 "X / 5"
- SelectionCard (circle 모드): 선택 시 `ring-4 ring-red-600`
- 5개 초과 클릭 → DaisyUI toast 경고
- 1개 미만이면 NextArrowButton disabled
- 화살표 버튼 → `/latest`
- **Jest 테스트**: 5개 초과 선택 방지, 1개 미만 버튼 비활성화

### Intro3 — `/latest/page.tsx` (교체)
- React Query로 `GET /api/content/region` fetch
- SelectionCard (square 모드, aspect-square + 이름 오버레이)
- Intro2와 동일 선택 로직 (1-5개)
- 화살표 버튼 → `/intro4`
- **Jest 테스트**: Intro2와 동일 패턴

### Intro4 — `/intro4/page.tsx` (신규)
- 검은 배경, "여행의 목적은?"
- PurposeCard 6개 (클릭형 카드 체크박스):
  맛집탐방 / K-Culture(촬영지) / 자연/트레킹 / 역사/문화 / 쇼핑 / 휴양
- 선택 시 `bg-red-900 border-red-600`
- 화살표 버튼 → `/intro5`
- **Jest 테스트**: 다중 선택, 토글 해제 확인

### Intro5 — `/intro5/page.tsx` (신규)
- "내가 생각하는 여행 예산은?"
- 듀얼 Range Slider (`<input type="range">` 2개 겹침)
  - 범위: 30,000 ~ 2,000,000원
  - 트랙: `linear-gradient(to right, #333 {minPct}%, #e50914 {minPct}%, #e50914 {maxPct}%, #333 {maxPct}%)`
  - 금액: `toLocaleString('ko-KR')` + ₩ 접두사
- 화살표 버튼 → `/my-list`
- **Jest 테스트**: 슬라이더 값 범위 경계 테스트

---

## Phase 3: 결과 화면

### My-list — `/my-list/page.tsx` (교체)
- Zustand에서 5개 선택값 읽기
- `duration === null`이면 `/browse` redirect
- 요약 카드 5개: 여행기간 / 아티스트(avatar grid) / 지역(태그) / 목적(badge) / 예산(범위)
- AI 추천 배너 (클릭 가능, 빨간 그라디언트 테두리):
  ```
  "AI기반으로 방문 경로를 추천해드려요.
   전주 한옥마을을 추천드립니다"
  ```
  → `router.push('/focus')`
- **Jest 테스트**: 온보딩 미완료 시 리다이렉트 확인

### Focus — `/focus/page.tsx` (신규)
레이아웃: `flex h-screen` (좌 60% 지도 / 우 40% 사이드패널)

**지도 (좌측):**
```typescript
const MapComponent = dynamic(() => import('./_component/map-component'), { ssr: false })
// react-leaflet MapContainer + OpenStreetMap TileLayer
// 전주 한옥마을: lat 35.8150, lng 127.1530
// 경로 노드 Marker + Popup
```

**사이드패널 (우측) — 아코디언 구조:**
```
[여행지명] 전주 한옥마을 여행

[당일치기]
  └─ 하루
      ├─ [오전] <DaisyUI collapse>
      │    └─ ItineraryNode: 경기전 → 전동성당 → 오목대
      └─ [오후] <DaisyUI collapse>
           └─ ItineraryNode: 한옥마을 공방 → 남부시장

[주변 맛집]
  └─ RestaurantCard: 이름, 거리, 카테고리
```
- DaisyUI `collapse collapse-arrow` 컴포넌트로 오전/오후 접기
- `duration` 값에 따라 날짜 구조 동적 생성 (당일치기: 1일, 1박2일: 2일, 2박3일: 3일)
- 초기: 정적 Mock 데이터 (추후 FastAPI GraphRAG 연동)
- **Jest 테스트**: duration별 날짜 구조 렌더링 확인

---

## Phase 4: 인프라 설정

### 미들웨어 — `src/middleware.ts` 수정
```typescript
return NextResponse.redirect(new URL('/login', request.url));  // broken-netflix.com 제거

matcher: ['/my-list', '/browse', '/movies', '/latest', '/intro4', '/intro5', '/focus',
          '/profile/account', '/profile/manage']
```

### next.config.js 수정
```javascript
const withPWA = require('next-pwa')({
  dest: 'public',
  disable: process.env.NODE_ENV === 'development'
});
// images.domains에 'images.unsplash.com' 추가
module.exports = withPWA(nextConfig);
```

### PWA 파일
**신규:** `public/manifest.json`
```json
{
  "name": "K-Ride", "short_name": "K-Ride",
  "start_url": "/browse", "display": "standalone",
  "background_color": "#141414", "theme_color": "#e50914"
}
```
**수정:** `src/app/layout.tsx` → manifest 메타데이터 추가

---

## 핵심 파일 목록

| 파일 | 작업 |
|------|------|
| `kride-project/api/content_router.py` | 신규 (FastAPI SDUI 콘텐츠 API) |
| `src/store/onboarding-store.ts` | 신규 (Zustand + persist) |
| `src/components/kride/componentMap.ts` | 신규 (SDUI 경량 컴포넌트맵) |
| `src/app/(afterLogin)/_component/conditional-header.tsx` | 신규 |
| `src/app/(afterLogin)/_component/selection-card.tsx` | 신규 (circle/square) |
| `src/app/(afterLogin)/_component/next-arrow-button.tsx` | 신규 |
| `src/app/(afterLogin)/layout.tsx` | 수정 |
| `src/app/(afterLogin)/browse/page.tsx` | 교체 (Intro1) |
| `src/app/(afterLogin)/movies/page.tsx` | 신규 (Intro2) |
| `src/app/(afterLogin)/latest/page.tsx` | 교체 (Intro3) |
| `src/app/(afterLogin)/intro4/page.tsx` | 신규 |
| `src/app/(afterLogin)/intro5/page.tsx` | 신규 |
| `src/app/(afterLogin)/my-list/page.tsx` | 교체 |
| `src/app/(afterLogin)/focus/page.tsx` | 신규 |
| `src/app/(afterLogin)/focus/_component/map-component.tsx` | 신규 (Leaflet) |
| `src/middleware.ts` | 수정 |
| `next.config.js` | 수정 (PWA) |
| `public/manifest.json` | 신규 |

---

## 주의 사항

- Leaflet SSR 불가 → `dynamic(..., { ssr: false })` 필수
- Zustand persist hydration mismatch → `hasHydrated` 플래그 패턴
- `/movies/page.tsx` + `/movies/[id]/page.tsx` 충돌 없음 (App Router 정상)
- SDUI componentMap은 K-Ride 전용 6개만 구현 (원본 18개 전체 이식 X)
- 이미지 파일은 Supabase Storage 또는 `/public/images/` 배치

## 검증 방법

1. `npm run dev` → 로그인 → `/browse` Intro1 3버튼 확인
2. 버튼 클릭 → `/movies` → React Query로 아티스트 목록 로드 확인 (Network 탭)
3. 1~5개 선택 제한 동작 확인 → 화살표 → `/latest` → 지역 선택
4. `/intro4` 목적 체크 → `/intro5` 슬라이더 → `/my-list` 요약 표시
5. AI 추천 배너 클릭 → `/focus` 지도 + duration별 아코디언 일정 패널
6. `npx jest` → 각 화면 단위 테스트 통과 확인
7. Chrome DevTools → Application → Manifest PWA 등록 확인
