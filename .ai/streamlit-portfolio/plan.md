# Streamlit 포트폴리오 Next.js 프로젝트 계획

## 목표

수업에서 만든 Streamlit 앱들을 한 곳에서 보여주는 **Next.js 포트폴리오 사이트**를 만들고 Vercel에 배포한다.

> Streamlit 앱은 Python 런타임이 필요하므로 Vercel에서 직접 실행 불가.
> Next.js 사이트는 **포트폴리오 허브** 역할 — Streamlit Community Cloud에 배포된 앱을 링크/임베드.

---

## 아키텍처

```
Vercel (Next.js)
└── 포트폴리오 허브 (프로젝트 카드, 설명, 링크)
        │
        ├──▶ Streamlit Community Cloud — streamlit_hr_comma_sep.py
        ├──▶ Streamlit Community Cloud — streamlit_tips.py
        └──▶ Streamlit Community Cloud — streamlit_iris (추후 추가)
```

---

## 프로젝트 폴더 위치

**실제 생성 위치 (새 레포):**
```
c:/Users/Samsung/Documents/Development/Personal_Projects/2026/streamlit-portfolio/
```
> humanEducation 레포와 분리하여 별도 GitHub 레포로 관리. Vercel은 이 레포를 바라봄.

**.ai 문서 위치 (계획/가이드):**
```
humanEducation/.ai/streamlit-portfolio/
├── plan.md    ← 현재 파일
└── guide.md   ← 배포 단계별 가이드
```

---

## 현재 Streamlit 프로젝트 목록

| 파일 | 데이터셋 | 주요 기능 |
| ---- | -------- | --------- |
| `streamlit_hr_comma_sep.py` | HR Comma Sep | 직원 이직 예측 (분류 모델) |
| `streamlit_tips.py` | Tips | 팁 금액 예측 (회귀 모델) |
| `민예린_4월1일_streamlit과제.ipynb` | HR Comma Sep | 수업 과제 버전 |

---

## 단계별 계획

### Phase 1: Streamlit 앱 배포 (Streamlit Community Cloud)

- [ ] GitHub에 `streamlit-apps` 레포 생성 (또는 humanEducation 레포 활용)
- [ ] `requirements.txt` 정비 (현재 colab/requirements.txt 확인 필요)
- [ ] [share.streamlit.io](https://share.streamlit.io) 에서 각 앱 배포
  - `streamlit_hr_comma_sep.py` → 배포 URL 확보
  - `streamlit_tips.py` → 배포 URL 확보
- [ ] 모델 파일(`.pkl`)이 GitHub에 올라가야 Streamlit Cloud에서 실행 가능 여부 확인

### Phase 2: Next.js 프로젝트 생성

- [ ] `npx create-next-app@latest streamlit-portfolio --typescript --tailwind --app` 실행
- [ ] 폴더 구조 확정 (아래 참고)
- [ ] 프로젝트 카드 컴포넌트 구현 (`ProjectCard.tsx`)
- [ ] 메인 페이지에 프로젝트 목록 렌더링
- [ ] (선택) iframe으로 Streamlit 앱 임베드

### Phase 3: Vercel 배포

- [ ] GitHub에 `streamlit-portfolio` 레포 push
- [ ] [vercel.com](https://vercel.com) 에서 레포 연결 → 자동 배포
- [ ] 커스텀 도메인 설정 (선택)

---

## Next.js 폴더 구조

```text
streamlit-portfolio/
├── app/
│   ├── layout.tsx          ← 공통 레이아웃 (헤더, 푸터)
│   ├── page.tsx            ← 메인 페이지 (프로젝트 카드 목록)
│   └── projects/
│       └── [slug]/
│           └── page.tsx    ← 개별 프로젝트 상세 페이지 (iframe 임베드)
├── components/
│   └── ProjectCard.tsx     ← 프로젝트 카드 컴포넌트
├── data/
│   └── projects.ts         ← 프로젝트 메타데이터 (제목, 설명, URL, 태그)
├── public/
│   └── thumbnails/         ← 앱 스크린샷
└── package.json
```

---

## 프로젝트 메타데이터 예시 (`data/projects.ts`)

```typescript
export const projects = [
  {
    slug: "hr-attrition",
    title: "직원 이직 예측 모델",
    description: "HR 데이터를 기반으로 직원의 이직 가능성을 예측하는 머신러닝 앱",
    tags: ["분류", "RandomForest", "HR"],
    streamlitUrl: "https://share.streamlit.io/...",  // 배포 후 업데이트
    thumbnail: "/thumbnails/hr-attrition.png",
  },
  {
    slug: "tips-regression",
    title: "팁 금액 예측 모델",
    description: "레스토랑 데이터를 기반으로 팁 금액을 예측하는 회귀 분석 앱",
    tags: ["회귀", "LinearRegression", "EDA"],
    streamlitUrl: "https://share.streamlit.io/...",
    thumbnail: "/thumbnails/tips.png",
  },
];
```

---

## 우선순위 매트릭스

| 우선순위 | 작업 | 완료 기준 |
| -------- | ---- | --------- |
| P0 | Streamlit 앱 GitHub 업로드 | 레포에 .py 파일 + requirements.txt 존재 |
| P0 | Streamlit Community Cloud 배포 | 공개 URL 생성 완료 |
| P1 | Next.js 프로젝트 생성 | `npm run dev` 로컬 실행 확인 |
| P1 | 프로젝트 카드 UI 구현 | 카드 목록 렌더링 확인 |
| P1 | Vercel 배포 | 공개 URL 접속 확인 |
| P2 | iframe 임베드 | 상세 페이지에서 앱 실행 확인 |
| P2 | 스크린샷 썸네일 추가 | 각 카드에 이미지 표시 |

---

## 제약사항 및 주의사항

- **모델 파일 크기**: `.pkl` 파일이 크면 GitHub LFS 또는 외부 스토리지(S3) 필요
- **Streamlit iframe**: CORS 설정에 따라 iframe 임베드가 차단될 수 있음 → 링크로 대체 가능
- **requirements.txt**: `scikit-learn`, `pandas`, `streamlit` 버전 명시 필요
- 시크릿/API 키 절대 커밋 금지 (`.env` 파일로 관리)
