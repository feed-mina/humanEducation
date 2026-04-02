# Streamlit 포트폴리오 배포 가이드

## 전체 흐름 요약

```
① Streamlit 앱 → GitHub 업로드
② GitHub → Streamlit Community Cloud 배포 (공개 URL 확보)
③ Next.js 프로젝트 생성 → 프로젝트 URL 연결
④ Next.js → GitHub → Vercel 자동 배포
```

---

## Step 1: Streamlit 앱 GitHub 레포 준비

### 1-1. humanEducation 레포에서 Streamlit 앱 경로 확인

현재 앱 파일 위치:
```
colab/
├── streamlit_hr_comma_sep.py
├── streamlit_tips.py
├── model/hr_comma_sep_model.pkl
└── requirements.txt
```

### 1-2. requirements.txt 확인 및 정비

Streamlit Community Cloud 배포에 필요한 최소 의존성:

```txt
streamlit
pandas
scikit-learn
matplotlib
seaborn
```

> 버전을 명시하지 않으면 Cloud가 최신 버전을 설치함. 충돌 시 버전 고정.

### 1-3. 모델 파일(.pkl) GitHub 업로드 여부 결정

| 파일 크기 | 방법 |
| --------- | ---- |
| < 100MB | GitHub에 직접 커밋 |
| > 100MB | Git LFS 또는 앱 실행 시 모델 재학습 |

현재 `hr_comma_sep_model.pkl`은 작은 크기이므로 직접 커밋 가능.

---

## Step 2: Streamlit Community Cloud 배포

### 2-1. 계정 생성

[share.streamlit.io](https://share.streamlit.io) → GitHub 계정으로 로그인

### 2-2. 새 앱 배포

1. **"New app"** 클릭
2. 레포지토리 선택: `humanEducation`
3. 브랜치: `main`
4. Main file path: `colab/streamlit_hr_comma_sep.py`
5. **"Deploy!"** 클릭

### 2-3. 배포 URL 메모

배포 완료 후 URL 형식:
```
https://<앱이름>.streamlit.app
```

이 URL을 `data/projects.ts`의 `streamlitUrl` 필드에 입력.

---

## Step 3: Next.js 프로젝트 생성

### 3-1. 새 프로젝트 생성

터미널에서 실행:

```bash
cd c:/Users/Samsung/Documents/Development/Personal_Projects/2026
npx create-next-app@latest streamlit-portfolio --typescript --tailwind --app --eslint
cd streamlit-portfolio
```

### 3-2. 프로젝트 메타데이터 파일 생성

`data/projects.ts` 파일 생성:

```typescript
export type Project = {
  slug: string;
  title: string;
  description: string;
  tags: string[];
  streamlitUrl: string;
  thumbnail?: string;
};

export const projects: Project[] = [
  {
    slug: "hr-attrition",
    title: "직원 이직 예측 모델",
    description: "HR 데이터를 기반으로 직원의 이직 가능성을 예측. RandomForest 분류 모델 사용.",
    tags: ["분류", "RandomForest", "HR"],
    streamlitUrl: "https://PLACEHOLDER.streamlit.app",  // Step 2 완료 후 업데이트
  },
  {
    slug: "tips-regression",
    title: "팁 금액 예측 모델",
    description: "레스토랑 결제 데이터로 팁 금액을 예측하는 회귀 분석 앱. EDA 시각화 포함.",
    tags: ["회귀", "EDA", "Seaborn"],
    streamlitUrl: "https://PLACEHOLDER.streamlit.app",
  },
];
```

### 3-3. ProjectCard 컴포넌트 생성

`components/ProjectCard.tsx`:

```tsx
import Link from "next/link";
import { Project } from "@/data/projects";

export function ProjectCard({ project }: { project: Project }) {
  return (
    <div className="border rounded-xl p-6 hover:shadow-lg transition-shadow">
      <h2 className="text-xl font-bold mb-2">{project.title}</h2>
      <p className="text-gray-600 mb-4">{project.description}</p>
      <div className="flex gap-2 mb-4 flex-wrap">
        {project.tags.map((tag) => (
          <span key={tag} className="bg-blue-100 text-blue-800 text-sm px-2 py-1 rounded">
            {tag}
          </span>
        ))}
      </div>
      <a
        href={project.streamlitUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-block bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
      >
        앱 실행 →
      </a>
    </div>
  );
}
```

### 3-4. 메인 페이지 수정

`app/page.tsx`:

```tsx
import { projects } from "@/data/projects";
import { ProjectCard } from "@/components/ProjectCard";

export default function Home() {
  return (
    <main className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-2">ML 프로젝트 포트폴리오</h1>
      <p className="text-gray-500 mb-10">Streamlit으로 만든 머신러닝 앱 모음</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {projects.map((project) => (
          <ProjectCard key={project.slug} project={project} />
        ))}
      </div>
    </main>
  );
}
```

### 3-5. 로컬 실행 확인

```bash
npm run dev
# http://localhost:3000 접속 확인
```

---

## Step 4: GitHub 레포 생성 및 push

```bash
# streamlit-portfolio 폴더 안에서
git init
git add .
git commit -m "feat: 초기 Streamlit 포트폴리오 Next.js 프로젝트"
```

GitHub에서 `streamlit-portfolio` 레포 생성 후:

```bash
git remote add origin https://github.com/<username>/streamlit-portfolio.git
git branch -M main
git push -u origin main
```

---

## Step 5: Vercel 배포

### 5-1. Vercel 프로젝트 연결

1. [vercel.com](https://vercel.com) 로그인 (GitHub 계정)
2. **"Add New Project"**
3. `streamlit-portfolio` 레포 선택 → **"Deploy"**

### 5-2. 자동 배포 확인

- 배포 완료 URL: `https://streamlit-portfolio-<hash>.vercel.app`
- 이후 `main` 브랜치 push 시 자동 재배포

### 5-3. 커스텀 도메인 설정 (선택)

Vercel 대시보드 → Project Settings → Domains → 원하는 도메인 입력

---

## 체크리스트

```
Phase 1 - Streamlit 배포
[ ] humanEducation 레포 colab/ 폴더 정리
[ ] requirements.txt 확인
[ ] streamlit_hr_comma_sep.py 배포 완료 → URL 메모
[ ] streamlit_tips.py 배포 완료 → URL 메모

Phase 2 - Next.js 개발
[ ] create-next-app 실행
[ ] data/projects.ts 에 실제 URL 업데이트
[ ] ProjectCard 컴포넌트 구현
[ ] npm run dev 로컬 확인

Phase 3 - Vercel 배포
[ ] GitHub 레포 push
[ ] Vercel 프로젝트 연결
[ ] 배포 URL 접속 확인
```

---

## 트러블슈팅

| 문제 | 원인 | 해결 |
| ---- | ---- | ---- |
| Streamlit Cloud에서 모델 로드 실패 | `.pkl` 파일 미포함 | GitHub에 `model/` 폴더 커밋 확인 |
| `ModuleNotFoundError` | requirements.txt 누락 | 패키지 이름 확인 후 추가 |
| iframe 차단 | Streamlit의 X-Frame-Options | `streamlitUrl` 링크 방식으로 변경 |
| Vercel 빌드 실패 | TypeScript 오류 | `npm run build` 로컬에서 먼저 확인 |
