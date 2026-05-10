# diary → content 프론트엔드 마이그레이션 계획

**작성자**: Frontend Engineer
**작성일**: 2026-03-01
**목적**: diary 관련 프론트엔드 코드를 content로 전환

---

## 📋 수정 대상 파일 (8개)

### 1. 화면 ID 매핑
- `components/constants/screenMap.ts`

### 2. 비즈니스 액션 핸들러
- `components/DynamicEngine/hook/useBusinessActions.tsx`

### 3. 사용자 액션 핸들러
- `components/DynamicEngine/hook/useUserActions.tsx`

### 4. 레이아웃 컴포넌트
- `components/layout/Header.tsx`

### 5. 페이지 보호 설정
- `app/view/[...slug]/page.tsx`

### 6. 메타데이터 훅
- `components/DynamicEngine/hook/usePageMetadata.tsx`

### 7. 버튼 필드
- `components/fields/ButtonField.tsx`

### 8. 테스트 파일
- `tests/rendering_optimization.test.tsx`

---

## 🔧 상세 수정 사항

### 1. screenMap.ts

**파일 경로**: `components/constants/screenMap.ts`

#### 수정 내용
```typescript
// 변경 전
export const SCREEN_MAP: Record<string, string> = {
  "MAIN_PAGE": "MAIN_PAGE",
  "LOGIN_PAGE": "LOGIN_PAGE",
  "DIARY_LIST": "DIARY_LIST",
  "DIARY_WRITE": "DIARY_WRITE",
  "DIARY_DETAIL": "DIARY_DETAIL",
  // ...
};

// 변경 후
export const SCREEN_MAP: Record<string, string> = {
  "MAIN_PAGE": "MAIN_PAGE",
  "LOGIN_PAGE": "LOGIN_PAGE",
  "CONTENT_LIST": "CONTENT_LIST",
  "CONTENT_WRITE": "CONTENT_WRITE",
  "CONTENT_DETAIL": "CONTENT_DETAIL",
  // ...
};
```

#### 변경 항목
- `DIARY_LIST` → `CONTENT_LIST`
- `DIARY_WRITE` → `CONTENT_WRITE`
- `DIARY_DETAIL` → `CONTENT_DETAIL`

---

### 2. useBusinessActions.tsx

**파일 경로**: `components/DynamicEngine/hook/useBusinessActions.tsx`

#### 수정 내용
```typescript
// 변경 전
case "DIARY_WRITE_SUBMIT":
    try {
        const res = await axios.post('/api/diary/write', {
            title: formData.title,
            content: formData.content,
            date: formData.date,
        });
        if (res.status === 200) {
            alert('일기 작성 완료');
            router.push('/view/DIARY_LIST');
        }
    } catch (error) {
        alert('일기 작성 실패');
    }
    break;

case "DIARY_UPDATE":
    try {
        const res = await axios.put(`/api/diary/${formData.diaryId}`, {
            title: formData.title,
            content: formData.content,
        });
        if (res.status === 200) {
            alert('일기 수정 완료');
            router.push('/view/DIARY_LIST');
        }
    } catch (error) {
        alert('일기 수정 실패');
    }
    break;

case "DIARY_DELETE":
    try {
        const res = await axios.delete(`/api/diary/${formData.diaryId}`);
        if (res.status === 200) {
            alert('일기 삭제 완료');
            router.push('/view/DIARY_LIST');
        }
    } catch (error) {
        alert('일기 삭제 실패');
    }
    break;

// 변경 후
case "CONTENT_WRITE_SUBMIT":
    try {
        const res = await axios.post('/api/content/write', {
            title: formData.title,
            contentText: formData.contentText,  // content → contentText
            date: formData.date,
        });
        if (res.status === 200) {
            alert('콘텐츠 작성 완료');
            router.push('/view/CONTENT_LIST');
        }
    } catch (error) {
        alert('콘텐츠 작성 실패');
    }
    break;

case "CONTENT_UPDATE":
    try {
        const res = await axios.put(`/api/content/${formData.contentId}`, {
            title: formData.title,
            contentText: formData.contentText,
        });
        if (res.status === 200) {
            alert('콘텐츠 수정 완료');
            router.push('/view/CONTENT_LIST');
        }
    } catch (error) {
        alert('콘텐츠 수정 실패');
    }
    break;

case "CONTENT_DELETE":
    try {
        const res = await axios.delete(`/api/content/${formData.contentId}`);
        if (res.status === 200) {
            alert('콘텐츠 삭제 완료');
            router.push('/view/CONTENT_LIST');
        }
    } catch (error) {
        alert('콘텐츠 삭제 실패');
    }
    break;
```

#### 변경 항목
- 액션 타입: `DIARY_WRITE_SUBMIT` → `CONTENT_WRITE_SUBMIT`
- 액션 타입: `DIARY_UPDATE` → `CONTENT_UPDATE`
- 액션 타입: `DIARY_DELETE` → `CONTENT_DELETE`
- API URL: `/api/diary/*` → `/api/content/*`
- 필드명: `formData.content` → `formData.contentText`
- 필드명: `formData.diaryId` → `formData.contentId`
- 리다이렉트: `/view/DIARY_LIST` → `/view/CONTENT_LIST`
- 메시지: "일기" → "콘텐츠"

---

### 3. useUserActions.tsx

**파일 경로**: `components/DynamicEngine/hook/useUserActions.tsx`

#### 수정 내용
```typescript
// 변경 전
case "DELETE_ALL_DIARIES":
    try {
        if (!confirm('정말로 모든 일기를 삭제하시겠습니까?')) return;
        const res = await axios.delete('/api/diary/delete-all');
        if (res.status === 200) {
            alert('모든 일기 삭제 완료');
            router.push('/view/DIARY_LIST');
        }
    } catch (error) {
        alert('일기 삭제 실패');
    }
    break;

// 변경 후
case "DELETE_ALL_CONTENTS":
    try {
        if (!confirm('정말로 모든 콘텐츠를 삭제하시겠습니까?')) return;
        const res = await axios.delete('/api/content/delete-all');
        if (res.status === 200) {
            alert('모든 콘텐츠 삭제 완료');
            router.push('/view/CONTENT_LIST');
        }
    } catch (error) {
        alert('콘텐츠 삭제 실패');
    }
    break;
```

#### 변경 항목
- 액션 타입: `DELETE_ALL_DIARIES` → `DELETE_ALL_CONTENTS`
- API URL: `/api/diary/delete-all` → `/api/content/delete-all`
- 리다이렉트: `/view/DIARY_LIST` → `/view/CONTENT_LIST`
- 메시지: "일기" → "콘텐츠"

---

### 4. Header.tsx

**파일 경로**: `components/layout/Header.tsx`

#### 수정 내용
```typescript
// 변경 전
<Link href="/view/DIARY_LIST">내 일기</Link>
<Link href="/view/DIARY_WRITE">일기 작성</Link>

// 변경 후
<Link href="/view/CONTENT_LIST">내 콘텐츠</Link>
<Link href="/view/CONTENT_WRITE">콘텐츠 작성</Link>
```

#### 변경 항목
- URL: `/view/DIARY_LIST` → `/view/CONTENT_LIST`
- URL: `/view/DIARY_WRITE` → `/view/CONTENT_WRITE`
- 라벨: "일기" → "콘텐츠"

---

### 5. page.tsx (보호된 화면 설정)

**파일 경로**: `app/view/[...slug]/page.tsx`

#### 수정 내용
```typescript
// 변경 전
const PROTECTED_SCREENS = [
  'DIARY_LIST',
  'DIARY_WRITE',
  'DIARY_DETAIL',
  // ...
];

// 변경 후
const PROTECTED_SCREENS = [
  'CONTENT_LIST',
  'CONTENT_WRITE',
  'CONTENT_DETAIL',
  // ...
];
```

#### 변경 항목
- `DIARY_LIST` → `CONTENT_LIST`
- `DIARY_WRITE` → `CONTENT_WRITE`
- `DIARY_DETAIL` → `CONTENT_DETAIL`

---

### 6. usePageMetadata.tsx

**파일 경로**: `components/DynamicEngine/hook/usePageMetadata.tsx`

#### 수정 내용
```typescript
// 변경 전 (코드 내 주석이나 로그에 diary 언급이 있다면)
console.log('Fetching diary metadata:', screenId);

// 변경 후
console.log('Fetching content metadata:', screenId);
```

#### 변경 항목
- 주석/로그 메시지: "diary" → "content"

---

### 7. ButtonField.tsx

**파일 경로**: `components/fields/ButtonField.tsx`

#### 수정 내용 (diary 관련 하드코딩이 있다면)
```typescript
// 변경 전
if (actionType === 'DIARY_WRITE_SUBMIT') {
  // ...
}

// 변경 후
if (actionType === 'CONTENT_WRITE_SUBMIT') {
  // ...
}
```

---

### 8. rendering_optimization.test.tsx

**파일 경로**: `tests/rendering_optimization.test.tsx`

#### 수정 내용
```typescript
// 변경 전
test('DIARY_LIST 화면 렌더링', async () => {
  render(<CommonPage params={Promise.resolve({ slug: ['DIARY_LIST'] })} />);
  await waitFor(() => {
    expect(screen.getByText('내 일기')).toBeInTheDocument();
  });
});

// 변경 후
test('CONTENT_LIST 화면 렌더링', async () => {
  render(<CommonPage params={Promise.resolve({ slug: ['CONTENT_LIST'] })} />);
  await waitFor(() => {
    expect(screen.getByText('내 콘텐츠')).toBeInTheDocument();
  });
});
```

#### 변경 항목
- 테스트명: "DIARY_LIST" → "CONTENT_LIST"
- 화면 ID: `['DIARY_LIST']` → `['CONTENT_LIST']`
- 기대값: "내 일기" → "내 콘텐츠"

---

## 📊 작업 순서 (중요!)

### 1단계: 화면 ID 매핑 수정 (15분)
```bash
# screenMap.ts 수정
# DIARY_* → CONTENT_*
```

### 2단계: 액션 핸들러 수정 (30분)
```bash
# useBusinessActions.tsx 수정 (CRUD 핸들러)
# useUserActions.tsx 수정 (DELETE_ALL 핸들러)
```

### 3단계: UI 컴포넌트 수정 (15분)
```bash
# Header.tsx 수정 (메뉴 링크)
# page.tsx 수정 (PROTECTED_SCREENS)
```

### 4단계: 테스트 파일 수정 (15분)
```bash
# rendering_optimization.test.tsx 수정
```

### 5단계: 린트 및 컴파일 확인 (10분)
```bash
npm run lint
npm run build
```

### 6단계: 테스트 실행 (10분)
```bash
npm run test
```

---

## ⚠️ 주의사항

### 1. 필드명 변경
- `formData.content` → `formData.contentText` (백엔드 DTO와 일치)
- `formData.diaryId` → `formData.contentId`

### 2. API 엔드포인트
- 모든 `/api/diary/*` → `/api/content/*` 일관성 유지

### 3. 사용자 메시지
- "일기" → "콘텐츠"로 일관되게 변경
- alert(), confirm() 메시지 모두 업데이트

### 4. URL 리다이렉트
- `router.push('/view/DIARY_*')` → `router.push('/view/CONTENT_*')`

### 5. 테스트 데이터
- MSW 핸들러에서 `/api/diary/*` → `/api/content/*` 변경
- 응답 데이터 구조 변경 (`diaryId` → `contentId`, `content` → `contentText`)

---

## 📌 예상 소요 시간

| 작업 | 예상 시간 |
|-----|----------|
| screenMap.ts 수정 | 15분 |
| useBusinessActions.tsx 수정 | 20분 |
| useUserActions.tsx 수정 | 10분 |
| Header.tsx 수정 | 5분 |
| page.tsx 수정 | 5분 |
| 테스트 파일 수정 | 15분 |
| 린트/컴파일 확인 | 10분 |
| 테스트 실행 | 10분 |
| **총 예상 시간** | **1시간 30분** |

---

## ✅ 체크리스트

### 화면 ID 및 라우팅
- [ ] screenMap.ts 업데이트 (DIARY → CONTENT)
- [ ] page.tsx PROTECTED_SCREENS 업데이트
- [ ] Header.tsx 메뉴 링크 업데이트

### 액션 핸들러
- [ ] useBusinessActions.tsx 수정 (CRUD)
- [ ] useUserActions.tsx 수정 (DELETE_ALL)
- [ ] ButtonField.tsx 수정 (필요 시)

### 데이터 구조
- [ ] formData 필드명 변경 (content → contentText)
- [ ] formData 필드명 변경 (diaryId → contentId)

### 테스트
- [ ] rendering_optimization.test.tsx 수정
- [ ] MSW 핸들러 업데이트 (필요 시)

### 검증
- [ ] 린트 성공 확인
- [ ] 컴파일 성공 확인
- [ ] 테스트 통과 확인

---

**다음 단계**: 백엔드 마이그레이션 완료 후 프론트엔드 코드 수정 시작
