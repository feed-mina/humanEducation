# Designer — Plan

> 이 파일은 UI/UX 설계 계획을 기록한다.
> 사용자의 명시적 승인("YES") 후에만 frontend_engineer에게 구현을 위임한다.

---

## Plan 작성 템플릿

```markdown
## [화면/컴포넌트 이름] 디자인 계획 — {날짜}

### 와이어프레임 (group 트리 구조)

```
engine-container (mobile | pc)
└── root-group (flex-col-layout)
    ├── header-group (flex-row-layout)
    │   ├── TEXT(title) [css: page-title]
    │   └── BUTTON(back) [css: btn-ghost]
    └── form-group (flex-col-layout)
        ├── INPUT(name) [css: input-base, label: "이름"]
        ├── SELECT(category) [css: select-base, label: "카테고리"]
        └── BUTTON(submit) [css: btn-primary, label: "저장"]
```

### css_class 정의 계획

| 클래스명 | 용도 | 신규/재사용 | 적용 대상 component_type |
|---------|------|-----------|------------------------|
| `page-title` | 페이지 제목 텍스트 스타일 | 신규 | TEXT |
| `btn-primary` | 주요 액션 버튼 (파란색 계열) | 재사용 | BUTTON |
| `btn-ghost` | 보조 액션 버튼 (투명 배경) | 신규 | BUTTON |
| `input-base` | 기본 입력 필드 스타일 | 재사용 | INPUT |
| `select-base` | 드롭다운 스타일 | 신규 | SELECT |

### 반응형 스펙

#### 모바일 (`engine-container mobile`)
- 레이아웃: 단일 컬럼, 풀 너비
- 버튼: 풀 너비 (`w-full`)
- 폰트 크기: base (16px)

#### PC (`engine-container pc`)
- 레이아웃: 카드 (max-w-md, 중앙 정렬)
- 버튼: 자동 너비
- 폰트 크기: base (16px)

### 컴포넌트 상태 정의

| 컴포넌트 | 기본 | 호버 | 포커스 | 비활성 | 에러 |
|----------|------|------|--------|--------|------|
| INPUT | border-gray | border-blue | ring-blue | opacity-50 | border-red |
| BUTTON(primary) | bg-blue-500 | bg-blue-600 | ring-blue | bg-gray-300 | - |

### DB 메타데이터 css_class 값 매핑

```sql
-- ui_metadata 레코드 예시 (삽입 시 참고)
INSERT INTO ui_metadata (screen_id, component_id, component_type, css_class, ...)
VALUES
  ('NEW_SCREEN', 'title-text', 'TEXT', 'page-title', ...),
  ('NEW_SCREEN', 'submit-btn', 'BUTTON', 'btn-primary', ...);
```

### 신규 CSS 파일 위치
- 전역 스타일: `metadata-project/app/globals.css`
- 컴포넌트 스타일: `metadata-project/components/fields/{Component}.module.css`

### 접근성 체크리스트
- [ ] 버튼: `aria-label` 또는 가시적 텍스트 레이블
- [ ] 입력: `id`와 `<label>` 연결 또는 `aria-label`
- [ ] 이미지: `alt` 텍스트
- [ ] 포커스 순서: 탭 키 이동 논리적 순서
- [ ] 색상 대비: WCAG AA 기준 (4.5:1)

### 담당자
- Frontend Engineer: CSS 클래스 구현 (`globals.css` 또는 CSS module)

### 승인 상태
[ ] 사용자 승인 대기 중
[x] 사용자 승인 완료 (날짜: ...)
[ ] 구현 완료
```

---

## 현재 계획 없음

아직 작성된 디자인 계획이 없습니다.