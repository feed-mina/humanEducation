# 최종 수정 완료 (2026-03-02)

## ✅ 해결된 문제

### 1. React DOM prop 경고 (setFormData)
**문제**:
```
React does not recognize the `setFormData` prop on a DOM element
>> ImageField, ButtonField, TextField
```

**원인**: DynamicEngine이 `{...rest}`로 모든 prop을 전달

**해결**:
```typescript
// DynamicEngine.tsx
// setFormData를 rest에서 분리하고, ADDRESS_SEARCH_GROUP만 전달
const {metadata, screenId, pageData, formData, setFormData, onChange, onAction, ...} = props;

const componentProps: any = {
    key: uId,
    id: uId,
    meta: node,
    data: finalData,
    onChange,
    onAction,
    ...rest
};

// ADDRESS_SEARCH_GROUP만 formData, setFormData 추가
if (typeKey === "ADDRESS_SEARCH_GROUP" && setFormData) {
    componentProps.formData = formData;
    componentProps.setFormData = setFormData;
}

return <Component {...componentProps} />;
```

### 2. AddressSearchGroup zip_code undefined
**문제**:
```
Cannot read properties of undefined (reading 'zip_code')
```

**원인**: formData가 undefined이거나 초기값 없음

**해결**:
```typescript
// AddressSearchGroup.tsx
const AddressSearchGroup = ({ meta, formData = {}, setFormData, onAction }: any) => {
    // 방어 코드 추가
    const handleComplete = (data: any) => {
        if (!setFormData) {
            console.warn('[AddressSearchGroup] setFormData is not provided');
            return;
        }
        // ...
    };

    // 옵셔널 체이닝 사용
    <input value={formData?.zip_code || ''} />
    <input value={formData?.road_address || ''} />
    <input value={formData?.detail_address || ''} />
}
```

### 3. CSP 정책 (http/https 혼재)
**해결**:
```typescript
// next.config.ts
"script-src 'self' 'unsafe-inline' 'unsafe-eval' http://t1.daumcdn.net https://t1.daumcdn.net"
```

### 4. CONTENT_LIST 키 불일치
**해결**:
```sql
-- fix_content_list_key_mismatch.sql
UPDATE ui_metadata
SET ref_data_id = 'content_list_source'
WHERE screen_id = 'CONTENT_LIST'
  AND component_id = 'CONTENT_CARD';
```

---

## 🧪 테스트 체크리스트

### ✅ 테스트 1: React 경고 제거
- [ ] 브라우저 콘솔에 "React does not recognize the `setFormData` prop" 경고 없음
- [ ] ImageField, ButtonField, TextField 정상 렌더링

### ✅ 테스트 2: 주소 검색 (CSP + formData)
- [ ] `/view/REGISTER_PAGE` 또는 `/view/ADDITIONAL_INFO_PAGE` 접속
- [ ] "주소 찾기" 버튼 클릭 → Daum 팝업 열림 (CSP 에러 없음)
- [ ] 주소 선택 → 우편번호, 도로명 주소 자동 입력
- [ ] 상세 주소 입력 가능
- [ ] 브라우저 콘솔에 zip_code undefined 에러 없음

### ✅ 테스트 3: 콘텐츠 리스트
- [ ] `/view/CONTENT_LIST` 접속
- [ ] 3개의 콘텐츠 카드 표시
- [ ] 각 카드에 제목, 작성자, 날짜 표시
- [ ] 카드 클릭 시 상세 페이지 이동

---

## 🔧 실행 순서

### 1. 프론트엔드 재시작
```bash
cd metadata-project
# Ctrl+C로 기존 서버 중지
npm run dev
```

### 2. 브라우저 캐시 클리어
```
F12 → Network 탭 → "Disable cache" 체크
또는 Ctrl+Shift+R (하드 리프레시)
```

### 3. 테스트 페이지 접속
```
http://localhost:3000/view/REGISTER_PAGE  (주소 검색 테스트)
http://localhost:3000/view/CONTENT_LIST   (콘텐츠 리스트 테스트)
```

---

## 📊 예상 결과

### 콘솔 (정상)
```javascript
// React 경고 없음
// CSP 에러 없음
// zip_code undefined 에러 없음
```

### 네트워크 (정상)
```
GET /api/ui/CONTENT_LIST
→ CONTENT_CARD의 refDataId: "content_list_source" ✅

POST /api/execute/GET_CONTENT_LIST_PAGE
→ data: [{ content_id: 4, title: "콘텐츠 1", ... }] ✅
```

### 화면 (정상)
- ✅ 주소 검색 팝업 정상 작동
- ✅ 콘텐츠 카드 3개 표시
- ✅ 카드 클릭 시 상세 페이지 이동

---

## 🐛 여전히 문제 발생 시

### 1. "setFormData prop" 경고가 여전히 보임
```bash
# 프론트엔드 완전 재시작
cd metadata-project
rm -rf .next
npm run dev
```

### 2. "zip_code undefined" 에러가 여전히 발생
```javascript
// 브라우저 콘솔에서 확인
console.log(formData)  // undefined인지 확인
```
→ 페이지 새로고침 (F5)

### 3. 콘텐츠 리스트가 여전히 안보임
```bash
# Redis 캐시 재클리어
docker exec -it my-redis redis-cli FLUSHDB

# 브라우저 콘솔에서 확인
# "[DynamicEngine] content_list_source 데이터가 배열이 아닙니다" 경고 확인
```

→ SQL 실행 확인: `fix_content_list_key_mismatch.sql`

---

## 📁 수정된 파일 목록

1. `metadata-project/next.config.ts` - CSP 정책
2. `metadata-project/app/view/[...slug]/page.tsx` - setFormData 전달
3. `metadata-project/components/DynamicEngine/type.ts` - setFormData 타입
4. `metadata-project/components/DynamicEngine/DynamicEngine.tsx` - 선택적 prop 전달
5. `metadata-project/components/fields/AddressSearchGroup.tsx` - 방어 코드
6. `fix_content_list_key_mismatch.sql` - DB 수정 스크립트

---

## 💡 핵심 교훈

1. **React prop 전달**: DOM 요소에 불필요한 prop 전달 금지
2. **방어 코드**: formData, setFormData 같은 필수 prop은 기본값 + 옵셔널 체이닝
3. **키 일치**: pageData 키와 ref_data_id는 정확히 일치해야 함
4. **CSP 정책**: 외부 스크립트는 http/https 모두 허용 필요
