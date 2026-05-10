# Architect — Mobile+Web Integration Plan (단계적 전환 방식)

**접근 방식**: Hybrid (Next.js 웹 유지 → Expo 모바일 추가 → 데이터 기반 통합 결정)
**작성일**: 2026-03-01
**담당**: Architect Agent

---

## Research Analysis (연구 분석)

### 현재 아키텍처 분석
- **웹**: Next.js 16.1.3 (React 19) + TypeScript
- **백엔드**: Spring Boot 3.1.4 (Java 17) + PostgreSQL 15 + Redis
- **SDUI 코어**: `ui_metadata` 테이블 기반 메타데이터 렌더링
- **트리 빌딩**: UiService.java O(n) LinkedHashMap 알고리즘
- **캐싱**: Redis TTL 1시간, 키: `{rolePrefix}_{screenId}`

### 단계적 전환 방식 분석

#### 장점
✅ **리스크 최소화**: 점진적 마이그레이션 (3-6개월)
✅ **검증된 전환**: SEO 트래픽 데이터 기반 의사결정
✅ **백워드 호환**: 기존 웹 서비스 무중단
✅ **단일 코드베이스 목표**: Expo로 통합 시 유지보수 비용 절감
✅ **OTA 업데이트**: Expo Updates로 앱 재배포 없이 UI 변경

#### 단점
❌ **긴 마이그레이션 기간**: 6개월 소요
❌ **중간 단계 오버헤드**: Phase 2에서 웹+앱 병행
❌ **의사결정 지연**: SEO 데이터 수집 후 최종 결정

### 3단계 전환 전략

```
Phase 1 (Month 1-2): Next.js 웹 플랫폼 파라미터 지원
┌──────────────────────────────────────────┐
│  Next.js Web (기존 유지)                  │
│  - useDeviceType (768px)                 │
│  - platform 파라미터 추가                │
│  - 백엔드 API 플랫폼 지원                │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│  Spring Boot API (Platform-Aware)        │
│  GET /api/ui/{screenId}?platform=web|mobile
└──────────────────────────────────────────┘

Phase 2 (Month 3-5): Expo 모바일 앱 개발
┌─────────────────────┐  ┌─────────────────┐
│  Next.js Web (유지)  │  │  Expo Mobile    │
│  - SEO 트래픽 측정  │  │  - iOS/Android  │
└─────────────────────┘  └─────────────────┘
              ↓                    ↓
┌──────────────────────────────────────────┐
│  Spring Boot API (Platform-Aware)        │
└──────────────────────────────────────────┘

Phase 3 (Month 6): 데이터 기반 의사결정
┌─────────────────────────────────────────┐
│  SEO 트래픽 분석                         │
│  - 높음: Next.js 웹 유지 + Expo 병행    │
│  - 낮음: Expo Web으로 웹 통합 (단일화)  │
└─────────────────────────────────────────┘
```

### Expo 선택 이유

1. **expo-router**: Next.js와 유사한 파일 기반 라우팅
2. **Expo Web**: react-native-web 내장 (웹+모바일 단일 코드)
3. **OTA Updates**: 앱 재배포 없이 JS 번들 업데이트 (SDUI 철학 일치)
4. **개발 환경 간소화**: Xcode/Android Studio 불필요
5. **네이티브 모듈**: 카메라, 위치, 푸시 알림 기본 제공

### 코드 재사용률

| Phase | 웹 | 모바일 | 재사용률 |
|-------|-----|--------|----------|
| Phase 1 | Next.js 100% | - | - |
| Phase 2 | Next.js 100% | Expo 85% | 85% (앱만) |
| Phase 3 (통합) | Expo Web 90% | Expo 100% | **90-95%** |

---

## Implementation Plan (구현 계획)

### Phase 1: Backend + Next.js 웹 플랫폼 지원 (Month 1-2)

#### 목표
- 백엔드 API platform 파라미터 지원
- Next.js useDeviceType 768px 표준화
- 모바일 뷰포트에서 platform=mobile 전송
- JSONB component_props 병합 로직

#### 1.1 Backend API 확장

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/controller/UiController.java`

```java
@GetMapping("/{screenId}")
public ApiResponse<List<UiResponseDto>> getUiMetadataList(
    @PathVariable String screenId,
    @RequestParam(required = false, defaultValue = "web") String platform,
    @RequestHeader(value = "X-Platform", required = false) String headerPlatform,
    @AuthenticationPrincipal UserDetails userDetails
) {
    // 플랫폼 화이트리스트
    if (!List.of("web", "mobile").contains(platform)) {
        platform = "web";
    }

    String userRole = extractRole(userDetails);
    List<UiResponseDto> tree = uiService.getUiTree(screenId, userRole, platform);
    return ApiResponse.success(tree);
}
```

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/service/UiService.java`

```java
@Cacheable(
    value = "uiMetadataCache",
    key = "#userRole + '_' + #screenId + '_' + #platform"
)
public List<UiResponseDto> getUiTree(String screenId, String userRole, String platform) {
    List<UiMetadata> entities = uiMetadataRepository
        .findByScreenIdOrderByOrderIndexAsc(screenId);

    return buildTree(entities, userRole, platform);
}

private String mergePlatformProps(String propsJson, String platform) {
    if (propsJson == null) return "{}";

    try {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> allProps = mapper.readValue(propsJson, Map.class);

        Map<String, Object> common =
            (Map<String, Object>) allProps.getOrDefault("common", new HashMap<>());
        Map<String, Object> platformSpecific =
            (Map<String, Object>) allProps.getOrDefault(platform, new HashMap<>());

        Map<String, Object> merged = new HashMap<>(common);
        merged.putAll(platformSpecific);

        return mapper.writeValueAsString(merged);
    } catch (JsonProcessingException e) {
        log.error("Invalid JSONB: {}", propsJson);
        return "{}";
    }
}
```

#### 1.2 Next.js 웹 통합

**파일**: `metadata-project/hooks/useDeviceType.tsx`

```typescript
export type Platform = 'mobile' | 'web';

export const useDeviceType = () => {
  const [platform, setPlatform] = useState<Platform>('web');

  useEffect(() => {
    const checkDevice = () => {
      setPlatform(window.innerWidth < 768 ? 'mobile' : 'web');
    };

    checkDevice();
    window.addEventListener('resize', checkDevice);
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  return { platform, isMobile: platform === 'mobile' };
};
```

**파일**: `metadata-project/services/axios.tsx`

```typescript
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const platform = window.innerWidth < 768 ? 'mobile' : 'web';
    config.headers['X-Platform'] = platform;
    config.params = { ...config.params, platform };
  }
  return config;
});
```

**파일**: `metadata-project/components/providers/MetadataProvider.tsx`

```typescript
export const MetadataProvider = ({ screenId, children }: Props) => {
  const { platform } = useDeviceType();

  const { data } = useQuery(
    [`${rolePrefix}_${screenId}_${platform}`],
    () => api.get(`/api/ui/${screenId}`, { params: { platform } })
  );

  return <MetadataContext.Provider value={data}>{children}</MetadataContext.Provider>;
};
```

#### 1.3 Database component_props 설정

```sql
-- 예시: LOGIN_PAGE 버튼
UPDATE ui_metadata SET component_props = '{
  "common": {
    "type": "submit"
  },
  "mobile": {
    "minHeight": 48,
    "fontSize": 16,
    "hapticFeedback": true
  },
  "web": {
    "minHeight": 40,
    "fontSize": 14
  }
}'::jsonb WHERE component_id = 'loginSubmitBtn';

-- 예시: DIARY_LIST 레이아웃
UPDATE ui_metadata SET component_props = '{
  "mobile": {
    "viewType": "list",
    "itemHeight": 100
  },
  "web": {
    "viewType": "grid",
    "columns": 3
  }
}'::jsonb WHERE component_id = 'diaryListContainer';
```

#### 1.4 Phase 1 완료 기준
- ✅ Backend API platform 파라미터 지원
- ✅ useDeviceType 768px 표준화
- ✅ 모바일 뷰포트에서 올바른 메타데이터 수신
- ✅ Redis 캐시 키 플랫폼별 분리
- ✅ 단위/통합 테스트 통과

---

### Phase 2: Expo 모바일 앱 개발 (Month 3-5)

#### 목표
- Expo 프로젝트 생성 (expo-router)
- DynamicEngine 포팅 (React → React Native)
- componentMapNative 구현
- SecureStore JWT 저장
- iOS/Android 빌드

#### 2.1 Expo 프로젝트 초기화

```bash
# Expo 프로젝트 생성
npx create-expo-app SDUI-expo-app --template tabs

cd SDUI-expo-app

# 필수 패키지 설치
npx expo install expo-router expo-secure-store
npx expo install axios @tanstack/react-query
npx expo install react-native-web react-dom  # Expo Web 지원
npx expo install @react-navigation/native
```

**디렉토리 구조**:
```
SDUI-expo-app/
├── app/                     # expo-router (파일 기반 라우팅)
│   ├── view/
│   │   └── [...slug].tsx   # 동적 라우팅 (Next.js와 유사)
│   ├── _layout.tsx         # 루트 레이아웃
│   └── index.tsx           # 홈 화면
├── components/
│   ├── DynamicEngine/
│   │   ├── DynamicEngine.tsx
│   │   └── useDynamicEngine.tsx
│   ├── constants/
│   │   └── componentMapNative.ts
│   └── fields/
│       ├── Input.native.tsx
│       ├── Modal.native.tsx
│       └── DatePicker.native.tsx
├── hooks/
│   └── usePlatform.ts
├── services/
│   └── api.ts              # SecureStore + Bearer token
├── app.json
└── package.json
```

#### 2.2 DynamicEngine 포팅

**파일**: `SDUI-expo-app/components/DynamicEngine/DynamicEngine.tsx`

```typescript
import { View, StyleSheet } from 'react-native';
import { Metadata } from '../../types/metadata';
import { componentMapNative } from '../constants/componentMapNative';

export const DynamicEngine = ({ metadata, screenId }: Props) => {
  const renderNode = (node: Metadata) => {
    const Component = componentMapNative[node.componentType];

    if (!Component) {
      console.warn(`Unknown component: ${node.componentType}`);
      return null;
    }

    return (
      <View key={node.componentId} style={resolveStyle(node.cssClass)}>
        <Component meta={node} />
      </View>
    );
  };

  return (
    <View style={styles.engineContainer}>
      {metadata.map(renderNode)}
    </View>
  );
};

const styles = StyleSheet.create({
  engineContainer: {
    flex: 1,
    padding: 16,
  },
});
```

#### 2.3 componentMapNative

**파일**: `SDUI-expo-app/components/constants/componentMapNative.ts`

```typescript
import InputNative from '../fields/Input.native';
import ButtonNative from '../fields/Button.native';
import ModalNative from '../fields/Modal.native';
import DatePickerNative from '../fields/DatePicker.native';

export const componentMapNative: Record<string, React.ComponentType<any>> = {
  INPUT: InputNative,
  BUTTON: ButtonNative,
  MODAL: ModalNative,
  DATETIME_PICKER: DatePickerNative,
  // ... 19개 컴포넌트
};
```

**예시: Modal.native.tsx**

```typescript
import { Modal, View, Text, Pressable, StyleSheet } from 'react-native';

export default function ModalNative({ meta, onConfirm, onClose }: Props) {
  return (
    <Modal visible={true} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.content}>
          <Text style={styles.title}>{meta.labelText}</Text>
          <Pressable onPress={onConfirm} style={styles.button}>
            <Text style={styles.buttonText}>확인</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 24,
    width: '90%',
    maxWidth: 400,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    textAlign: 'center',
    fontSize: 16,
  },
});
```

#### 2.4 SecureStore JWT 저장

**파일**: `SDUI-expo-app/services/api.ts`

```typescript
import axios from 'axios';
import * as SecureStore from 'expo-secure-store';

const api = axios.create({
  baseURL: 'https://api.yourapp.com',
});

api.interceptors.request.use(async (config) => {
  const token = await SecureStore.getItemAsync('accessToken');

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  config.headers['X-Platform'] = 'mobile';
  config.params = { ...config.params, platform: 'mobile' };

  return config;
});

export const saveTokens = async (accessToken: string, refreshToken: string) => {
  await SecureStore.setItemAsync('accessToken', accessToken);
  await SecureStore.setItemAsync('refreshToken', refreshToken);
};

export const clearTokens = async () => {
  await SecureStore.deleteItemAsync('accessToken');
  await SecureStore.deleteItemAsync('refreshToken');
};

export default api;
```

#### 2.5 expo-router 라우팅

**파일**: `SDUI-expo-app/app/view/[...slug].tsx`

```typescript
import { useLocalSearchParams } from 'expo-router';
import { MetadataProvider } from '../../components/providers/MetadataProvider';
import { DynamicEngine } from '../../components/DynamicEngine/DynamicEngine';

export default function CommonPage() {
  const { slug } = useLocalSearchParams<{ slug: string[] }>();
  const screenId = Array.isArray(slug) ? slug[0] : slug;

  return (
    <MetadataProvider screenId={screenId}>
      <DynamicEngine />
    </MetadataProvider>
  );
}
```

#### 2.6 Phase 2 완료 기준
- ✅ Expo 앱 iOS/Android 빌드 성공
- ✅ DynamicEngine 정상 렌더링
- ✅ 19개 컴포넌트 중 15개 호환, 4개 네이티브 재작성
- ✅ SecureStore JWT 저장 확인
- ✅ 로그인 → 리스트 → 상세 플로우 테스트

---

### Phase 3: 데이터 기반 통합 결정 (Month 6)

#### 목표
- Next.js 웹 SEO 트래픽 분석
- Expo 모바일 앱 사용자 피드백 수집
- 최종 아키텍처 결정

#### 3.1 SEO 트래픽 분석

**측정 도구**: Google Analytics 4

```javascript
// 측정 항목
- 유기적 검색 트래픽 (Organic Search)
- 직접 방문 vs 검색 유입 비율
- 핵심 화면 (LOGIN_PAGE, DIARY_LIST) SEO 전환율
```

**의사결정 기준**:

| SEO 트래픽 비율 | 결정 |
|----------------|------|
| > 30% | **Next.js 웹 유지** + Expo 모바일 병행 (2개 코드베이스) |
| 10-30% | 마케팅 랜딩만 Next.js, 나머지 Expo Web 통합 |
| < 10% | **Expo Web 전체 통합** (단일 코드베이스) |

#### 3.2 Expo Web 통합 (SEO 트래픽 낮을 경우)

**마이그레이션 작업**:
```bash
# Next.js 페이지를 Expo Web으로 포팅
metadata-project/app/view/[...slug]/page.tsx
→ SDUI-expo-app/app/view/[...slug].tsx (이미 완료)

# 웹 전용 컴포넌트 추가
components/fields/Modal.tsx
→ components/fields/Modal.web.tsx (react-modal)
→ Platform.select({ web: ModalWeb, native: ModalNative })
```

**Expo Web 빌드**:
```bash
# 웹 빌드
npx expo export:web

# 정적 파일 생성 (dist/)
# Netlify/Vercel에 배포
```

#### 3.3 Next.js 병행 유지 (SEO 트래픽 높을 경우)

**공통 로직 추출**:
```bash
# Yarn Workspaces 구성
SDUI/
├── packages/
│   ├── web/              # Next.js
│   ├── mobile/           # Expo
│   └── shared/           # 공통 타입/액션
```

**shared 패키지**:
```typescript
// shared/types/metadata.ts
export interface Metadata { ... }

// shared/actions/useUserActions.ts
export const useUserActions = (platform: 'web' | 'mobile') => {
  // 플랫폼별 JWT 저장 분기
};
```

#### 3.4 Phase 3 완료 기준
- ✅ SEO 데이터 수집 (3개월 기간)
- ✅ 최종 아키텍처 결정
- ✅ 통합 작업 완료 (Expo Web) 또는 병행 구조 확정
- ✅ 프로덕션 배포 완료

---

## Security Considerations (보안 고려사항)

### 1. JWT 저장 방식

| Platform | Storage | 보안 수준 |
|----------|---------|----------|
| Next.js Web | HttpOnly Cookie | ✅ 높음 (XSS 방어) |
| Expo Mobile | SecureStore (Keychain/Keystore) | ✅ 높음 (루팅 방어) |
| Expo Web | ❌ localStorage (불가) → Cookie | ⚠️ 웹과 동일 |

### 2. Certificate Pinning (Expo 앱)

```typescript
// SDUI-expo-app/app/_layout.tsx
import * as Network from 'expo-network';

useEffect(() => {
  // SSL Pinning (Native Modules 필요)
  // expo-dev-client로 Custom Development Build 생성
}, []);
```

### 3. Rate Limiting (Backend)

```java
@Component
public class RateLimitFilter extends OncePerRequestFilter {
    @Override
    protected void doFilterInternal(HttpServletRequest request, ...) {
        String key = "rate_limit:" + request.getRemoteAddr();
        Long count = redisTemplate.opsForValue().increment(key);

        if (count == 1) {
            redisTemplate.expire(key, 1, TimeUnit.MINUTES);
        }

        if (count > 100) {
            response.setStatus(429);
            return;
        }

        filterChain.doFilter(request, response);
    }
}
```

### 4. JSONB SQL Injection 방어

```java
// UiService.java
private String sanitizeComponentProps(String props) {
    try {
        objectMapper.readTree(props); // JSON 유효성 검증
        return props;
    } catch (JsonProcessingException e) {
        log.error("Invalid JSONB: {}", props);
        return "{}";
    }
}
```

---

## Test Plan (테스트 계획)

### Phase 1 테스트 (Next.js 웹)

```typescript
// tests/platform_detection.test.tsx
test('sends mobile platform when width < 768px', async () => {
  global.innerWidth = 375;
  render(<MetadataProvider screenId="LOGIN_PAGE" />);

  await waitFor(() => {
    expect(mockAxios.get).toHaveBeenCalledWith(
      expect.stringContaining('?platform=mobile')
    );
  });
});
```

### Phase 2 테스트 (Expo 앱)

```typescript
// SDUI-expo-app/__tests__/DynamicEngine.test.tsx
import { render } from '@testing-library/react-native';

test('renders native components', () => {
  const metadata = [{ componentType: 'INPUT', componentId: 'email' }];
  const { getByTestId } = render(<DynamicEngine metadata={metadata} />);

  expect(getByTestId('input-email')).toBeTruthy();
});
```

### E2E 테스트 (Detox)

```typescript
// e2e/login.e2e.ts
describe('Login Flow', () => {
  it('should login and store token in SecureStore', async () => {
    await element(by.id('emailInput')).typeText('test@example.com');
    await element(by.id('passwordInput')).typeText('password123');
    await element(by.id('loginBtn')).tap();

    await waitFor(element(by.id('diaryListScreen')))
      .toBeVisible()
      .withTimeout(5000);
  });
});
```

---

## Dependencies (의존성)

### Depends on
- ✅ 없음 (첫 번째 작업)

### Blocks
- Backend Engineer: API 구현
- Frontend Engineer: 웹/앱 통합
- Planner: component_props 설계
- Designer: 반응형 CSS
- QA Engineer: 테스트 시작

---

## Timeline (타임라인)

```
Month 1-2: Phase 1 (Backend + Next.js 웹)
  Week 1-2: Backend API 플랫폼 지원
  Week 3-4: Next.js useDeviceType 통합

Month 3-5: Phase 2 (Expo 모바일 앱)
  Week 5-6:  Expo 프로젝트 초기화
  Week 7-9:  DynamicEngine 포팅
  Week 10-12: componentMapNative 구현

Month 6: Phase 3 (데이터 기반 결정)
  Week 13-14: SEO 트래픽 분석
  Week 15-16: 통합 또는 병행 구조 확정
```

---

## Rollback Plan (롤백 계획)

### Phase 1 롤백
```java
// UiController.java
platform = "web"; // 강제로 web 고정
```

### Phase 2 롤백
- Expo 앱 TestFlight/Play Store 비공개 처리
- 웹 서비스만 유지

### Phase 3 롤백
- Expo Web 통합 실패 시 Next.js 웹 복구

---

## Success Metrics (성공 지표)

| Phase | 지표 | 목표 |
|-------|------|------|
| Phase 1 | API 응답 시간 | < 200ms |
| Phase 1 | Redis 캐시 히트율 | > 70% |
| Phase 2 | 앱 크래시율 | < 1% (Firebase Crashlytics) |
| Phase 2 | 코드 재사용률 | > 85% |
| Phase 3 | SEO 트래픽 비율 | 측정 후 결정 |
| Phase 3 | 단일 코드베이스 전환 | 90-95% 재사용 |

---

**다음 단계**: Backend Engineer plan.md 작성 및 Phase 1 구현 시작
