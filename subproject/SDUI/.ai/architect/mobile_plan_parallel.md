# Architect — Mobile+Web Integration Plan (병행 접근 방식)

**접근 방식**: Next.js 웹 + React Native 모바일 병행 운영 (2개 코드베이스)
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

### 병행 접근 방식 분석

#### 장점
✅ **리스크 최소화**: 기존 Next.js 웹 서비스 무중단
✅ **점진적 개발**: 웹 유지하면서 모바일 앱 개발
✅ **플랫폼별 최적화**: 각 플랫폼에 최적화된 UX (Next.js SSR, RN 네이티브)
✅ **SEO 지원**: Next.js SSR/SSG 유지
✅ **기술 스택 독립성**: 웹/앱 각각 최신 기술 적용 가능

#### 단점
❌ **유지보수 비용 2배**: 2개 코드베이스 관리
❌ **로직 중복**: DynamicEngine, 액션 핸들러 등 중복 구현
❌ **동기화 부담**: 웹/앱 기능 일치 유지 필요
❌ **개발 리소스**: 프론트엔드 개발자 2팀 필요

### 플랫폼 감지 전략

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                          │
├─────────────────────────────────────────────────────────┤
│  Next.js Web                  │  React Native Mobile    │
│  - useDeviceType (768px)      │  - Platform.OS          │
│  - X-Platform: "web"/"mobile" │  - X-Platform: "mobile" │
│  - HttpOnly Cookie (JWT)      │  - Bearer (SecureStore) │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            Backend API (Platform-Aware)                  │
├─────────────────────────────────────────────────────────┤
│  GET /api/ui/{screenId}?platform=web|mobile             │
│  - UiController: @RequestParam platform                 │
│  - UiService: 플랫폼 기반 필터링 + JSONB 병합            │
│  - Redis Cache: UI:{rolePrefix}_{screenId}_{platform}  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  Database Layer                          │
├─────────────────────────────────────────────────────────┤
│  ui_metadata:                                            │
│  - component_props JSONB: {"common":{}, "mobile":{}}    │
│  - allowed_roles: RBAC                                  │
└─────────────────────────────────────────────────────────┘
```

### 데이터 플로우

#### Web (Next.js)
```
1. 브라우저 뷰포트 감지 (useDeviceType)
   - width < 768px → platform = "mobile"
   - width ≥ 768px → platform = "web"

2. MetadataProvider
   GET /api/ui/LOGIN_PAGE?platform=mobile
   Headers: { X-Platform: "mobile" }
   Cookies: { accessToken: "..." }

3. DynamicEngine 렌더링
   - componentMap (React 컴포넌트)
   - CSS className 적용
```

#### Mobile (React Native)
```
1. 플랫폼 감지 (Platform.OS)
   - 항상 platform = "mobile"

2. API 요청
   GET /api/ui/LOGIN_PAGE?platform=mobile
   Headers: {
     X-Platform: "mobile",
     Authorization: "Bearer {token}"
   }

3. DynamicEngineNative 렌더링
   - componentMapNative (React Native 컴포넌트)
   - StyleSheet 스타일 적용
```

### 코드베이스 구조

```
SDUI/
├── metadata-project/          # Next.js 웹 (기존)
│   ├── components/
│   │   ├── DynamicEngine/
│   │   ├── constants/componentMap.tsx
│   │   └── fields/           # 웹 전용 컴포넌트
│   ├── hooks/useDeviceType.tsx
│   └── services/axios.tsx
│
├── mobile-app/                # React Native 앱 (신규)
│   ├── src/
│   │   ├── components/
│   │   │   ├── DynamicEngine/  # 웹에서 포팅
│   │   │   ├── constants/componentMapNative.ts
│   │   │   └── fields/       # 네이티브 컴포넌트
│   │   ├── services/api.ts   # SecureStore + Bearer
│   │   └── navigation/
│   ├── ios/
│   └── android/
│
├── shared/ (선택적)           # 공통 로직 추출
│   ├── types/                # TypeScript 타입
│   ├── utils/                # 유틸 함수
│   └── actions/              # 액션 핸들러
│
└── SDUI-server/              # Spring Boot (공통 백엔드)
    └── domain/ui/            # 플랫폼 파라미터 지원
```

---

## Implementation Plan (구현 계획)

### Phase 1: Backend API 플랫폼 지원 (Week 1-2)

#### 1.1 UiController 확장
**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/controller/UiController.java`

```java
@GetMapping("/{screenId}")
public ApiResponse<List<UiResponseDto>> getUiMetadataList(
    @PathVariable String screenId,
    @RequestParam(required = false, defaultValue = "web") String platform,
    @RequestHeader(value = "X-Platform", required = false) String headerPlatform,
    @AuthenticationPrincipal UserDetails userDetails
) {
    // 플랫폼 화이트리스트 검증
    if (!List.of("web", "mobile").contains(platform)) {
        platform = "web";
    }

    // 헤더-파라미터 불일치 로깅 (보안 모니터링)
    if (headerPlatform != null && !headerPlatform.equals(platform)) {
        log.warn("Platform mismatch - param: {}, header: {}, user: {}",
                 platform, headerPlatform, userDetails.getUsername());
    }

    String userRole = extractRole(userDetails);
    List<UiResponseDto> tree = uiService.getUiTree(screenId, userRole, platform);
    return ApiResponse.success(tree);
}
```

#### 1.2 UiService 플랫폼 필터링
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

private List<UiResponseDto> buildTree(
    List<UiMetadata> entities,
    String userRole,
    String platform
) {
    Map<String, UiResponseDto> nodeMap = new LinkedHashMap<>();

    for (UiMetadata entity : entities) {
        // RBAC 필터링
        if (!isAccessible(entity, userRole)) {
            continue;
        }

        UiResponseDto dto = new UiResponseDto(entity);

        // JSONB component_props 병합
        dto.setComponentProps(mergePlatformProps(
            entity.getComponentProps(),
            platform
        ));

        nodeMap.put(entity.getComponentId(), dto);
    }

    // 트리 구조 빌딩 (기존 로직 유지)
    return buildHierarchy(nodeMap);
}
```

#### 1.3 JSONB Props 병합 로직
**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/dto/UiResponseDto.java`

```java
@SuppressWarnings("unchecked")
private String mergePlatformProps(String propsJson, String platform) {
    if (propsJson == null || propsJson.isBlank()) {
        return "{}";
    }

    try {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> allProps = mapper.readValue(propsJson, Map.class);

        // common props (모든 플랫폼 공통)
        Map<String, Object> commonProps =
            (Map<String, Object>) allProps.getOrDefault("common", new HashMap<>());

        // 플랫폼별 props (common 오버라이드)
        Map<String, Object> platformProps =
            (Map<String, Object>) allProps.getOrDefault(platform, new HashMap<>());

        Map<String, Object> merged = new HashMap<>(commonProps);
        merged.putAll(platformProps); // 플랫폼 설정이 우선

        return mapper.writeValueAsString(merged);
    } catch (JsonProcessingException e) {
        log.error("Failed to parse component_props: {}", propsJson, e);
        return "{}";
    }
}
```

### Phase 2: Next.js 웹 플랫폼 통합 (Week 3-4)

#### 2.1 useDeviceType 표준화 (768px)
**파일**: `metadata-project/hooks/useDeviceType.tsx`

```typescript
export type Platform = 'mobile' | 'web';

export const useDeviceType = () => {
  const [platform, setPlatform] = useState<Platform>('web');

  useEffect(() => {
    const checkDevice = () => {
      const width = window.innerWidth;
      setPlatform(width < 768 ? 'mobile' : 'web');
    };

    checkDevice();
    window.addEventListener('resize', checkDevice);
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  return {
    platform,
    isMobile: platform === 'mobile',
    isWeb: platform === 'web'
  };
};
```

#### 2.2 axios 인터셉터 (X-Platform 헤더)
**파일**: `metadata-project/services/axios.tsx`

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || '/api',
  withCredentials: true, // Cookie 전송
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const width = window.innerWidth;
    const platform = width < 768 ? 'mobile' : 'web';

    config.headers['X-Platform'] = platform;
    config.params = { ...config.params, platform };
  }
  return config;
});

export default api;
```

#### 2.3 MetadataProvider 통합
**파일**: `metadata-project/components/providers/MetadataProvider.tsx`

```typescript
export const MetadataProvider = ({ screenId, children }: Props) => {
  const { platform } = useDeviceType();

  const { data, isLoading, error } = useQuery(
    [`${rolePrefix}_${screenId}_${platform}`], // 플랫폼별 캐시 키
    () => api.get(`/api/ui/${screenId}`, { params: { platform } })
  );

  // ... 기존 로직
};
```

### Phase 3: React Native 앱 개발 (Week 5-10)

#### 3.1 프로젝트 초기화
```bash
# React Native CLI 방식
npx react-native init SDUIMobileApp --template react-native-template-typescript

cd SDUIMobileApp

# 필수 패키지 설치
npm install axios @react-native-async-storage/async-storage
npm install expo-secure-store  # JWT 보안 저장
npm install @react-navigation/native @react-navigation/stack
npm install @tanstack/react-query
```

#### 3.2 DynamicEngine 포팅
**파일**: `mobile-app/src/components/DynamicEngine/DynamicEngine.tsx`

```typescript
import { View, Text, StyleSheet } from 'react-native';
import { Metadata } from '../../types/metadata';
import { componentMapNative } from '../constants/componentMapNative';

interface Props {
  metadata: Metadata[];
  screenId: string;
  pageData?: Record<string, any>;
  formData?: Record<string, any>;
}

export const DynamicEngine = ({ metadata, screenId, pageData, formData }: Props) => {
  const renderNode = (node: Metadata) => {
    const Component = componentMapNative[node.componentType];

    if (!Component) {
      console.warn(`Unknown component type: ${node.componentType}`);
      return null;
    }

    return (
      <View key={node.componentId} style={resolveStyle(node.cssClass)}>
        <Component meta={node} pageData={pageData} formData={formData} />
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

#### 3.3 componentMapNative
**파일**: `mobile-app/src/components/constants/componentMapNative.ts`

```typescript
import InputNative from '../fields/InputNative';
import TextNative from '../fields/TextNative';
import ButtonNative from '../fields/ButtonNative';
import ImageNative from '../fields/ImageNative';
import ModalNative from '../fields/ModalNative';
import DatePickerNative from '../fields/DatePickerNative';

export const componentMapNative: Record<string, React.ComponentType<any>> = {
  INPUT: InputNative,
  TEXT: TextNative,
  BUTTON: ButtonNative,
  IMAGE: ImageNative,
  MODAL: ModalNative,
  DATETIME_PICKER: DatePickerNative,
  // ... 19개 컴포넌트
};
```

#### 3.4 SecureStore JWT 저장
**파일**: `mobile-app/src/services/api.ts`

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

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      const refreshToken = await SecureStore.getItemAsync('refreshToken');
      // Token refresh logic...
    }
    return Promise.reject(error);
  }
);

export default api;
```

### Phase 4: 공통 로직 추출 (Week 11-12)

#### 4.1 Shared 패키지 구성
```bash
# Yarn Workspaces 루트 설정
# SDUI/package.json
{
  "name": "sdui-monorepo",
  "private": true,
  "workspaces": [
    "metadata-project",
    "mobile-app",
    "shared"
  ]
}
```

#### 4.2 공통 타입 정의
**파일**: `shared/types/metadata.ts`

```typescript
export interface Metadata {
  componentId: string;
  componentType: string;
  labelText: string;
  placeholder?: string;
  componentProps: Record<string, any>;
  cssClass?: string;
  actionType?: string;
  groupDirection?: 'ROW' | 'COLUMN';
  children?: Metadata[];
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}
```

#### 4.3 공통 액션 핸들러
**파일**: `shared/actions/useUserActions.ts`

```typescript
export const useUserActions = (platform: 'web' | 'mobile') => {
  const handleLogin = async (email: string, password: string) => {
    const response = await api.post('/api/auth/login', { email, password });

    if (platform === 'web') {
      // Cookie 방식 (자동 저장)
    } else {
      // SecureStore 저장
      await SecureStore.setItemAsync('accessToken', response.data.accessToken);
    }
  };

  return { handleLogin };
};
```

---

## Security Considerations (보안 고려사항)

### 1. 플랫폼 위장 방어

**위협**: X-Platform 헤더 조작으로 모바일 메타데이터 불법 접근

**완화 방안**:
```java
// UiController.java
if (headerPlatform != null && !headerPlatform.equals(platform)) {
    log.warn("Platform mismatch detected - IP: {}, User: {}",
             request.getRemoteAddr(), userDetails.getUsername());
    // 임계값 초과 시 차단 (Rate Limiting)
}
```

### 2. JSONB SQL Injection

**위협**: component_props에 악의적 JSON 삽입

**완화 방안**:
```java
// JSON 파싱 검증
try {
    objectMapper.readTree(componentProps);
} catch (JsonProcessingException e) {
    log.error("Invalid JSONB detected: {}", componentProps);
    return "{}"; // 기본값 반환
}
```

### 3. JWT 저장 위치

| 플랫폼 | ❌ 취약한 방식 | ✅ 안전한 방식 |
|--------|---------------|---------------|
| Web | localStorage | HttpOnly Cookie |
| Mobile | AsyncStorage | SecureStore (Keychain/Keystore) |

### 4. Rate Limiting

**구현**: Bucket4j + Redis

```java
@Component
public class RateLimitFilter extends OncePerRequestFilter {
    @Override
    protected void doFilterInternal(HttpServletRequest request, ...) {
        String clientIp = request.getRemoteAddr();
        String key = "rate_limit:" + clientIp;

        Long requests = redisTemplate.opsForValue().increment(key);
        if (requests == 1) {
            redisTemplate.expire(key, 1, TimeUnit.MINUTES);
        }

        if (requests > 100) { // 분당 100 요청
            response.setStatus(429); // Too Many Requests
            return;
        }

        filterChain.doFilter(request, response);
    }
}
```

### 5. Certificate Pinning (모바일 앱)

```typescript
// mobile-app/src/config/security.ts
import { setCustomCA } from 'react-native-ssl-pinning';

export const enableCertificatePinning = () => {
  setCustomCA({
    certs: ['sha256/YOUR_CERT_HASH'],
    hostname: 'api.yourapp.com',
  });
};
```

---

## Test Plan (테스트 계획)

### 단위 테스트

#### Backend (JUnit 5)
```java
// UiServiceTest.java
@Test
void getUiTree_withMobilePlatform_returnsMobileProps() {
    List<UiResponseDto> result = uiService.getUiTree(
        "LOGIN_PAGE", "ROLE_USER", "mobile"
    );

    UiResponseDto button = result.stream()
        .filter(dto -> "submitBtn".equals(dto.getComponentId()))
        .findFirst().orElseThrow();

    Map<String, Object> props = objectMapper.readValue(
        button.getComponentProps(), Map.class
    );
    assertEquals(48, props.get("minHeight")); // 모바일 터치 타겟
}

@Test
void getUiTree_withInvalidPlatform_defaultsToWeb() {
    List<UiResponseDto> result = uiService.getUiTree(
        "LOGIN_PAGE", "ROLE_USER", "invalid"
    );
    assertNotNull(result);
}
```

#### Frontend - Web (Jest)
```typescript
// tests/platform_detection.test.tsx
test('sends mobile platform when viewport < 768px', async () => {
  global.innerWidth = 375;
  global.dispatchEvent(new Event('resize'));

  render(<MetadataProvider screenId="LOGIN_PAGE"><DynamicEngine /></MetadataProvider>);

  await waitFor(() => {
    expect(mockAxios.get).toHaveBeenCalledWith(
      expect.stringContaining('?platform=mobile'),
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-Platform': 'mobile' })
      })
    );
  });
});
```

#### Frontend - Mobile (Jest + RNTL)
```typescript
// mobile-app/__tests__/DynamicEngine.test.tsx
import { render } from '@testing-library/react-native';

test('renders native DatePicker', () => {
  const metadata = [
    { componentType: 'DATETIME_PICKER', componentId: 'date1' }
  ];

  const { getByTestId } = render(<DynamicEngine metadata={metadata} />);
  expect(getByTestId('native-date-picker')).toBeTruthy();
});
```

### 통합 테스트

#### API 플랫폼 파라미터 (Spring Boot Test)
```java
@Test
void getUiMetadata_withBearerToken_returnsSuccess() {
    String token = jwtUtil.createAccessToken(testUser);

    mockMvc.perform(get("/api/ui/LOGIN_PAGE")
            .param("platform", "mobile")
            .header("Authorization", "Bearer " + token)
            .header("X-Platform", "mobile"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.success").value(true));
}
```

### E2E 테스트

#### Web (Playwright)
```typescript
test('LOGIN_PAGE: mobile layout at 375px', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/view/LOGIN_PAGE');

  const submitBtn = page.locator('button[type="submit"]');
  const box = await submitBtn.boundingBox();
  expect(box?.height).toBeGreaterThanOrEqual(48); // 터치 타겟
});
```

#### Mobile (Detox)
```typescript
describe('Login Flow (Mobile App)', () => {
  it('should login with SecureStore token', async () => {
    await element(by.id('emailInput')).typeText('test@example.com');
    await element(by.id('passwordInput')).typeText('password123');
    await element(by.id('loginSubmitBtn')).tap();

    await waitFor(element(by.id('diaryListScreen')))
      .toBeVisible()
      .withTimeout(5000);
  });
});
```

### 보안 회귀 테스트

```java
@Test
void rateLimiting_exceedsLimit_returns429() {
    for (int i = 0; i < 101; i++) {
        mockMvc.perform(get("/api/ui/LOGIN_PAGE"));
    }

    mockMvc.perform(get("/api/ui/LOGIN_PAGE"))
        .andExpect(status().isTooManyRequests());
}
```

---

## Dependencies (의존성)

### Depends on (선행 작업)
- ✅ 없음 (첫 번째 작업)

### Blocks (후행 작업)
- Backend Engineer: API 구현
- Frontend Engineer: 웹/앱 통합
- Planner: component_props 설계
- Designer: CSS 클래스 정의
- QA Engineer: 테스트 시작

---

## Implementation Sequence (구현 순서)

```
Week 1-2:  Backend API 플랫폼 지원
Week 3-4:  Next.js 웹 통합
Week 5-10: React Native 앱 개발
Week 11-12: 공통 로직 추출 (Shared 패키지)
```

---

## Rollback Plan (롤백 계획)

1. **Backend API 롤백**: platform 파라미터 무시, 항상 web 반환
2. **웹 롤백**: useDeviceType 비활성화, 항상 platform="web"
3. **모바일 앱 긴급 패치**: CodePush로 JS 번들 업데이트

---

## Success Metrics (성공 지표)

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| API 응답 시간 | < 200ms | JMeter 부하 테스트 |
| 코드 재사용률 | 60-70% | SonarQube 분석 |
| Redis 캐시 히트율 | > 70% | Redis INFO stats |
| 플랫폼별 사용자 오류율 | < 1% | Sentry 모니터링 |

---

**다음 단계**: Backend Engineer plan.md 작성 및 UiController 구현 시작
