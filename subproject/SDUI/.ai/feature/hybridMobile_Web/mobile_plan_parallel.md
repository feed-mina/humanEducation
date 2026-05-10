# Backend Engineer — Mobile+Web Integration Plan (병행 접근 방식)

**접근 방식**: Next.js 웹 + React Native 모바일 병행 운영
**작성일**: 2026-03-01
**담당**: Backend Engineer Agent

---

## Research Analysis (연구 분석)

### 현재 백엔드 아키텍처

**기술 스택**:
- Spring Boot 3.1.4 (Java 17)
- PostgreSQL 15 (port 5433)
- Redis 6 (port 6379)
- Gradle 8.x

**핵심 컴포넌트**:
- `UiService.java`: 메타데이터 트리 빌딩 (O(n) LinkedHashMap)
- `JwtAuthenticationFilter`: JWT 검증 (Cookie 기반)
- `RedisConfig`: 캐시 설정 (TTL 1시간)
- `SecurityConfig`: 엔드포인트 권한 관리

### 현재 취약점 분석

기존 research.md에서 식별된 취약점:

| ID | 취약점 | 심각도 | 현재 상태 |
|----|--------|--------|----------|
| **CVE-001** | `/api/execute/**` 무인증 | CRITICAL | `.permitAll()` |
| **CVE-002** | JWT localStorage (웹) | HIGH | ✅ Cookie로 변경 완료 |
| **CVE-003** | WebSocket 인증 누락 | HIGH | 미해결 |
| **CVE-004** | JwtFilter role 하드코딩 | MEDIUM | ✅ JWT claims 추출 완료 |
| **CVE-005** | Redis SQL 캐시 무기한 | LOW | TTL 없음 |

### 모바일 앱 지원을 위한 추가 요구사항

#### 1. Authorization Bearer 헤더 지원
- **현재**: Cookie 기반 JWT 인증만 지원
- **문제**: React Native 앱은 HttpOnly Cookie 사용 불가
- **해결**: Bearer 토큰 방식 병행 지원

#### 2. 플랫폼 파라미터 처리
- **현재**: 모든 클라이언트에 동일한 메타데이터 반환
- **문제**: 모바일/웹 화면 차이 처리 불가
- **해결**: `?platform=mobile|web` 파라미터 추가

#### 3. JSONB component_props 병합
- **현재**: component_props를 그대로 반환
- **문제**: 플랫폼별 설정 처리 불가
- **해결**: `{"common":{}, "mobile":{}, "web":{}}` 구조 병합

#### 4. Redis 캐시 키 전략
- **현재**: `{rolePrefix}_{screenId}`
- **문제**: 플랫폼 구분 없이 캐싱
- **해결**: `{rolePrefix}_{screenId}_{platform}` 확장

---

## Implementation Plan (구현 계획)

### 1. SecurityConfig 수정 (CVE-001 수정 포함)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/config/SecurityConfig.java`

**현재 코드** (Line 85):
```java
.requestMatchers("/api/execute/**").permitAll() // ❌ CRITICAL 취약점
```

**수정 후**:
```java
// CVE-001 수정: 인증 필수로 변경
.requestMatchers("/api/execute/**").authenticated()
```

**전체 SecurityConfig**:
```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Autowired
    private JwtAuthenticationFilter jwtAuthenticationFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            .authorizeHttpRequests(auth -> auth
                // 인증 불필요 (공개 엔드포인트)
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/kakao/**").permitAll()
                .requestMatchers("/api/email/**").permitAll()
                .requestMatchers("/health").permitAll()

                // ✅ CVE-001 수정: 인증 필수
                .requestMatchers("/api/execute/**").authenticated()

                // 나머지 모든 API는 인증 필수
                .anyRequest().authenticated()
            )
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedOrigins(List.of(
            "http://localhost:3000",  // Next.js 웹 개발
            "http://localhost:8081",  // React Native Metro
            "https://yourapp.com"     // 프로덕션
        ));
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setAllowCredentials(true); // Cookie 지원
        config.setExposedHeaders(List.of("Authorization")); // Bearer 토큰 노출

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return source;
    }
}
```

### 2. JwtAuthenticationFilter 수정 (Bearer 토큰 지원)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/security/JwtAuthenticationFilter.java`

**현재 문제**:
- Cookie에서만 JWT 추출
- React Native 앱은 Cookie 사용 불가

**수정 후**:
```java
@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;
    private final UserRepository userRepository;

    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain filterChain
    ) throws ServletException, IOException {

        String token = null;

        // 1️⃣ Cookie에서 JWT 추출 (웹 클라이언트)
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if ("accessToken".equals(cookie.getName())) {
                    token = cookie.getValue();
                    break;
                }
            }
        }

        // 2️⃣ Authorization 헤더에서 JWT 추출 (모바일 앱)
        if (token == null) {
            String authHeader = request.getHeader("Authorization");
            if (authHeader != null && authHeader.startsWith("Bearer ")) {
                token = authHeader.substring(7);
            }
        }

        // 3️⃣ JWT 검증 및 인증 처리
        if (token != null) {
            try {
                Claims claims = jwtUtil.validateToken(token);
                String email = claims.getSubject();
                String role = claims.get("role", String.class); // ✅ CVE-004 수정

                // User 조회
                User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new UsernameNotFoundException("User not found"));

                // Spring Security 인증 객체 생성
                UsernamePasswordAuthenticationToken authentication =
                    new UsernamePasswordAuthenticationToken(
                        user,
                        null,
                        List.of(new SimpleGrantedAuthority(role))
                    );

                SecurityContextHolder.getContext().setAuthentication(authentication);

            } catch (ExpiredJwtException e) {
                log.warn("Expired JWT token: {}", e.getMessage());
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                return;
            } catch (JwtException e) {
                log.error("Invalid JWT token: {}", e.getMessage());
                response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                return;
            }
        }

        filterChain.doFilter(request, response);
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getServletPath();
        // 인증 불필요 경로는 필터 스킵
        return path.startsWith("/api/auth/")
            || path.startsWith("/api/kakao/")
            || path.startsWith("/api/email/")
            || path.equals("/health");
    }
}
```

### 3. UiController 플랫폼 파라미터 추가

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/controller/UiController.java`

**수정 전**:
```java
@GetMapping("/{screenId}")
public ApiResponse<List<UiResponseDto>> getUiMetadataList(
    @PathVariable String screenId,
    @AuthenticationPrincipal UserDetails userDetails
) {
    String userRole = extractRole(userDetails);
    List<UiResponseDto> tree = uiService.getUiTree(screenId, userRole);
    return ApiResponse.success(tree);
}
```

**수정 후**:
```java
@RestController
@RequestMapping("/api/ui")
@RequiredArgsConstructor
@Slf4j
public class UiController {

    private final UiService uiService;

    @GetMapping("/{screenId}")
    public ApiResponse<List<UiResponseDto>> getUiMetadataList(
        @PathVariable String screenId,
        @RequestParam(required = false, defaultValue = "web") String platform,
        @RequestHeader(value = "X-Platform", required = false) String headerPlatform,
        @RequestHeader(value = "User-Agent", required = false) String userAgent,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        // 플랫폼 화이트리스트 검증
        if (!List.of("web", "mobile").contains(platform)) {
            log.warn("Invalid platform parameter: {}, defaulting to 'web'", platform);
            platform = "web";
        }

        // 플랫폼 불일치 감지 (보안 모니터링)
        if (headerPlatform != null && !headerPlatform.equals(platform)) {
            log.warn("Platform mismatch detected - param: {}, header: {}, UA: {}, user: {}",
                     platform, headerPlatform, userAgent,
                     userDetails != null ? userDetails.getUsername() : "anonymous");
        }

        String userRole = extractRole(userDetails);
        List<UiResponseDto> tree = uiService.getUiTree(screenId, userRole, platform);
        return ApiResponse.success(tree);
    }

    private String extractRole(UserDetails userDetails) {
        if (userDetails == null) {
            return "ROLE_GUEST";
        }

        return userDetails.getAuthorities().stream()
            .findFirst()
            .map(GrantedAuthority::getAuthority)
            .orElse("ROLE_USER");
    }
}
```

### 4. UiService 플랫폼 필터링

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/service/UiService.java`

**수정 전**:
```java
@Cacheable(value = "uiMetadataCache", key = "#userRole + '_' + #screenId")
public List<UiResponseDto> getUiTree(String screenId, String userRole) {
    // ...
}
```

**수정 후**:
```java
@Service
@RequiredArgsConstructor
@Slf4j
public class UiService {

    private final UiMetadataRepository uiMetadataRepository;
    private final ObjectMapper objectMapper;

    /**
     * 플랫폼별 UI 메타데이터 트리 빌딩
     * Redis 캐시 키: {userRole}_{screenId}_{platform}
     */
    @Cacheable(
        value = "uiMetadataCache",
        key = "#userRole + '_' + #screenId + '_' + #platform"
    )
    public List<UiResponseDto> getUiTree(String screenId, String userRole, String platform) {
        log.info("Building UI tree - screen: {}, role: {}, platform: {}",
                 screenId, userRole, platform);

        List<UiMetadata> entities = uiMetadataRepository
            .findByScreenIdOrderByOrderIndexAsc(screenId);

        if (entities.isEmpty()) {
            log.warn("No metadata found for screenId: {}", screenId);
            return Collections.emptyList();
        }

        return buildTree(entities, userRole, platform);
    }

    /**
     * 트리 구조 빌딩 (O(n) 알고리즘 유지)
     */
    private List<UiResponseDto> buildTree(
        List<UiMetadata> entities,
        String userRole,
        String platform
    ) {
        Map<String, UiResponseDto> nodeMap = new LinkedHashMap<>();
        Map<String, List<UiResponseDto>> childrenMap = new HashMap<>();

        // 1단계: 모든 노드 생성 (RBAC 필터링 + JSONB 병합)
        for (UiMetadata entity : entities) {
            // RBAC 필터링
            if (!isAccessible(entity, userRole)) {
                continue;
            }

            UiResponseDto dto = new UiResponseDto();
            dto.setComponentId(entity.getComponentId());
            dto.setComponentType(entity.getComponentType());
            dto.setLabelText(entity.getLabelText());
            dto.setPlaceholder(entity.getPlaceholder());
            dto.setActionType(entity.getActionType());
            dto.setGroupDirection(entity.getGroupDirection());
            dto.setCssClass(entity.getCssClass());
            dto.setRefDataId(entity.getRefDataId());

            // ✅ JSONB component_props 플랫폼별 병합
            dto.setComponentProps(mergePlatformProps(
                entity.getComponentProps(),
                platform
            ));

            nodeMap.put(entity.getComponentId(), dto);

            // children map 초기화
            String parentId = entity.getParentGroupId();
            if (parentId != null) {
                childrenMap.computeIfAbsent(parentId, k -> new ArrayList<>())
                           .add(dto);
            }
        }

        // 2단계: 부모-자식 관계 설정
        for (Map.Entry<String, List<UiResponseDto>> entry : childrenMap.entrySet()) {
            String parentId = entry.getKey();
            List<UiResponseDto> children = entry.getValue();

            UiResponseDto parent = nodeMap.get(parentId);
            if (parent != null) {
                parent.setChildren(children);
            }
        }

        // 3단계: 루트 노드만 반환 (parent_group_id == null)
        return nodeMap.values().stream()
            .filter(dto -> entities.stream()
                .filter(e -> e.getComponentId().equals(dto.getComponentId()))
                .anyMatch(e -> e.getParentGroupId() == null))
            .collect(Collectors.toList());
    }

    /**
     * JSONB component_props 병합 로직
     * 구조: {"common": {...}, "mobile": {...}, "web": {...}}
     * 결과: common + platform (platform이 common 오버라이드)
     */
    @SuppressWarnings("unchecked")
    private String mergePlatformProps(String componentPropsJson, String platform) {
        if (componentPropsJson == null || componentPropsJson.isBlank()) {
            return "{}";
        }

        try {
            Map<String, Object> allProps = objectMapper.readValue(
                componentPropsJson, Map.class
            );

            // common props (모든 플랫폼 공통)
            Map<String, Object> commonProps =
                (Map<String, Object>) allProps.getOrDefault("common", new HashMap<>());

            // 플랫폼별 props
            Map<String, Object> platformProps =
                (Map<String, Object>) allProps.getOrDefault(platform, new HashMap<>());

            // 병합: common + platform (플랫폼 설정이 우선)
            Map<String, Object> merged = new HashMap<>(commonProps);
            merged.putAll(platformProps);

            return objectMapper.writeValueAsString(merged);

        } catch (JsonProcessingException e) {
            log.error("Failed to parse component_props: {}", componentPropsJson, e);
            return "{}";
        }
    }

    /**
     * RBAC 접근 제어 (allowed_roles 확인)
     */
    private boolean isAccessible(UiMetadata entity, String userRole) {
        String allowedRoles = entity.getAllowedRoles();

        // allowed_roles가 null이면 모두 접근 가능
        if (allowedRoles == null || allowedRoles.isBlank()) {
            return true;
        }

        // 쉼표 구분 역할 목록 확인
        return Arrays.asList(allowedRoles.split(","))
            .contains(userRole);
    }
}
```

### 5. Rate Limiting 추가 (Bucket4j + Redis)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/security/RateLimitFilter.java` (신규)

```java
@Component
@RequiredArgsConstructor
@Slf4j
public class RateLimitFilter extends OncePerRequestFilter {

    private final StringRedisTemplate redisTemplate;

    private static final int MAX_REQUESTS_PER_MINUTE = 100;

    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain filterChain
    ) throws ServletException, IOException {

        String clientIp = getClientIp(request);
        String key = "rate_limit:" + clientIp;

        try {
            Long requestCount = redisTemplate.opsForValue().increment(key);

            // 첫 요청 시 TTL 설정 (1분)
            if (requestCount == 1) {
                redisTemplate.expire(key, 1, TimeUnit.MINUTES);
            }

            // 임계값 초과 시 429 반환
            if (requestCount > MAX_REQUESTS_PER_MINUTE) {
                log.warn("Rate limit exceeded for IP: {}, count: {}", clientIp, requestCount);
                response.setStatus(HttpServletResponse.SC_TOO_MANY_REQUESTS);
                response.setContentType("application/json");
                response.getWriter().write(
                    "{\"success\":false,\"message\":\"Too many requests. Please try again later.\"}"
                );
                return;
            }

        } catch (Exception e) {
            log.error("Rate limiting error: {}", e.getMessage());
            // Rate limiting 실패 시에도 요청은 통과 (가용성 우선)
        }

        filterChain.doFilter(request, response);
    }

    /**
     * 클라이언트 IP 추출 (프록시 환경 고려)
     */
    private String getClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty()) {
            ip = request.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isEmpty()) {
            ip = request.getRemoteAddr();
        }
        return ip;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        // 정적 리소스, health check는 Rate Limiting 제외
        String path = request.getServletPath();
        return path.equals("/health") || path.startsWith("/static/");
    }
}
```

**SecurityConfig에 필터 추가**:
```java
@Bean
public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
    http
        // ...
        .addFilterBefore(rateLimitFilter, JwtAuthenticationFilter.class)
        .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

    return http.build();
}
```

### 6. AuthController 수정 (모바일 앱 Bearer 토큰 지원)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/user/controller/AuthController.java`

**현재 문제**:
- 로그인 성공 시 Cookie에만 JWT 저장
- 모바일 앱은 Cookie 사용 불가

**수정 후**:
```java
@PostMapping("/login")
public ResponseEntity<ApiResponse<LoginResponse>> login(
    @RequestBody LoginRequest request,
    HttpServletResponse response,
    @RequestHeader(value = "X-Platform", required = false) String platform
) {
    User user = authService.authenticate(request.getEmail(), request.getPassword());

    String accessToken = jwtUtil.createAccessToken(user);
    String refreshToken = jwtUtil.createRefreshToken(user);

    // 1️⃣ 웹 클라이언트: Cookie 저장
    if ("web".equals(platform) || platform == null) {
        Cookie accessCookie = new Cookie("accessToken", accessToken);
        accessCookie.setHttpOnly(true);
        accessCookie.setSecure(true); // HTTPS only
        accessCookie.setPath("/");
        accessCookie.setMaxAge(60 * 60); // 1시간
        response.addCookie(accessCookie);

        Cookie refreshCookie = new Cookie("refreshToken", refreshToken);
        refreshCookie.setHttpOnly(true);
        refreshCookie.setSecure(true);
        refreshCookie.setPath("/");
        refreshCookie.setMaxAge(7 * 24 * 60 * 60); // 7일
        response.addCookie(refreshCookie);
    }

    // 2️⃣ 모바일 앱: Response Body에 토큰 반환
    LoginResponse loginResponse = new LoginResponse();
    loginResponse.setUserId(user.getId());
    loginResponse.setEmail(user.getEmail());
    loginResponse.setNickname(user.getNickname());
    loginResponse.setRole(user.getRole());

    if ("mobile".equals(platform)) {
        loginResponse.setAccessToken(accessToken);   // 모바일만 토큰 반환
        loginResponse.setRefreshToken(refreshToken);
    }

    return ResponseEntity.ok(ApiResponse.success(loginResponse));
}
```

**LoginResponse DTO**:
```java
@Data
public class LoginResponse {
    private Long userId;
    private String email;
    private String nickname;
    private String role;

    // 모바일 전용 (웹은 null)
    private String accessToken;
    private String refreshToken;
}
```

### 7. RedisConfig TTL 설정 (CVE-005 수정)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/config/RedisConfig.java`

**추가**:
```java
@Configuration
@EnableCaching
public class RedisConfig {

    @Bean
    public RedisCacheManager cacheManager(RedisConnectionFactory connectionFactory) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofHours(1))  // ✅ CVE-005 수정: 1시간 TTL
            .serializeKeysWith(
                RedisSerializationContext.SerializationPair
                    .fromSerializer(new StringRedisSerializer())
            )
            .serializeValuesWith(
                RedisSerializationContext.SerializationPair
                    .fromSerializer(new GenericJackson2JsonRedisSerializer())
            );

        return RedisCacheManager.builder(connectionFactory)
            .cacheDefaults(config)
            .build();
    }
}
```

---

## Security Considerations (보안 고려사항)

### 1. JWT 저장 위치

| Platform | Storage | XSS | CSRF | 루팅/탈옥 |
|----------|---------|-----|------|-----------|
| Web | HttpOnly Cookie | ✅ 안전 | ⚠️ CSRF 토큰 필요 | N/A |
| Mobile | SecureStore (Bearer) | ✅ 안전 | ✅ 안전 | ⚠️ 루팅 시 위험 |

**완화 방안**:
- 웹: CSRF 토큰 추가 (Spring Security CSRF)
- 모바일: Root Detection + Certificate Pinning

### 2. JSONB SQL Injection

**위협**: component_props에 악의적 JSON 삽입

**완화**:
```java
// JSON 파싱 검증
try {
    objectMapper.readTree(componentProps);
} catch (JsonProcessingException e) {
    log.error("Invalid JSONB: {}", componentProps);
    return "{}";
}

// PreparedStatement 사용 (JPA는 자동 처리)
```

### 3. Rate Limiting Bypass

**위협**: IP 위장 (X-Forwarded-For 조작)

**완화**:
```java
// 여러 헤더 확인
String ip = request.getHeader("X-Forwarded-For");
if (ip == null) ip = request.getHeader("X-Real-IP");
if (ip == null) ip = request.getRemoteAddr();

// User-based Rate Limiting 추가
String userId = userDetails.getUsername();
String key = "rate_limit:user:" + userId;
```

### 4. Platform Header 위장

**위협**: X-Platform 헤더 조작

**완화**:
```java
// User-Agent 교차 검증
if ("mobile".equals(platform) && !userAgent.contains("ReactNative")) {
    log.warn("Suspicious platform claim: {}, UA: {}", platform, userAgent);
}

// Rate Limiting으로 남용 방지
```

---

## Test Plan (테스트 계획)

### 단위 테스트 (JUnit 5)

**파일**: `SDUI-server/src/test/java/com/domain/demo_backend/domain/ui/service/UiServiceTest.java`

```java
@SpringBootTest
class UiServiceTest {

    @Autowired
    private UiService uiService;

    @Test
    void getUiTree_withMobilePlatform_returnsMobileProps() {
        // Given
        String screenId = "LOGIN_PAGE";
        String userRole = "ROLE_USER";
        String platform = "mobile";

        // When
        List<UiResponseDto> result = uiService.getUiTree(screenId, userRole, platform);

        // Then
        assertNotNull(result);
        UiResponseDto button = result.stream()
            .filter(dto -> "submitBtn".equals(dto.getComponentId()))
            .findFirst()
            .orElseThrow();

        Map<String, Object> props = objectMapper.readValue(
            button.getComponentProps(), Map.class
        );
        assertEquals(48, props.get("minHeight")); // 모바일 터치 타겟
    }

    @Test
    void getUiTree_withWebPlatform_returnsWebProps() {
        List<UiResponseDto> result = uiService.getUiTree("LOGIN_PAGE", "ROLE_USER", "web");

        UiResponseDto button = result.stream()
            .filter(dto -> "submitBtn".equals(dto.getComponentId()))
            .findFirst()
            .orElseThrow();

        Map<String, Object> props = objectMapper.readValue(
            button.getComponentProps(), Map.class
        );
        assertEquals(40, props.get("minHeight")); // 웹 버튼
    }

    @Test
    void getUiTree_withInvalidPlatform_defaultsToWeb() {
        List<UiResponseDto> result = uiService.getUiTree("LOGIN_PAGE", "ROLE_USER", "invalid");
        assertNotNull(result);
    }

    @Test
    void mergePlatformProps_withValidJson_mergesCorrectly() {
        String json = "{\"common\":{\"a\":1},\"mobile\":{\"a\":2,\"b\":3}}";
        String merged = uiService.mergePlatformProps(json, "mobile");

        Map<String, Object> result = objectMapper.readValue(merged, Map.class);
        assertEquals(2, result.get("a")); // mobile이 common 오버라이드
        assertEquals(3, result.get("b"));
    }
}
```

### 통합 테스트 (Spring Boot Test)

**파일**: `SDUI-server/src/test/java/com/domain/demo_backend/domain/ui/controller/UiControllerIntegrationTest.java`

```java
@SpringBootTest
@AutoConfigureMockMvc
class UiControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private JwtUtil jwtUtil;

    @Test
    void getUiMetadata_withBearerToken_returnsSuccess() throws Exception {
        // Given
        User testUser = new User("test@example.com", "ROLE_USER");
        String token = jwtUtil.createAccessToken(testUser);

        // When & Then
        mockMvc.perform(get("/api/ui/LOGIN_PAGE")
                .param("platform", "mobile")
                .header("Authorization", "Bearer " + token)
                .header("X-Platform", "mobile"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.success").value(true))
            .andExpect(jsonPath("$.data").isArray())
            .andExpect(jsonPath("$.data[0].componentProps").exists());
    }

    @Test
    void getUiMetadata_withCookie_returnsSuccess() throws Exception {
        User testUser = new User("test@example.com", "ROLE_USER");
        String token = jwtUtil.createAccessToken(testUser);

        mockMvc.perform(get("/api/ui/LOGIN_PAGE")
                .param("platform", "web")
                .cookie(new Cookie("accessToken", token)))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.success").value(true));
    }

    @Test
    void getUiMetadata_withoutAuth_returns403() throws Exception {
        mockMvc.perform(get("/api/ui/LOGIN_PAGE"))
            .andExpect(status().isForbidden());
    }
}
```

### 보안 회귀 테스트

```java
@Test
void executeApi_withoutAuth_returns403() throws Exception {
    // CVE-001 수정 검증
    mockMvc.perform(post("/api/execute/GET_DIARY_LIST"))
        .andExpect(status().isForbidden());
}

@Test
void rateLimiting_exceedsLimit_returns429() throws Exception {
    String token = jwtUtil.createAccessToken(testUser);

    for (int i = 0; i < 101; i++) {
        mockMvc.perform(get("/api/ui/LOGIN_PAGE")
                .header("Authorization", "Bearer " + token));
    }

    mockMvc.perform(get("/api/ui/LOGIN_PAGE")
            .header("Authorization", "Bearer " + token))
        .andExpect(status().isTooManyRequests());
}

@Test
void platformMismatch_logsWarning() throws Exception {
    String token = jwtUtil.createAccessToken(testUser);

    mockMvc.perform(get("/api/ui/LOGIN_PAGE")
            .param("platform", "web")
            .header("X-Platform", "mobile") // 불일치
            .header("Authorization", "Bearer " + token))
        .andExpect(status().isOk());

    // 로그 검증 필요 (logback-test.xml의 MemoryAppender 사용)
}
```

---

## Dependencies (의존성)

### Depends on
- Architect: JSONB 스키마 정의, API 스펙 확정

### Blocks
- Frontend Engineer: API 배포 후 웹/앱 통합 가능
- QA Engineer: 백엔드 테스트 완료 후 E2E 테스트 시작

---

## Implementation Sequence (구현 순서)

```
Week 1:
  Day 1-2: SecurityConfig CVE-001 수정 + JwtAuthenticationFilter Bearer 지원
  Day 3-4: UiController/UiService 플랫폼 파라미터 추가
  Day 5:   단위 테스트 작성

Week 2:
  Day 1-2: Rate Limiting 구현
  Day 3-4: AuthController 모바일 지원
  Day 5:   통합 테스트 + 보안 회귀 테스트
```

---

## Rollback Plan (롤백 계획)

1. **플랫폼 파라미터 비활성화**:
```java
platform = "web"; // 강제로 web 고정
```

2. **Bearer 토큰 임시 비활성화**:
```java
// JwtAuthenticationFilter에서 Authorization 헤더 처리 주석 처리
```

3. **CVE-001 긴급 롤백**:
```java
.requestMatchers("/api/execute/**").permitAll() // 임시 복구
```

---

## Success Metrics (성공 지표)

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| API 응답 시간 | < 200ms | JMeter 부하 테스트 |
| Redis 캐시 히트율 | > 70% | Redis INFO stats |
| JWT 검증 성공률 | > 99.9% | 로그 분석 |
| Rate Limiting 정확도 | 100% | 단위 테스트 |
| CVE 취약점 | 0 CRITICAL | OWASP ZAP 스캔 |

---

**다음 단계**: Frontend Engineer plan.md 작성 및 웹/앱 클라이언트 통합
