# Backend Engineer — Mobile+Web Integration Plan (단계적 전환 방식)

**접근 방식**: Hybrid (Next.js 웹 유지 → Expo 모바일 추가 → 데이터 기반 통합 결정)
**작성일**: 2026-03-01
**담당**: Backend Engineer Agent

---

## Research Analysis (연구 분석)

### 단계적 전환 전략

Hybrid 방식은 3단계로 진행되며, 백엔드는 **Phase 1**에서 한 번만 수정:

```
Phase 1 (Month 1-2): 백엔드 API 플랫폼 지원 추가
  → UiController/UiService platform 파라미터 지원
  → JwtAuthenticationFilter Bearer 토큰 지원
  → CVE-001~005 취약점 수정

Phase 2 (Month 3-5): Expo 모바일 앱 개발
  → 백엔드 변경 없음 (Phase 1에서 준비 완료)
  → 모바일 앱은 Bearer 토큰 + platform=mobile 사용

Phase 3 (Month 6): 데이터 기반 결정
  → 백엔드 변경 없음
  → SEO 트래픽 분석 후 프론트엔드만 통합 결정
```

### 백엔드 목표 (Phase 1)

✅ **플랫폼 중립적 API**: 웹/앱 모두 지원하는 단일 API
✅ **인증 방식 병행**: Cookie (웹) + Bearer (앱) 동시 지원
✅ **캐시 최적화**: 플랫폼별 Redis 캐시 분리
✅ **보안 강화**: CVE 5개 취약점 수정 + Rate Limiting

### 백엔드 변경 사항 요약

| 컴포넌트 | 변경 내용 | 목적 |
|----------|----------|------|
| SecurityConfig | `.permitAll()` → `.authenticated()` | CVE-001 수정 |
| JwtAuthenticationFilter | Bearer 토큰 추출 추가 | 모바일 앱 지원 |
| UiController | `@RequestParam platform` 추가 | 플랫폼별 메타데이터 |
| UiService | JSONB 병합 로직 추가 | component_props 플랫폼 분기 |
| RedisConfig | TTL 1시간 설정 | CVE-005 수정 |
| RateLimitFilter | 신규 생성 | API 남용 방지 |
| AuthController | 모바일 응답 토큰 추가 | Bearer 토큰 반환 |

---

## Implementation Plan (구현 계획)

### Phase 1: Backend API 플랫폼 지원 (Month 1-2)

**Week 1**: 보안 취약점 수정 + Bearer 토큰 지원
**Week 2**: 플랫폼 파라미터 처리 + JSONB 병합 로직

#### 1. SecurityConfig 수정 (CVE-001)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/config/SecurityConfig.java`

**변경사항**:
```java
// ❌ BEFORE
.requestMatchers("/api/execute/**").permitAll()

// ✅ AFTER
.requestMatchers("/api/execute/**").authenticated()
```

**전체 코드**:
```java
@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    private final RateLimitFilter rateLimitFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/kakao/**").permitAll()
                .requestMatchers("/api/email/**").permitAll()
                .requestMatchers("/health").permitAll()
                .requestMatchers("/api/execute/**").authenticated() // ✅ 수정
                .anyRequest().authenticated()
            )
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            .addFilterBefore(rateLimitFilter, JwtAuthenticationFilter.class)
            .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();

        // Phase 2에서 Expo 앱 추가 시 origin 확장 가능
        config.setAllowedOrigins(List.of(
            "http://localhost:3000",     // Next.js dev
            "http://localhost:8081",     // Expo Metro bundler (Phase 2)
            "https://yourapp.com"        // Production
        ));

        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setAllowCredentials(true);
        config.setExposedHeaders(List.of("Authorization")); // Bearer 토큰 노출

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return source;
    }
}
```

#### 2. JwtAuthenticationFilter 수정 (Cookie + Bearer)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/security/JwtAuthenticationFilter.java`

**현재 문제**:
- Cookie에서만 JWT 추출
- Expo 앱은 Cookie 사용 불가 → Bearer 토큰 필요

**수정 후**:
```java
@Component
@RequiredArgsConstructor
@Slf4j
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;
    private final UserRepository userRepository;

    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain filterChain
    ) throws ServletException, IOException {

        String token = extractToken(request);

        if (token != null) {
            try {
                Claims claims = jwtUtil.validateToken(token);
                String email = claims.getSubject();
                String role = claims.get("role", String.class); // ✅ CVE-004 수정

                User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new UsernameNotFoundException("User not found"));

                UsernamePasswordAuthenticationToken authentication =
                    new UsernamePasswordAuthenticationToken(
                        user,
                        null,
                        List.of(new SimpleGrantedAuthority(role))
                    );

                SecurityContextHolder.getContext().setAuthentication(authentication);

            } catch (ExpiredJwtException e) {
                log.warn("Expired JWT: {}", e.getMessage());
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                return;
            } catch (JwtException e) {
                log.error("Invalid JWT: {}", e.getMessage());
                response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                return;
            }
        }

        filterChain.doFilter(request, response);
    }

    /**
     * JWT 토큰 추출: Cookie (웹) 또는 Authorization 헤더 (모바일)
     */
    private String extractToken(HttpServletRequest request) {
        // 1️⃣ Cookie 우선 (웹 클라이언트)
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if ("accessToken".equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }

        // 2️⃣ Authorization 헤더 (모바일 앱)
        String authHeader = request.getHeader("Authorization");
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            return authHeader.substring(7);
        }

        return null;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getServletPath();
        return path.startsWith("/api/auth/")
            || path.startsWith("/api/kakao/")
            || path.startsWith("/api/email/")
            || path.equals("/health");
    }
}
```

#### 3. UiController 플랫폼 파라미터

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/controller/UiController.java`

```java
@RestController
@RequestMapping("/api/ui")
@RequiredArgsConstructor
@Slf4j
public class UiController {

    private final UiService uiService;

    /**
     * UI 메타데이터 조회 (플랫폼별)
     * @param screenId 화면 ID (LOGIN_PAGE, DIARY_LIST 등)
     * @param platform 플랫폼 (web, mobile)
     * @param headerPlatform X-Platform 헤더 (검증용)
     * @param userDetails 인증된 사용자 정보
     */
    @GetMapping("/{screenId}")
    public ApiResponse<List<UiResponseDto>> getUiMetadataList(
        @PathVariable String screenId,
        @RequestParam(required = false, defaultValue = "web") String platform,
        @RequestHeader(value = "X-Platform", required = false) String headerPlatform,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        // 플랫폼 화이트리스트
        if (!List.of("web", "mobile").contains(platform)) {
            log.warn("Invalid platform: {}, defaulting to 'web'", platform);
            platform = "web";
        }

        // 플랫폼 불일치 로깅 (보안 모니터링)
        if (headerPlatform != null && !headerPlatform.equals(platform)) {
            log.warn("Platform mismatch - param: {}, header: {}, user: {}",
                     platform, headerPlatform,
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

#### 4. UiService JSONB 병합

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/ui/service/UiService.java`

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class UiService {

    private final UiMetadataRepository uiMetadataRepository;
    private final ObjectMapper objectMapper;

    /**
     * 플랫폼별 UI 메타데이터 트리 조회
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

    private List<UiResponseDto> buildTree(
        List<UiMetadata> entities,
        String userRole,
        String platform
    ) {
        Map<String, UiResponseDto> nodeMap = new LinkedHashMap<>();
        Map<String, List<UiResponseDto>> childrenMap = new HashMap<>();

        for (UiMetadata entity : entities) {
            // RBAC 필터링
            if (!isAccessible(entity, userRole)) {
                continue;
            }

            UiResponseDto dto = toDto(entity, platform);
            nodeMap.put(entity.getComponentId(), dto);

            if (entity.getParentGroupId() != null) {
                childrenMap.computeIfAbsent(
                    entity.getParentGroupId(),
                    k -> new ArrayList<>()
                ).add(dto);
            }
        }

        // 부모-자식 관계 설정
        childrenMap.forEach((parentId, children) -> {
            UiResponseDto parent = nodeMap.get(parentId);
            if (parent != null) {
                parent.setChildren(children);
            }
        });

        // 루트 노드만 반환
        return nodeMap.values().stream()
            .filter(dto -> entities.stream()
                .anyMatch(e -> e.getComponentId().equals(dto.getComponentId())
                            && e.getParentGroupId() == null))
            .collect(Collectors.toList());
    }

    private UiResponseDto toDto(UiMetadata entity, String platform) {
        UiResponseDto dto = new UiResponseDto();
        dto.setComponentId(entity.getComponentId());
        dto.setComponentType(entity.getComponentType());
        dto.setLabelText(entity.getLabelText());
        dto.setPlaceholder(entity.getPlaceholder());
        dto.setActionType(entity.getActionType());
        dto.setGroupDirection(entity.getGroupDirection());
        dto.setCssClass(entity.getCssClass());
        dto.setRefDataId(entity.getRefDataId());

        // ✅ JSONB 플랫폼별 병합
        dto.setComponentProps(mergePlatformProps(
            entity.getComponentProps(),
            platform
        ));

        return dto;
    }

    /**
     * component_props JSONB 병합
     * 입력: {"common": {...}, "mobile": {...}, "web": {...}}
     * 출력: common + platform (platform이 common 오버라이드)
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

            Map<String, Object> common =
                (Map<String, Object>) allProps.getOrDefault("common", new HashMap<>());

            Map<String, Object> platformSpecific =
                (Map<String, Object>) allProps.getOrDefault(platform, new HashMap<>());

            // 병합: common + platform
            Map<String, Object> merged = new HashMap<>(common);
            merged.putAll(platformSpecific);

            return objectMapper.writeValueAsString(merged);

        } catch (JsonProcessingException e) {
            log.error("Invalid JSONB component_props: {}", componentPropsJson, e);
            return "{}";
        }
    }

    private boolean isAccessible(UiMetadata entity, String userRole) {
        String allowedRoles = entity.getAllowedRoles();
        if (allowedRoles == null || allowedRoles.isBlank()) {
            return true;
        }
        return Arrays.asList(allowedRoles.split(",")).contains(userRole);
    }
}
```

#### 5. Rate Limiting

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/security/RateLimitFilter.java` (신규)

```java
@Component
@RequiredArgsConstructor
@Slf4j
public class RateLimitFilter extends OncePerRequestFilter {

    private final StringRedisTemplate redisTemplate;
    private static final int MAX_REQUESTS = 100; // 분당 100 요청

    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain filterChain
    ) throws ServletException, IOException {

        String clientIp = getClientIp(request);
        String key = "rate_limit:" + clientIp;

        try {
            Long count = redisTemplate.opsForValue().increment(key);

            if (count == 1) {
                redisTemplate.expire(key, 1, TimeUnit.MINUTES);
            }

            if (count > MAX_REQUESTS) {
                log.warn("Rate limit exceeded: IP={}, count={}", clientIp, count);
                response.setStatus(429);
                response.setContentType("application/json");
                response.getWriter().write(
                    "{\"success\":false,\"message\":\"Too many requests\"}"
                );
                return;
            }

        } catch (Exception e) {
            log.error("Rate limiting error: {}", e.getMessage());
        }

        filterChain.doFilter(request, response);
    }

    private String getClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null) ip = request.getHeader("X-Real-IP");
        if (ip == null) ip = request.getRemoteAddr();
        return ip;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        return request.getServletPath().equals("/health");
    }
}
```

#### 6. AuthController 모바일 지원

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/domain/user/controller/AuthController.java`

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

    // 웹: Cookie 저장
    if ("web".equals(platform) || platform == null) {
        addCookie(response, "accessToken", accessToken, 3600); // 1시간
        addCookie(response, "refreshToken", refreshToken, 604800); // 7일
    }

    // 응답 생성
    LoginResponse loginResponse = new LoginResponse();
    loginResponse.setUserId(user.getId());
    loginResponse.setEmail(user.getEmail());
    loginResponse.setNickname(user.getNickname());
    loginResponse.setRole(user.getRole());

    // 모바일: 토큰 반환
    if ("mobile".equals(platform)) {
        loginResponse.setAccessToken(accessToken);
        loginResponse.setRefreshToken(refreshToken);
    }

    return ResponseEntity.ok(ApiResponse.success(loginResponse));
}

private void addCookie(HttpServletResponse response, String name, String value, int maxAge) {
    Cookie cookie = new Cookie(name, value);
    cookie.setHttpOnly(true);
    cookie.setSecure(true); // HTTPS only
    cookie.setPath("/");
    cookie.setMaxAge(maxAge);
    response.addCookie(cookie);
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

#### 7. RedisConfig TTL (CVE-005)

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/config/RedisConfig.java`

```java
@Configuration
@EnableCaching
public class RedisConfig {

    @Bean
    public RedisCacheManager cacheManager(RedisConnectionFactory connectionFactory) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofHours(1))  // ✅ CVE-005 수정
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

### Phase 2-3: 백엔드 변경 없음

Phase 1에서 백엔드는 플랫폼 중립적으로 구현되므로, Phase 2(Expo 앱 개발)와 Phase 3(통합 결정)에서는 **백엔드 코드 변경 불필요**.

#### Phase 2 준비 사항
- ✅ Bearer 토큰 지원 (이미 완료)
- ✅ platform=mobile 파라미터 처리 (이미 완료)
- ✅ CORS Expo Metro bundler 허용 (SecurityConfig에 이미 추가)

#### Phase 3 준비 사항
- ✅ SEO 트래픽 분석은 프론트엔드 작업
- ✅ Expo Web 통합 시 백엔드 변경 없음 (동일 API 사용)

---

## Security Considerations (보안 고려사항)

### 수정된 취약점

| ID | 취약점 | 수정 내용 | 검증 방법 |
|----|--------|----------|----------|
| CVE-001 | `/api/execute/**` 무인증 | `.authenticated()` 추가 | 단위 테스트 |
| CVE-002 | localStorage JWT | ✅ 이미 Cookie 사용 | N/A |
| CVE-004 | role 하드코딩 | JWT claims 추출 | 통합 테스트 |
| CVE-005 | Redis 무기한 캐시 | TTL 1시간 설정 | Redis TTL 확인 |

### 신규 보안 기능

#### 1. Rate Limiting
- **목표**: IP 기반 분당 100 요청 제한
- **우회 방지**: X-Forwarded-For, X-Real-IP 다중 검증

#### 2. Platform Header 검증
- **목표**: X-Platform 헤더 위장 감지
- **방법**: 파라미터-헤더 불일치 로깅

#### 3. JSONB 검증
- **목표**: component_props SQL Injection 방지
- **방법**: JSON 파싱 유효성 검사

---

## Test Plan (테스트 계획)

### 단위 테스트 (JUnit 5)

**파일**: `SDUI-server/src/test/java/com/domain/demo_backend/domain/ui/service/UiServiceTest.java`

```java
@SpringBootTest
class UiServiceTest {

    @Autowired
    private UiService uiService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void getUiTree_withMobilePlatform_returnsMobileProps() throws Exception {
        List<UiResponseDto> result = uiService.getUiTree(
            "LOGIN_PAGE", "ROLE_USER", "mobile"
        );

        UiResponseDto button = result.stream()
            .filter(dto -> "submitBtn".equals(dto.getComponentId()))
            .findFirst()
            .orElseThrow();

        Map<String, Object> props = objectMapper.readValue(
            button.getComponentProps(), Map.class
        );
        assertEquals(48, props.get("minHeight")); // 모바일
    }

    @Test
    void getUiTree_withWebPlatform_returnsWebProps() throws Exception {
        List<UiResponseDto> result = uiService.getUiTree(
            "LOGIN_PAGE", "ROLE_USER", "web"
        );

        UiResponseDto button = result.stream()
            .filter(dto -> "submitBtn".equals(dto.getComponentId()))
            .findFirst()
            .orElseThrow();

        Map<String, Object> props = objectMapper.readValue(
            button.getComponentProps(), Map.class
        );
        assertEquals(40, props.get("minHeight")); // 웹
    }
}
```

### 통합 테스트

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
        User user = new User("test@example.com", "ROLE_USER");
        String token = jwtUtil.createAccessToken(user);

        mockMvc.perform(get("/api/ui/LOGIN_PAGE")
                .param("platform", "mobile")
                .header("Authorization", "Bearer " + token)
                .header("X-Platform", "mobile"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.success").value(true));
    }

    @Test
    void executeApi_withoutAuth_returns403() throws Exception {
        // CVE-001 수정 검증
        mockMvc.perform(post("/api/execute/GET_DIARY_LIST"))
            .andExpect(status().isForbidden());
    }

    @Test
    void rateLimiting_exceedsLimit_returns429() throws Exception {
        for (int i = 0; i < 101; i++) {
            mockMvc.perform(get("/api/ui/LOGIN_PAGE"));
        }

        mockMvc.perform(get("/api/ui/LOGIN_PAGE"))
            .andExpect(status().isTooManyRequests());
    }
}
```

---

## Dependencies (의존성)

### Depends on
- Architect: JSONB 스키마 정의

### Blocks
- Frontend Engineer (Phase 1): 웹 통합 시작 가능
- Frontend Engineer (Phase 2): Expo 앱 개발 시작 가능
- QA Engineer: 백엔드 테스트 완료 후 E2E 가능

---

## Timeline (타임라인)

```
Month 1-2: Phase 1 (Backend 한 번만 수정)
  Week 1: CVE-001~005 수정 + Bearer 토큰 지원
  Week 2: 플랫폼 파라미터 + JSONB 병합 + Rate Limiting

Month 3-6: Phase 2-3 (Backend 변경 없음)
  → Expo 앱 개발 및 SEO 데이터 수집 (Frontend 작업)
```

---

## Rollback Plan (롤백 계획)

### Phase 1 롤백
```java
// 플랫폼 파라미터 무시
platform = "web"; // 강제 web 고정

// CVE-001 긴급 롤백 (비추천)
.requestMatchers("/api/execute/**").permitAll()
```

---

## Success Metrics (성공 지표)

| Phase | 지표 | 목표 |
|-------|------|------|
| Phase 1 | API 응답 시간 | < 200ms |
| Phase 1 | Redis 캐시 히트율 | > 70% |
| Phase 1 | CVE 취약점 | 0 CRITICAL |
| Phase 2-3 | 백엔드 변경 | 0회 (불필요) |

---

**다음 단계**: Frontend Engineer plan.md 작성 (Phase 1: 웹 통합, Phase 2: Expo 앱 개발)
