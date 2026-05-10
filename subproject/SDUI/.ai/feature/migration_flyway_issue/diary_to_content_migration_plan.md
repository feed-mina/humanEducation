# diary → content 백엔드 마이그레이션 계획

**작성자**: Backend Engineer
**작성일**: 2026-03-01
**목적**: diary 관련 백엔드 코드를 content로 전환

---

## 📋 수정 대상 파일 (6개)

### 1. 엔티티
- `domain/diary/domain/Diary.java` → `domain/content/domain/Content.java`

### 2. 리포지토리
- `domain/diary/domain/DiaryRepository.java` → `domain/content/domain/ContentRepository.java`

### 3. DTO
- `domain/diary/dto/DiaryRequest.java` → `domain/content/dto/ContentRequest.java`
- `domain/diary/dto/DiaryResponse.java` → `domain/content/dto/ContentResponse.java`

### 4. 서비스
- `domain/diary/service/DiaryService.java` → `domain/content/service/ContentService.java`

### 5. 컨트롤러
- `domain/diary/controller/DiaryController.java` → `domain/content/controller/ContentController.java`

---

## 🔧 상세 수정 사항

### 1. Content.java (엔티티)

**파일 경로**: `domain/content/domain/Content.java`

#### 수정 내용
```java
// 변경 전
@Entity
@Table(name = "diary")
public class Diary {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "diary_id")
    private Long diaryId;

    @Column(name = "content")
    private String content;
}

// 변경 후
@Entity
@Table(name = "content")
public class Content {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "content_id")
    private Long contentId;

    @Column(name = "content_text")  // content → content_text (컬럼명 중복 방지)
    private String contentText;
}
```

#### 주요 변경점
- 클래스명: `Diary` → `Content`
- 테이블명: `@Table(name = "diary")` → `@Table(name = "content")`
- PK 필드: `diaryId` → `contentId`
- PK 컬럼: `diary_id` → `content_id`
- content 필드: `content` → `contentText`
- 컬럼명: `@Column(name = "content")` → `@Column(name = "content_text")`

---

### 2. ContentRepository.java

```java
// 변경 전
public interface DiaryRepository extends JpaRepository<Diary, Long> {
    List<Diary> findByUserSqno(Long userSqno);
}

// 변경 후
public interface ContentRepository extends JpaRepository<Content, Long> {
    List<Content> findByUserSqno(Long userSqno);
}
```

---

### 3. ContentRequest.java (DTO)

```java
// 변경 전
public class DiaryRequest {
    private String content;
}

// 변경 후
public class ContentRequest {
    private String contentText;  // content → contentText
}
```

---

### 4. ContentResponse.java (DTO)

```java
// 변경 전
public class DiaryResponse {
    private Long diaryId;
    private String content;

    public DiaryResponse(Diary diary) {
        this.diaryId = diary.getDiaryId();
        this.content = diary.getContent();
    }
}

// 변경 후
public class ContentResponse {
    private Long contentId;
    private String contentText;

    public ContentResponse(Content content) {
        this.contentId = content.getContentId();
        this.contentText = content.getContentText();
    }
}
```

---

### 5. ContentService.java

```java
// 변경 전
@Service
public class DiaryService {
    private final DiaryRepository diaryRepository;

    public DiaryResponse createDiary(DiaryRequest request, String email) {
        Diary diary = new Diary();
        diary.setContent(request.getContent());
        Diary saved = diaryRepository.save(diary);
        return new DiaryResponse(saved);
    }
}

// 변경 후
@Service
public class ContentService {
    private final ContentRepository contentRepository;

    public ContentResponse createContent(ContentRequest request, String email) {
        Content content = new Content();
        content.setContentText(request.getContentText());
        Content saved = contentRepository.save(content);
        return new ContentResponse(saved);
    }
}
```

---

### 6. ContentController.java

```java
// 변경 전
@RestController
@RequestMapping("/api/diary")
public class DiaryController {
    private final DiaryService diaryService;

    @PostMapping("/write")
    public ApiResponse<DiaryResponse> createDiary(@RequestBody DiaryRequest request) {
        DiaryResponse response = diaryService.createDiary(request, userDetails.getUserEmail());
        return ApiResponse.success(response);
    }

    @DeleteMapping("/delete-all")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<Void> deleteAllDiaries() {
        diaryService.deleteAll();
        return ApiResponse.success(null);
    }
}

// 변경 후
@RestController
@RequestMapping("/api/content")  // /api/diary → /api/content
public class ContentController {
    private final ContentService contentService;

    @PostMapping("/write")
    public ApiResponse<ContentResponse> createContent(@RequestBody ContentRequest request) {
        ContentResponse response = contentService.createContent(request, userDetails.getUserEmail());
        return ApiResponse.success(response);
    }

    @DeleteMapping("/delete-all")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<Void> deleteAllContents() {
        contentService.deleteAll();
        return ApiResponse.success(null);
    }
}
```

---

## 📊 작업 순서 (중요!)

### 1단계: 새 패키지 및 파일 생성 (30분)
```bash
mkdir -p src/main/java/com/domain/demo_backend/domain/content/{controller,service,domain,dto}

# 파일 복사 후 리네임
cp domain/diary/domain/Diary.java domain/content/domain/Content.java
# (6개 파일 모두 복사)
```

### 2단계: 클래스명 및 내용 수정 (1시간)
- Content.java 수정 (테이블명, 컬럼명)
- ContentRepository.java 수정
- ContentRequest/Response.java 수정
- ContentService.java 수정
- ContentController.java 수정

### 3단계: Import 문 정리 (15분)
- IntelliJ IDEA: `Ctrl + Alt + O` (Optimize Imports)
- 모든 `import ...diary...` → `import ...content...`

### 4단계: 컴파일 확인 (15분)
```bash
./gradlew clean build
```

### 5단계: 기존 diary 패키지 삭제 (5분)
```bash
rm -rf src/main/java/com/domain/demo_backend/domain/diary
```

### 6단계: 테스트 실행 (15분)
```bash
./gradlew test
```

---

## ⚠️ 주의사항

### 1. 컬럼명 충돌 주의
- `content` 테이블의 `content` 컬럼 → `content_text`로 변경 필수
- SQL 예약어 충돌 방지

### 2. FK 제약조건
- `Diary` 엔티티의 `@ManyToOne` 관계가 있다면 `Content`로 변경

### 3. 로깅
- 로그 메시지 내 "diary" → "content"로 변경

---

## 📌 예상 소요 시간

| 작업 | 예상 시간 |
|-----|----------|
| 파일 생성 및 복사 | 30분 |
| 클래스명/내용 수정 | 1시간 |
| Import 정리 | 15분 |
| 컴파일 확인 | 15분 |
| diary 패키지 삭제 | 5분 |
| 테스트 실행 | 15분 |
| **총 예상 시간** | **2시간 20분** |
