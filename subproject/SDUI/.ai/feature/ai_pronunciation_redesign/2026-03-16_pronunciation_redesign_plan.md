# Pronunciation Feature 재설계 계획

**날짜**: 2026-03-16
**브랜치**: feature/addAIChore

---

## 배경 및 문제점

### 기존 구현의 문제
| 항목 | 기존 | 문제점 |
|------|------|--------|
| 엔진 | FastAPI `SequenceMatcher` (텍스트 유사도) | 사용자 발화와 AI 응답을 비교 → 내용이 다르면 항상 0점 |
| 기준 텍스트 | AI 직전 응답 전체 | 사용자가 다른 말을 하면 기준이 무의미 |
| 평가 대상 | 텍스트 문자열 유사도 | 실제 발음·표현 품질 미반영 |
| 트리거 | 번역 안 된 경우 전체 | 한국어 번역 모드도 포함 |

**근본 원인**: "발음 채점"이 아닌 "텍스트 복사 정확도"를 측정하고 있었음

---

## 새 설계

### 요구사항 (사용자 선택)
- **평가 대상**: 발음 정확도 + 영어 표현/문법 교정 (둘 다)
- **기준 텍스트**: GPT가 사용자 발화 기반으로 이상적 표현 생성 후 비교
- **결과 표시**: 별도 점수 카드 (현재 방식 유지)
- **트리거 모드**: General Mic 모드에서만 (한국어 번역 모드 제외)

### 새 흐름

```
사용자 General Mic 발화
  → STT (Whisper) → spoken 텍스트
  → [wasTranslated = false → 평가 실행]
  → POST /api/ai/pronunciation { spoken, language }
  → Spring PronunciationService
      → OpenAiClientV2.evaluateExpression(spoken, language)
      → GPT-4o: "평가 + 이상적 표현 생성"
      → { score, feedback, idealExpression }
  → 프론트 pronunciationData 저장
  → ConversationPanelV2 점수 카드에 표시
```

---

## 변경 파일 목록

### 백엔드 (Spring Boot)

#### 1. `SDUI-server/src/main/java/.../client/OpenAiClientV2.java`
- `evaluateExpression(String spoken, String language)` 메서드 추가
- Chat Completions API (non-streaming) 호출
- GPT 프롬프트:
  ```
  User said the following in [Language]: "[spoken]"
  Evaluate their expression quality. Respond ONLY in JSON:
  {"score": <0-100>, "feedback": "<one sentence>", "idealExpression": "<natural corrected version>"}
  ```
- `response_format: { type: "json_object" }` 사용으로 순수 JSON 보장

#### 2. `SDUI-server/src/main/java/.../dto/PronunciationResponse.java`
- `idealExpression` 필드 추가

#### 3. `SDUI-server/src/main/java/.../service/PronunciationService.java`
- FastAPI 프록시 코드 전체 제거
- `OpenAiClientV2` 주입 (`@RequiredArgsConstructor`)
- `evaluateExpression` 호출로 대체

### 프론트엔드 (Next.js)

#### 4. `metadata-project/lib/types/ai.ts`
- `ChatMessage`에 `pronunciationIdeal?: string` 추가
- `pronunciationExpected` 제거 (더 이상 사용 안 함)

#### 5. `metadata-project/services/aiService.ts`
- `checkPronunciation(spoken, expected, language)` → `checkPronunciation(spoken, language)`
- `expected` 파라미터 제거

#### 6. `metadata-project/lib/hooks/useAIChatLogic.ts`
- `lastAssistantMsg` 조회 로직 제거
- `aiService.checkPronunciation(originalTranscript, language)` 로 단순화
- `pronunciationExpected` → `pronunciationIdeal` 로 변경

#### 7. `metadata-project/components/fields/ai/ConversationPanelV2.tsx`
- 점수 카드 내 "기준" → "추천 표현"으로 레이블 변경
- `pronunciationExpected` → `pronunciationIdeal` 표시
- 텍스트 라벨: `내 발음` → `내 표현`

---

## FastAPI 변경 없음

`pronounce-api/app/main.py`의 `/pronunciation-score` 엔드포인트는 유지.
Spring Boot가 FastAPI를 더 이상 호출하지 않으므로 사실상 비활성화되지만,
추후 다른 용도로 재사용 가능하도록 삭제하지 않음.

---

## 점수 해석 기준 (변경 없음)

| 점수 | 레벨 | 의미 |
|------|------|------|
| 85~100 | excellent | 매우 자연스러운 영어 표현 |
| 65~84 | good | 작은 개선 필요 |
| 45~64 | fair | 연습 필요 |
| 0~44 | poor | 기본 표현 개선 필요 |
