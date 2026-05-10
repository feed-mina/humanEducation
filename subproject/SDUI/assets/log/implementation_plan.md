# Implementation Plan - AI Chat UI & Backend Fixes

This plan addresses the `AccessDeniedException` during AI response streaming and the incorrect STT transcription (phonetic English instead of Korean).

## Proposed Changes

### Backend

#### [MODIFY] [SecurityConfig.java](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/SDUI-server/src/main/java/com/domain/demo_backend/global/config/SecurityConfig.java)
- Allow `DispatcherType.ASYNC` in `authorizeHttpRequests` to prevent security filters from blocking internal SSE dispatches.

#### [MODIFY] [AiSttController.java](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/controller/AiSttController.java)
- Ensure logging is clear about received language.

### Frontend

#### [MODIFY] [AIChatComponent.tsx](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/metadata-project/components/fields/AIChatComponent.tsx)
- Pass the `language` state (e.g., 'ko' or 'en') to the `/api/ai/stt` request.
- This will fix the issue where Korean speech was transcribed as "annyeonghaseyo" (phonetic English) because the backend defaulted to 'en'.

#### [MODIFY] [AI_CHAT.css](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/metadata-project/app/styles/AI_CHAT.css)
- Subtle tweaks to ensure the conversation panel and robot icons are well-aligned during streaming.

## Verification Plan

### Automated Tests
- N/A (Manual verification is more effective for SSE and Voice interaction in this local environment).

### Manual Verification
1. **SSE Security Check**:
   - Open the AI Chat page.
   - Send a voice message.
   - Verify that the robot's response starts streaming (text appears incrementally) without 403 errors in the log.
2. **STT Language Check**:
   - Speak in Korean.
   - Click "답변완료".
   - Verify that the transcribed text is in Korean (e.g., "안녕하세요") instead of phonetic English ("annyeonghaseyo").
3. **UI Interaction**:
   - Verify that the "Stop" (ㅁ) button and "Submit" (답변완료) button work as expected to end recording and trigger response.
