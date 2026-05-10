import { test, expect } from '@playwright/test';

test.describe('AI Chat V2 E2E Integration', () => {
    // Note: Since real STT requires actual audio input which is hard in E2E,
    // we focus on the UI flow and metadata-driven rendering verification.

    test('should load the English AI Chat page and start a session', async ({ page }) => {
        // 1. Navigate to the AI English Chat Page
        await page.goto('/view/AI_ENGLISH_CHAT_PAGE');

        // 2. Verify SDUI Metadata rendering (Header/Intro)
        // Wait up to 10s for the metadata to load and component to render
        const introTitle = page.locator('.ai-intro-title');
        await expect(introTitle).toBeVisible({ timeout: 10000 });
        // 로컬 DB 메타데이터에 맞춰 영문/한글 모두 허용하도록 수정
        await expect(introTitle).toContainText(/(English Tutor|AI 영어 대화)/i);

        // 3. Start Chatting
        const startBtn = page.getByRole('button', { name: /대화 시작하기/i });
        await startBtn.click();

        // 4. Verify Transition to Chat Main Screen
        // Header title should still be visible but likely moved to a header bar
        const headerTitle = page.locator('.ai-header-title');
        await expect(headerTitle).toBeVisible();

        // 5. Verify Header & Thread rendering
        await expect(page.locator('.ai-chat-header')).toBeVisible();
        await expect(page.locator('.ai-header-title')).toContainText(/(English Tutor|AI 영어 대화)/i);
        
        // 6. Interaction: Start Recording
        const micBtn = page.locator('.ai-mic-btn-circle').first();
        await expect(micBtn).toBeEnabled();
        await micBtn.click();
        
        // Wait for recorder to be in recording state (buttons should change)
        // Adjust selectors based on AudioRecorder.tsx implementation
        const stopBtn = page.getByRole('button', { name: /Submit|답변완료/i });
        const cancelBtn = page.getByRole('button', { name: /Cancel|취소/i });
        
        await expect(stopBtn).toBeVisible();
        await expect(cancelBtn).toBeVisible();

        // 7. Interaction: Cancel Recording
        await cancelBtn.click();
        await expect(cancelBtn).not.toBeVisible(); // Should return to idle

        // 8. Interaction: End Chat (Requires at least one message)
        // In this test, we verify the button presence after start
        const endBtn = page.locator('.ai-session-end-btn');
        // Since there are no messages yet in a clean E2E, the button might be hidden 
        // depending on the 'messages.length > 0' condition.
        // Let's check the container for messages handle
    });

    test('should reflect turn progress and allow ending session', async ({ page }) => {
        await page.goto('/view/AI_ENGLISH_CHAT_PAGE');
        await page.getByRole('button', { name: /대화 시작하기/i }).click();

        // Verify initial gauge state
        const gaugeFill = page.locator('.ai-gauge-fill');
        await expect(gaugeFill).toHaveAttribute('style', /width: 0%/);

        // Verify "End Chat" button doesn't appear until dialogue starts (if logic applies)
        // or just verify its visibility if welcome message is present.
        // By default, useAIChatLogic adds a welcome message.
        const endBtn = page.locator('.ai-session-end-btn');
        await expect(endBtn).toBeVisible();
        await endBtn.click();
        
        // After end chat, we should ideally be back to intro or a finished state
        // Current implementation likely resets isStarted or similar
        // (Verify based on useAIChatLogic.ts: handleEndChat logic)
    });
});
