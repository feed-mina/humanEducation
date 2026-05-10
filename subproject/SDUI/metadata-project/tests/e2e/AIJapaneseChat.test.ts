import { test, expect } from '@playwright/test';

test.describe('AI Japanese Chat E2E Integration', () => {
    test('should load the AI Japanese Chat page and show cherry blossom theme', async ({ page }) => {
        await page.goto('/view/AI_JAPANESE_CHAT_PAGE');

        // Verify intro title
        const introTitle = page.locator('.ai-intro-title');
        await expect(introTitle).toBeVisible({ timeout: 10000 });
        await expect(introTitle).toContainText(/AI Japanese Tutor/i);

        // Verify start button
        const startBtn = page.getByRole('button', { name: /대화 시작하기|Start/i });
        await expect(startBtn).toBeVisible();
        await startBtn.click();

        // Verify theme class application (cssClass: ai-japanese-theme)
        const container = page.locator('.ai-chat-standalone-container');
        await expect(container).toHaveClass(/ai-japanese-theme/);
    });
});
