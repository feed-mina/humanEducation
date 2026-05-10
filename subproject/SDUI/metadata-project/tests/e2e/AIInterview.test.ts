import { test, expect } from '@playwright/test';

test.describe('AI Interview E2E Integration', () => {
    test('should load the AI Interview page and show professional intro', async ({ page }) => {
        await page.goto('/view/AI_INTERVIEW_PAGE');

        // Verify professional intro elements
        const introTitle = page.locator('.ai-intro-title');
        await expect(introTitle).toBeVisible({ timeout: 10000 });
        await expect(introTitle).toContainText(/AI 면접관/);

        // Fill in resume text (button is disabled without resume)
        const resumeTextarea = page.locator('.ai-resume-textarea');
        await resumeTextarea.fill('이름: 홍길동, 경력: 3년, 기술: React, Spring Boot');

        const startBtn = page.getByRole('button', { name: '면접 시작하기' });
        await expect(startBtn).toBeVisible();
        await startBtn.click();

        // Verify transition to main interview panel
        const headerTitle = page.locator('.ai-header-title');
        await expect(headerTitle).toContainText(/AI 면접관/);
    });
});
