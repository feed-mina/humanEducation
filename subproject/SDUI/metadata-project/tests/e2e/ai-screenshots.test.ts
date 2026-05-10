import { test } from '@playwright/test';
import path from 'path';

const SCREENSHOT_DIR = path.resolve(__dirname, '../../../../assets/log/3월17일');

test.describe('AI Pages 현재 UI 스크린샷', () => {

    test('AI_JAPANESE_CHAT_PAGE — 초기 화면', async ({ page }) => {
        await page.goto('/view/AI_JAPANESE_CHAT_PAGE');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1500);
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'AI_JAPANESE_intro.png'),
            fullPage: true,
        });
    });

    test('AI_ENGLISH_CHAT_PAGE — 초기 화면', async ({ page }) => {
        await page.goto('/view/AI_ENGLISH_CHAT_PAGE');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1500);
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'AI_ENGLISH_intro.png'),
            fullPage: true,
        });
    });

    test('AI_INTERVIEW_PAGE — 초기 화면', async ({ page }) => {
        await page.goto('/view/AI_INTERVIEW_PAGE');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1500);
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'AI_INTERVIEW_intro.png'),
            fullPage: true,
        });
    });

    test('AI_ENGLISH_CHAT_PAGE — 레코더 바 (녹음 중 시뮬)', async ({ page }) => {
        await page.goto('/view/AI_ENGLISH_CHAT_PAGE');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1500);

        // 인트로 → 채팅 시작
        const startBtn = page.locator('.ai-start-btn');
        if (await startBtn.isVisible()) {
            await startBtn.click();
            await page.waitForTimeout(2000);
        }

        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'AI_ENGLISH_recorder_bar.png'),
            fullPage: true,
        });
    });

    test('AI_INTERVIEW_PAGE — 레코더 바 (이력서 입력 후)', async ({ page }) => {
        await page.goto('/view/AI_INTERVIEW_PAGE');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1500);

        const textarea = page.locator('.ai-resume-textarea');
        if (await textarea.isVisible()) {
            await textarea.fill('이름: 홍길동, 경력: React 3년, Spring Boot 2년');
        }

        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'AI_INTERVIEW_with_resume.png'),
            fullPage: true,
        });
    });
});
