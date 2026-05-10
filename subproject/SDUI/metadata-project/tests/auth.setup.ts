import { test as setup, expect } from '@playwright/test';

const authFile = 'playwright/.auth/user.json';

setup('authenticate', async ({ page }) => {
    // 1. 로그인 페이지 이동
    await page.goto('/view/LOGIN_PAGE');

    // 2. 로그인 수행 (이메일/ID + 도메인 직접 입력)
    await page.getByPlaceholder(/ID/i).fill('user1');
    
    // '직접 입력' 선택하여 도메인 입력필드 활성화
    await page.locator('select').selectOption('직접 입력');
    // 코드상 오타(domian-input)에 맞춰 셀렉터 지정
    await page.locator('.domian-input').fill('test.com');

    await page.getByPlaceholder(/Password/i).fill('Test1234!');
    
    // 메인 컨텐츠 영역의 로그인 버튼을 명시적으로 클릭
    await page.locator('main').getByRole('button', { name: '로그인' }).click();

    // 3. 로그인 완료 대기 (URL 변화: 메인 페이지로 이동 확인)
    await page.waitForURL('**/MAIN_PAGE', { timeout: 15000 });

    // 4. 인증 상태(쿠키, 로컬스토리지)를 파일로 저장
    await page.context().storageState({ path: authFile });
});