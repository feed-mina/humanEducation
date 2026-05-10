// tests/TestLogger.ts
import fs from 'fs';
import path from 'path';

export const logTestSuccess = (screenId: string) => {
    const dirPath = path.join(process.cwd(), 'tests/logs');
    const filePath = path.join(dirPath, 'frontend-success.log');

    // 1. 폴더가 없으면 생성
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }

    // 2. 로그 내용 작성 (날짜, 화면ID, 상태)
    const timestamp = new Date().toLocaleString();
    const logMessage = `[${timestamp}] PASS: ${screenId} - 메타데이터 API 호출 1회 확인 완료\n`;

    // 3. 파일 끝에 추가 (Append)
    fs.appendFileSync(filePath, logMessage);
};