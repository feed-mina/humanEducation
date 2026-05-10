// tests/CustomReporter.js
const fs = require('fs');
const path = require('path');

class CustomReporter {
    constructor(globalConfig, options) {
        this._globalConfig = globalConfig;
        this._options = options;
    }

    // 모든 테스트가 종료되었을 때 실행되는 훅(Hook)
    onRunComplete(contexts, results) {
        const dirPath = path.join(process.cwd(), 'tests/logs');
        const filePath = path.join(dirPath, 'frontend-summary.log');

        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }

        //   실행된 테스트 파일 목록 추출 [cite: 2026-01-26]
        const testFiles = results.testResults
            .map(test => `- ${path.relative(process.cwd(), test.testFilePath)}`)
            .join('\n');
        const timestamp = new Date().toLocaleString();
        const duration = ((Date.now() - results.startTime) / 1000).toFixed(3);

        // 네가 원했던 요약 정보 구성 [cite: 2026-02-20]
        const summaryMessage = `
[${timestamp}] JEST TEST SUMMARY
---------------------------------
Executed Test Files:
${testFiles}
Test Suites: ${results.numPassedTestSuites} passed, ${results.numFailedTestSuites} failed, ${results.numTotalTestSuites} total
Tests:       ${results.numPassedTests} passed, ${results.numFailedTests} failed, ${results.numTotalTests} total
Snapshots:   ${results.numTotalSnapshots} total
Time:        ${duration} s
---------------------------------
\n`;

        fs.appendFileSync(filePath, summaryMessage);
        console.log(`\n✔ 테스트 요약 정보가 저장되었습니다: ${filePath}`);
    }
}

module.exports = CustomReporter;