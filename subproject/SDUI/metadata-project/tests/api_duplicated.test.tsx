import React from 'react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { renderWithProviders } from './test-utils';
import { MetadataProvider } from '../components/providers/MetadataProvider';
import { waitFor } from '@testing-library/react';
// @@@@  실제 상수가 정의된 파일에서 SCREEN_MAP을 불러온다.
import { SCREEN_MAP } from '../components/constants/screenMap';
import { logTestSuccess } from './TestLogger';

// 중복 제거된 screenId 목록 생성
const uniqueScreenIds = Array.from(new Set(Object.values(SCREEN_MAP)));

const server = setupServer();
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// @@@@  test.each를 통해 모든 화면을 자동으로 반복 테스트한다.
test.each(uniqueScreenIds)('%s 화면의 메타데이터 중복 호출 검증', async (screenId) => {
    let callCount = 0;

    server.use(
        http.get(`/api/ui/${screenId}`, () => {
            callCount++;
            return HttpResponse.json({ success: true, data: [] });
        })
    );

    renderWithProviders(
        <MetadataProvider screenId={screenId}>
            <div>테스트 대상: {screenId}</div>
        </MetadataProvider>
    );

    await waitFor(() => {
        if (callCount !== 1) throw new Error(`${screenId} 호출 횟수: ${callCount}`);
    });

    // @@@@ 테스트 성공 시 파일로 기록을 남긴다.
    logTestSuccess(screenId);
});