import { aiService, PRONUNCIATION_ENDPOINT } from '@/services/aiService';
import api from '@/services/axios';

jest.mock('@/services/axios');

describe('aiService.checkPronunciation', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('spoken과 language를 body에 담아 pronunciation 엔드포인트를 호출해야 함', async () => {
        (api.post as jest.Mock).mockResolvedValue({
            data: {
                data: {
                    score: 85,
                    feedback: 'Great expression!',
                    idealExpression: 'What a beautiful day it is!',
                },
            },
        });

        const result = await aiService.checkPronunciation("It's a beautiful day", 'en');

        expect(api.post).toHaveBeenCalledTimes(1);
        expect(api.post).toHaveBeenCalledWith(PRONUNCIATION_ENDPOINT, {
            spoken: "It's a beautiful day",
            language: 'en',
        });

        expect(result.score).toBe(85);
        expect(result.feedback).toBe('Great expression!');
        expect(result.idealExpression).toBe('What a beautiful day it is!');
    });

    it('일본어 language 코드로 호출할 수 있어야 함', async () => {
        (api.post as jest.Mock).mockResolvedValue({
            data: {
                data: {
                    score: 90,
                    feedback: '自然な表現です',
                    idealExpression: '今日は良いお天気ですね',
                },
            },
        });

        const result = await aiService.checkPronunciation('今日はいい天気ですね', 'ja');

        expect(api.post).toHaveBeenCalledWith(PRONUNCIATION_ENDPOINT, {
            spoken: '今日はいい天気ですね',
            language: 'ja',
        });
        expect(result.score).toBe(90);
        expect(result.idealExpression).toBe('今日は良いお天気ですね');
    });

    it('응답 data가 null이면 기본값 { score: 0, feedback: "" }을 반환해야 함', async () => {
        (api.post as jest.Mock).mockResolvedValue({ data: null });

        const result = await aiService.checkPronunciation('test', 'en');

        expect(result.score).toBe(0);
        expect(result.feedback).toBe('');
        expect(result.idealExpression).toBeUndefined();
    });

    it('PRONUNCIATION_ENDPOINT 상수가 /api/ai/pronunciation이어야 함', () => {
        expect(PRONUNCIATION_ENDPOINT).toBe('/api/ai/pronunciation');
    });
});
