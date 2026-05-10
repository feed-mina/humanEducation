import api from './axios';

/**
 * AI 관련 API 호출 전담 서비스 모듈
 * - useAIChatLogic 등 훅에서 직접 axios를 호출하지 않고 이 모듈만 호출
 * - 엔드포인트 변경 시 이 파일 하나만 수정
 * - 테스트 시 이 모듈만 모킹하면 됨
 */

const STT_ENDPOINT = '/api/ai/stt';
const CHAT_ENDPOINT = '/api/ai/v2/chat/stream';
const TRANSLATE_ENDPOINT = '/api/ai/v2/chat/translate';
const PRONUNCIATION_ENDPOINT = '/api/ai/pronunciation';

export const aiService = {
    /**
     * 오디오 Blob을 Whisper STT로 변환하여 텍스트 반환
     * @param blob 녹음된 오디오
     * @param language STT 언어 코드 (예: 'en', 'ko', 'ja')
     * @returns 변환된 텍스트 (빈 문자열 가능)
     */
    async stt(blob: Blob, language: string): Promise<string> {
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');
        const res = await api.post(STT_ENDPOINT, formData, {
            params: { language },
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return res.data?.data?.text || '';
    },

    /**
     * 텍스트를 지정된 언어로 번역
     * @param text 원문 텍스트
     * @param target 목표 언어 코드 (예: 'en', 'ko')
     * @returns 번역된 텍스트
     */
    async translate(text: string, target: string): Promise<string> {
        const res = await api.post(TRANSLATE_ENDPOINT, { text, target });
        return res.data?.data || text;
    },

    /**
     * GPT로 사용자 발화의 표현 품질 평가
     * @param spoken   STT로 변환된 사용자 발화 텍스트
     * @param language 언어 코드 (en, ja)
     * @returns { score: 0~100, feedback: string, idealExpression: string }
     */
    async checkPronunciation(
        spoken: string,
        language: string,
    ): Promise<{ score: number; feedback: string; idealExpression?: string }> {
        const res = await api.post(PRONUNCIATION_ENDPOINT, { spoken, language });
        return res.data?.data || { score: 0, feedback: '' };
    },
};

export { STT_ENDPOINT, CHAT_ENDPOINT, TRANSLATE_ENDPOINT, PRONUNCIATION_ENDPOINT };
