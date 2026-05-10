/**
 * 객체 내의 jsonb 필드를 찾아 자동으로 파싱하는 함수
 * @param data 원본 데이터 객체
 * @returns 파싱이 완료된 새로운 객체
 */
export const parseJsonbFields = (data: any): any => {
    if (!data || typeof data !== 'object') return data;

    const processed = { ...data };

    Object.keys(processed).forEach((key) => {
        const field = processed[key];

        // 백엔드에서 내려주는 jsonb 특유의 구조인지 확인
        if (field && typeof field === 'object' && field.type === 'jsonb') {
            if (typeof field.value === 'string' && field.value.trim() !== '') {
                try {
                    // JSON 문자열을 실제 객체/배열로 변환
                    processed[key] = JSON.parse(field.value);
                } catch (e) {
                    // console.error(`[JSONB 파싱 에러] 필드: ${key}, 값: ${field.value}`, e);
                    processed[key] = field.value; // 실패 시 문자열 유지
                }
            } else {
                processed[key] = field.null ? null : (field.value || null);
            }
        }
    });

    return processed;
};