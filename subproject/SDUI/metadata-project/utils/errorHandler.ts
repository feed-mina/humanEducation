/**
 * 에러 객체에서 사용자에게 표시할 메시지를 추출합니다.
 * @param error - catch된 에러 객체
 * @param defaultMessage - 기본 메시지 (선택)
 * @returns 사용자에게 표시할 문자열 메시지
 */
export const extractErrorMessage = (error: any, defaultMessage: string = '요청 처리 중 오류가 발생했습니다'): string => {
    // 1. Axios 에러 응답 처리
    if (error?.response?.data) {
        const data = error.response.data;

        // 백엔드에서 message 필드를 보낸 경우
        if (typeof data.message === 'string') {
            return data.message;
        }

        // 백엔드에서 error 필드를 보낸 경우
        if (typeof data.error === 'string') {
            return data.error;
        }

        // data 자체가 문자열인 경우
        if (typeof data === 'string') {
            return data;
        }
    }

    // 2. Error 객체의 message 속성
    if (error?.message && typeof error.message === 'string') {
        return error.message;
    }

    // 3. 문자열이 직접 전달된 경우
    if (typeof error === 'string') {
        return error;
    }

    // 4. 기본 메시지 반환
    return defaultMessage;
};

/**
 * 에러를 콘솔에 로깅하고 사용자에게 alert을 표시합니다.
 * @param error - catch된 에러 객체
 * @param context - 에러 발생 위치/컨텍스트 (디버깅용)
 * @param defaultMessage - 기본 메시지
 */
export const handleError = (error: any, context: string, defaultMessage?: string): void => {
    const message = extractErrorMessage(error, defaultMessage);

    // 개발 환경에서는 콘솔에 상세 에러 출력
    if (process.env.NODE_ENV === 'development') {
        console.error(`[${context}] Error:`, error);
        console.error(`[${context}] Error Response:`, error?.response);
    }

    // 사용자에게 alert 표시
    alert(message);
};

/**
 * HTTP 상태 코드별 기본 메시지를 반환합니다.
 * @param statusCode - HTTP 상태 코드
 * @returns 상태 코드에 맞는 기본 메시지
 */
export const getStatusMessage = (statusCode?: number): string => {
    if (!statusCode) return '요청 처리 중 오류가 발생했습니다';

    const statusMessages: Record<number, string> = {
        400: '잘못된 요청입니다',
        401: '인증이 필요합니다',
        403: '접근 권한이 없습니다',
        404: '요청한 리소스를 찾을 수 없습니다',
        409: '중복된 데이터입니다',
        422: '입력값을 확인해주세요',
        500: '서버 오류가 발생했습니다',
        502: '게이트웨이 오류가 발생했습니다',
        503: '서비스를 일시적으로 사용할 수 없습니다',
    };

    return statusMessages[statusCode] || `오류가 발생했습니다 (${statusCode})`;
};

/**
 * 에러를 분석하여 상세 정보를 반환합니다.
 * @param error - catch된 에러 객체
 * @returns 에러 상세 정보 객체
 */
export const analyzeError = (error: any) => {
    return {
        message: extractErrorMessage(error),
        statusCode: error?.response?.status,
        statusText: error?.response?.statusText,
        defaultMessage: getStatusMessage(error?.response?.status),
        data: error?.response?.data,
        isNetworkError: !error?.response && error?.request,
        isTimeout: error?.code === 'ECONNABORTED',
    };
};
