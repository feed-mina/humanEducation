import { useCallback, useRef } from 'react';

interface UseSSEStreamV2Options {
    onChunk: (chunk: string) => void;
    onDone?: () => void;
    onError?: (message: string) => void;
}

/**
 * useSSEStreamV2 — V2 테스트용 훅
 * 엔드포인트: /api/ai/v2/chat/stream
 * 디버그 로깅 추가 (콘솔에서 확인 가능)
 */
export function useSSEStreamV2({ onChunk, onDone, onError }: UseSSEStreamV2Options) {
    const abortRef = useRef<AbortController | null>(null);

    const stream = useCallback(async (url: string, body: object) => {
        abortRef.current?.abort();
        const controller = new AbortController();
        abortRef.current = controller;

        console.log('[V2 SSE] 스트리밍 시작:', url, body);

        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(body),
                signal: controller.signal,
            });

            console.log('[V2 SSE] 응답 수신 - status:', res.status, 'ok:', res.ok);

            if (!res.ok || !res.body) {
                console.error('[V2 SSE] 연결 실패:', res.status);
                onError?.('서버 연결에 실패했습니다.');
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let chunkCount = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() ?? '';

                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;

                    // 'data: ' 또는 'data:' 뒤의 실제 데이터 추출
                    const data = line.startsWith('data: ') 
                        ? line.slice(6).trim() 
                        : line.slice(5).trim();

                    console.log('[V2 SSE] data 라인 인식됨:', data.substring(0, 80));

                    if (data === '[DONE]') {
                        console.log('[V2 SSE] 완료. 총 청크:', chunkCount);
                        onDone?.();
                        return;
                    }

                    try {
                        let parsed = JSON.parse(data);
                        // 만약 서버에서 JSON을 중복으로 따옴표 감싸 보냈을 경우를 대비한 방어 로직
                        if (typeof parsed === 'string') {
                            parsed = JSON.parse(parsed);
                        }

                        if (parsed.error) {
                            console.error('[V2 SSE] 서버 에러:', parsed.error);
                            onError?.(parsed.error);
                            return;
                        }

                        // content가 undefined가 아니면 (빈 문자열이어도) 처리
                        if (parsed.content !== undefined && parsed.content !== null) {
                            chunkCount++;
                            console.log(`[V2 SSE] 청크 #${chunkCount}:`, parsed.content);
                            onChunk(parsed.content);
                        }
                    } catch {
                        console.warn('[V2 SSE] JSON 파싱 실패 (무시):', data);
                    }
                }
            }
            onDone?.();
        } catch (err: any) {
            if (err?.name !== 'AbortError') {
                console.error('[V2 SSE] 스트리밍 오류:', err);
                onError?.('스트리밍 중 오류가 발생했습니다.');
            }
        }
    }, [onChunk, onDone, onError]);

    const abort = useCallback(() => {
        abortRef.current?.abort();
    }, []);

    return { stream, abort };
}
