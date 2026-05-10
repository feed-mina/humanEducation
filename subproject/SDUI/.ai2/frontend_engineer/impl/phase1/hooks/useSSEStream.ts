// DESTINATION: metadata-project/lib/hooks/useSSEStream.ts
import { useCallback, useRef } from 'react';

interface UseSSEStreamOptions {
    onChunk: (chunk: string) => void;
    onDone?: () => void;
    onError?: (message: string) => void;
}

export function useSSEStream({ onChunk, onDone, onError }: UseSSEStreamOptions) {
    const abortRef = useRef<AbortController | null>(null);

    const stream = useCallback(async (url: string, body: object) => {
        // 기존 스트림 중단
        abortRef.current?.abort();
        const controller = new AbortController();
        abortRef.current = controller;

        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include', // 쿠키(JWT) 전달
                body: JSON.stringify(body),
                signal: controller.signal,
            });

            if (!res.ok || !res.body) {
                onError?.('서버 연결에 실패했습니다.');
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() ?? '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;

                    const data = line.slice(6).trim();
                    if (data === '[DONE]') {
                        onDone?.();
                        return;
                    }

                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.error) {
                            onError?.(parsed.error);
                            return;
                        }
                        if (parsed.content) {
                            onChunk(parsed.content);
                        }
                    } catch {
                        // non-JSON SSE 라인 무시
                    }
                }
            }
            onDone?.();
        } catch (err: any) {
            if (err?.name !== 'AbortError') {
                onError?.('스트리밍 중 오류가 발생했습니다.');
            }
        }
    }, [onChunk, onDone, onError]);

    const abort = useCallback(() => {
        abortRef.current?.abort();
    }, []);

    return { stream, abort };
}
