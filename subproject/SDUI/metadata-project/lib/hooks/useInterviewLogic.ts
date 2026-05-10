import { useState, useCallback, useRef, useEffect } from 'react';
import { ChatMessage } from '@/lib/types/ai';
import { useSSEStreamV2 } from '@/lib/hooks/useSSEStreamV2';
import { useAudioRecorder } from '@/lib/hooks/useAudioRecorder';
import api from '@/services/axios';

interface UseInterviewLogicProps {
    language: string;
    resumeText: string;
    resumeFileKey?: string | null;
    systemPromptTemplate?: string | null;
    onError?: (msg: string) => void;
}

/**
 * useInterviewLogic
 * ──
 * AI 면접 전용 훅.
 * - POST /api/ai/interview/start  → 이력서 분석 + 첫 질문 SSE 스트리밍
 * - POST /api/ai/interview/answer → 답변 제출 + 꼬리 질문 SSE 스트리밍
 * - 클라이언트가 대화 이력(history)을 유지해 매 요청마다 전달 (서버 세션 불필요)
 */
export function useInterviewLogic({ language, resumeText, resumeFileKey, systemPromptTemplate, onError }: UseInterviewLogicProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

    // 클로저 내에서 최신 messages를 참조하기 위한 ref
    const messagesRef = useRef<ChatMessage[]>([]);
    useEffect(() => { messagesRef.current = messages; }, [messages]);

    const { stream, abort } = useSSEStreamV2({
        onChunk: (chunk) => {
            setMessages(prev => {
                const last = prev[prev.length - 1];
                if (last?.role === 'assistant') {
                    return [...prev.slice(0, -1), { ...last, content: last.content + chunk }];
                }
                return [...prev, { role: 'assistant', content: chunk }];
            });
        },
        onDone: () => setIsStreaming(false),
        onError: (err) => {
            setIsStreaming(false);
            onError?.(err || '면접 서버 연결에 실패했습니다.');
        },
    });

    /**
     * 면접 시작: 이력서 분석 → 첫 질문 스트리밍
     */
    const startInterview = useCallback(async () => {
        setIsStreaming(true);
        setMessages([{ role: 'assistant', content: '' }]);
        await stream('/api/ai/interview/start', {
            resumeText,
            resumeFileKey: resumeFileKey || undefined,
            systemPromptTemplate: systemPromptTemplate || undefined,
            language,
        });
    }, [stream, resumeText, resumeFileKey, systemPromptTemplate, language]);

    /**
     * 답변 제출: STT → /api/ai/interview/answer → 꼬리 질문 스트리밍
     */
    const { state: recordingState, startRecording, stopRecording, cancelRecording, resetState } = useAudioRecorder({
        onAudioReady: async (blob) => {
            try {
                const formData = new FormData();
                formData.append('audio', blob, 'recording.webm');

                const res = await api.post('/api/ai/stt', formData, {
                    params: { language },
                    headers: { 'Content-Type': 'multipart/form-data' },
                });

                const answerText: string = res.data?.data?.text || '';
                if (!answerText.trim()) {
                    resetState();
                    return;
                }

                const snapshot = messagesRef.current;
                const userMsg: ChatMessage = {
                    role: 'user',
                    content: answerText,
                    audioUrl: URL.createObjectURL(blob),
                };

                const updatedMsgs = [...snapshot, userMsg];
                setMessages(updatedMsgs);
                resetState();

                // 어시스턴트 빈 버블 추가 후 스트리밍
                setIsStreaming(true);
                setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

                await stream('/api/ai/interview/answer', {
                    answerText,
                    history: snapshot.map(({ role, content }) => ({ role, content })),
                    resumeText,
                    systemPromptTemplate: systemPromptTemplate || undefined,
                    language,
                });
            } catch (err) {
                console.error('[useInterviewLogic] STT 처리 실패:', err);
                resetState();
            }
        },
        onAnalyser: setAnalyser,
    });

    const handleStartRecording = useCallback((_mode: string) => {
        // 면접은 단일 언어 모드 — mode 파라미터 무시
        startRecording();
    }, [startRecording]);

    const handleEndInterview = useCallback(() => {
        abort();
        setMessages([]);
    }, [abort]);

    return {
        messages,
        isStreaming,
        recordingState,
        analyser,
        startInterview,
        handleStartRecording,
        stopRecording,
        cancelRecording,
        handleEndInterview,
    };
}
