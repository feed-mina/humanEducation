// DESTINATION: metadata-project/components/fields/AIChatComponent.tsx
'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AIChatConfig, ChatMessage } from '@/lib/types/ai';
import { useSSEStream } from '@/lib/hooks/useSSEStream';
import { useAudioRecorder } from '@/lib/hooks/useAudioRecorder';
import ConversationPanel from '@/components/fields/ai/ConversationPanel';
import AudioRecorder from '@/components/fields/ai/AudioRecorder';
import MembershipUpgradeModal from '@/components/fields/ai/MembershipUpgradeModal';
import api from '@/services/axios';

// DynamicEngine이 전달하는 표준 props
interface AIChatComponentProps {
    meta: {
        labelText?: string;
        label_text?: string;
        cssClass?: string;
        css_class?: string;
        actionType?: string;
        action_type?: string;
        placeholder?: string;
        isReadonly?: boolean;
        is_readonly?: boolean;
    };
    data?: AIChatConfig; // pageData[ref_data_id] — query_master에서 조회한 설정값
    [key: string]: any;
}

export default function AIChatComponent({ meta, data }: AIChatComponentProps) {
    // ── 메타데이터에서 읽기 (DB에서 관리, 하드코딩 없음) ──
    const title = meta.labelText || meta.label_text || 'AI 대화';
    const containerClass = meta.cssClass || meta.css_class || '';
    const isDisabled = meta.isReadonly === true || meta.is_readonly === true;

    // language: data.language 우선, 없으면 actionType으로 판단
    const actionType = meta.actionType || meta.action_type || '';
    const language = (data?.language as 'en' | 'ko') ?? (actionType.includes('EN') ? 'en' : 'ko');

    // query_master 설정값 (모두 DB에서 관리, fallback만 하드코딩)
    const micBtnLabel = data?.mic_btn_label ?? '🎤 녹음 시작';
    const submitBtnLabel = data?.submit_btn_label ?? '답변완료';
    const endBtnLabel = data?.end_btn_label ?? '대화 종료';
    const welcomeMessage = data?.welcome_message ?? '';
    const upgradeMessage = data?.upgrade_message ?? '이 기능은 멤버십이 필요합니다.';

    // ── 상태 ──
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [showUpgradeModal, setShowUpgradeModal] = useState(false);
    const [upgradeModalMsg, setUpgradeModalMsg] = useState(upgradeMessage);
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
    const conversationStartedRef = useRef(false);

    // ── SSE 스트림 훅 ──
    const { stream, abort } = useSSEStream({
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
        onError: (msg) => {
            setIsStreaming(false);
            setUpgradeModalMsg(msg || upgradeMessage);
            setShowUpgradeModal(true);
        },
    });

    // ── AI에게 메시지 전송 ──
    const sendToAI = useCallback(async (msgs: ChatMessage[]) => {
        setIsStreaming(true);
        // 빈 AI 메시지 버블 추가 후 스트리밍으로 채움
        setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
        await stream('/api/ai/chat/stream', { messages: msgs, language });
    }, [stream, language]);

    // ── 마운트 시: 멤버십 체크 + 환영 메시지 ──
    useEffect(() => {
        if (isDisabled) return;

        const init = async () => {
            try {
                const res = await api.get('/api/v1/user-memberships/current');
                const membership = res.data?.data;

                if (!membership) {
                    setUpgradeModalMsg(upgradeMessage);
                    setShowUpgradeModal(true);
                    return;
                }

                // 환영 메시지: 한 번만 표시
                if (welcomeMessage && !conversationStartedRef.current) {
                    conversationStartedRef.current = true;
                    setMessages([{ role: 'assistant', content: welcomeMessage }]);
                }
            } catch {
                // 401은 axios 인터셉터 처리
            }
        };

        init();
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // ── 오디오 녹음 훅 ──
    const { state: recordingState, startRecording, stopRecording, resetState } = useAudioRecorder({
        onAudioReady: async (blob) => {
            try {
                const formData = new FormData();
                formData.append('audio', blob, 'recording.webm');

                const res = await api.post('/api/ai/stt', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });
                const transcript: string = res.data?.data?.text || '';

                if (!transcript.trim()) {
                    resetState();
                    return;
                }

                const userMsg: ChatMessage = { role: 'user', content: transcript };
                const updatedMsgs = [...messages, userMsg];
                setMessages(updatedMsgs);
                resetState();
                await sendToAI(updatedMsgs);
            } catch {
                resetState();
            }
        },
        onAnalyser: setAnalyser,
    });

    // ── 대화 종료 ──
    const handleEnd = () => {
        abort();
        setMessages([]);
        conversationStartedRef.current = false;
    };

    return (
        <div className={`ai-chat-container ${containerClass}`}>
            <div className="ai-chat-header">
                <h2 className="ai-chat-title">{title}</h2>
                {messages.length > 0 && (
                    <button className="ai-end-btn admin-back-btn" onClick={handleEnd}>
                        {endBtnLabel}
                    </button>
                )}
            </div>

            <ConversationPanel messages={messages} isStreaming={isStreaming} />

            <AudioRecorder
                state={recordingState}
                analyser={analyser}
                micBtnLabel={micBtnLabel}
                submitBtnLabel={submitBtnLabel}
                onStart={startRecording}
                onStop={stopRecording}
                disabled={isDisabled || isStreaming}
            />

            <MembershipUpgradeModal
                isOpen={showUpgradeModal}
                message={upgradeModalMsg}
                onClose={() => setShowUpgradeModal(false)}
            />
        </div>
    );
}
