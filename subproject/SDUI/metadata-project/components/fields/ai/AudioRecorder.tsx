'use client';

import React from 'react';
import { RecordingState } from '@/lib/types/ai';
import Waveform from './Waveform';
import MicIcon from '@/components/assets/icons/ai/MicIcon';
import StopIcon from '@/components/assets/icons/ai/StopIcon';

interface AudioRecorderProps {
    state: RecordingState;
    analyser: AnalyserNode | null;
    micBtnLabel: string;
    submitBtnLabel: string;
    onStart: (mode: string) => void;
    onStop: () => void;
    onCancel?: () => void;
    disabled?: boolean;
    className?: string;
    micClassName?: string;
    language?: string;
}

export default function AudioRecorder({
    state, analyser, micBtnLabel, submitBtnLabel,
    onStart, onStop, onCancel, disabled,
    className, micClassName, language,
}: AudioRecorderProps) {
    const isRecording = state === 'recording';
    const isProcessing = state === 'processing';

    return (
        <div className={`ai-recorder-container premium-glass ${className || ''}`}>
            {!isRecording && !isProcessing && (
                <div className="ai-recorder-intro">
                    <p className="ai-recorder-instruction">
                        모드를 선택하여 대화를 시작하세요.
                    </p>
                    
                    <div className="ai-recorder-modes">
                        {/* English/General Mode Button */}
                        <div className="ai-recorder-mode-item">
                            <button
                                className={`ai-mic-btn-circle ${micClassName || ''}`}
                                onClick={() => onStart(language || 'en')}
                                disabled={disabled || isProcessing}
                                title="English Mode (Phonetic)"
                            >
                                <MicIcon color="white" />
                            </button>
                            <span className="ai-recorder-mode-label en-label">General Mic</span>
                        </div>

                        {/* Korean Mode Button */}
                        <div className="ai-recorder-mode-item">
                            <button
                                className={`ai-mic-btn-circle ko-mic-circle ${micClassName || ''}`}
                                onClick={() => onStart('ko')}
                                disabled={disabled || isProcessing}
                                title="Speak in Korean (Translation)"
                            >
                                <div className="ai-ko-mic-content">
                                    <span className="ai-ko-kr">KR</span>
                                    <span className="ai-ko-text">KOREAN</span>
                                </div>
                            </button>
                            <span className="ai-recorder-mode-label ko-label">한국어로 말하기</span>
                        </div>
                    </div>
                </div>
            )}

            {isRecording && (
                <div className="ai-recorder-active">
                    <p className="ai-recorder-status-text">🔴 녹음 중... 완료 후 버튼을 눌러주세요</p>

                    <Waveform analyser={analyser} isActive={isRecording} />

                    <div className="ai-recorder-actions-row">
                        <button
                            className="ai-action-btn-cancel"
                            onClick={onCancel}
                        >
                            취소
                        </button>
                        <button
                            className={`ai-mic-btn-circle active-recording ${micClassName || ''}`}
                            onClick={onStop}
                            disabled={disabled || isProcessing}
                        >
                            <StopIcon />
                        </button>
                        <button
                            className="ai-action-btn-finish"
                            onClick={onStop}
                        >
                            답변 완료
                        </button>
                    </div>
                </div>
            )}

            {isProcessing && (
                <div className="ai-recorder-processing">
                    <div className="loading-spinner-small" />
                    <span className="ai-processing-text">AI가 답변을 생각하고 있어요...</span>
                </div>
            )}
        </div>
    );
}
