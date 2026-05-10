// DESTINATION: metadata-project/components/fields/ai/AudioRecorder.tsx
'use client';

import React from 'react';
import { RecordingState } from '@/lib/types/ai';
import Waveform from './Waveform';

interface AudioRecorderProps {
    state: RecordingState;
    analyser: AnalyserNode | null;
    micBtnLabel: string;
    submitBtnLabel: string;
    onStart: () => void;
    onStop: () => void;
    disabled?: boolean;
}

export default function AudioRecorder({
    state, analyser, micBtnLabel, submitBtnLabel,
    onStart, onStop, disabled,
}: AudioRecorderProps) {
    const isRecording = state === 'recording';
    const isProcessing = state === 'processing';

    return (
        <div className="audio-recorder-bar">
            <button
                className={`mic-btn ${isRecording ? 'mic-btn--active' : ''}`}
                onClick={isRecording ? onStop : onStart}
                disabled={disabled || isProcessing}
                aria-label={isRecording ? '녹음 중지' : micBtnLabel}
            >
                {isRecording ? '⏹' : '🎤'}
            </button>

            <Waveform analyser={analyser} isActive={isRecording} />

            {isRecording && (
                <button className="submit-btn content-btn" onClick={onStop}>
                    {submitBtnLabel}
                </button>
            )}

            {isProcessing && (
                <span className="processing-label">처리 중...</span>
            )}
        </div>
    );
}
