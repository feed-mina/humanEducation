import React from 'react';
import { render, screen } from '@testing-library/react';
import AIChatComponentV2 from '@/components/fields/AIChatComponentV2';
import { AIChatComponentProps } from '@/lib/types/ai';

// Mock dependency hooks
jest.mock('@/lib/hooks/useAIChatLogic', () => ({
    useAIChatLogic: () => ({
        messages: [],
        isStreaming: false,
        recordingState: 'idle',
        analyser: null,
        userMessageCount: 0,
        handleStartRecording: jest.fn(),
        stopRecording: jest.fn(),
        cancelRecording: jest.fn(),
        handleEndChat: jest.fn(),
    }),
}));

describe('AI Japanese Chat (AIChatComponentV2 with JA meta)', () => {
    const mockProps: AIChatComponentProps = {
        meta: {
            labelText: 'Japanese Tutor',
            target_language: 'ja',
            cssClass: 'ai-japanese-theme',
        },
        data: {
            welcome_message: 'こんにちは',
        },
    };

    it('renders the intro screen with Japanese welcome expectation', () => {
        render(<AIChatComponentV2 {...mockProps} />);
        expect(screen.getByText('Japanese Tutor')).toBeInTheDocument();
        expect(screen.getByText('대화 시작하기')).toBeInTheDocument(); 
    });
});
