import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import AIInterviewComponent from '@/components/fields/AIInterviewComponent';
import { AIChatComponentProps } from '@/lib/types/ai';

const mockUseInterviewLogic = jest.fn().mockReturnValue({
    messages: [],
    isStreaming: false,
    recordingState: 'idle',
    analyser: null,
    startInterview: jest.fn().mockResolvedValue(undefined),
    handleStartRecording: jest.fn(),
    stopRecording: jest.fn(),
    cancelRecording: jest.fn(),
    handleEndInterview: jest.fn(),
});

jest.mock('@/lib/hooks/useInterviewLogic', () => ({
    useInterviewLogic: (props: unknown) => mockUseInterviewLogic(props),
}));

describe('AIInterviewComponent', () => {
    const mockProps: AIChatComponentProps = {
        meta: {
            labelText: 'AI Interview Test',
        },
        data: {
            upgrade_message: 'Premium required.',
        },
    };

    it('renders the intro screen with title and resume textarea', () => {
        render(<AIInterviewComponent {...mockProps} />);
        expect(screen.getByText('AI Interview Test')).toBeInTheDocument();
        expect(screen.getByText('면접 시작하기')).toBeInTheDocument();
        expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    it('shows Korean defaults when no data provided', () => {
        render(<AIInterviewComponent meta={{ labelText: 'Test' }} />);
        expect(screen.getByText('면접 시작하기')).toBeInTheDocument();
        expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    it('passes system_prompt_template from meta to useInterviewLogic', () => {
        const propsWithTemplate: AIChatComponentProps = {
            meta: {
                labelText: 'Test',
                system_prompt_template: '당신은 Java 백엔드 전문 면접관입니다.',
            },
        };
        render(<AIInterviewComponent {...propsWithTemplate} />);
        expect(mockUseInterviewLogic).toHaveBeenCalledWith(
            expect.objectContaining({ systemPromptTemplate: '당신은 Java 백엔드 전문 면접관입니다.' })
        );
    });

    it('shows main interview panel after starting interview', async () => {
        render(<AIInterviewComponent {...mockProps} />);
        const textarea = screen.getByRole('textbox');
        fireEvent.change(textarea, { target: { value: '이름: 홍길동, 경력: 3년' } });

        const startBtn = screen.getByText('면접 시작하기');
        await act(async () => {
            fireEvent.click(startBtn);
        });

        await waitFor(() => {
            expect(screen.getByText('면접 종료')).toBeInTheDocument();
        });
    });
});
