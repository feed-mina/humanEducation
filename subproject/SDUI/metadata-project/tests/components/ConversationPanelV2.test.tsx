import React from 'react';
import { render, screen } from '@testing-library/react';
import ConversationPanelV2 from '@/components/fields/ai/ConversationPanelV2';
import { ChatMessage } from '@/lib/types/ai';

// jsdom에서 scrollIntoView가 구현되어 있지 않아서 mock 처리
window.HTMLElement.prototype.scrollIntoView = jest.fn();

describe('ConversationPanelV2', () => {

    describe('메시지 렌더링', () => {
        it('system 메시지는 렌더링하지 않아야 함', () => {
            const messages: ChatMessage[] = [
                { role: 'system', content: 'system instruction' },
                { role: 'assistant', content: 'Hello!' },
            ];
            render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(screen.queryByText('system instruction')).not.toBeInTheDocument();
            expect(screen.getByText('Hello!')).toBeInTheDocument();
        });

        it('user 메시지에 턴 번호와 단어 수를 표시해야 함', () => {
            const messages: ChatMessage[] = [
                { role: 'user', content: 'Hello world test' },
            ];
            render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(screen.getByText(/1턴/)).toBeInTheDocument();
            expect(screen.getByText(/3 단어/)).toBeInTheDocument();
        });

        it('isStreaming이 true일 때 로딩 인디케이터를 표시해야 함', () => {
            render(<ConversationPanelV2 messages={[]} isStreaming={true} />);
            expect(screen.getByText('AI가 생각 중입니다...')).toBeInTheDocument();
        });

        it('isStreaming이 false일 때 로딩 인디케이터를 표시하지 않아야 함', () => {
            render(<ConversationPanelV2 messages={[]} isStreaming={false} />);
            expect(screen.queryByText('AI가 생각 중입니다...')).not.toBeInTheDocument();
        });
    });

    describe('표현 평가 배지 (pronunciationScore)', () => {
        it('pronunciationScore가 있으면 평가 배지를 렌더링해야 함', () => {
            const messages: ChatMessage[] = [
                {
                    role: 'user',
                    content: "It's a beautiful day",
                    pronunciationScore: 85,
                    pronunciationSpoken: "It's a beautiful day",
                    pronunciationIdeal: 'What a beautiful day it is!',
                    pronunciationFeedback: 'Great expression!',
                },
            ];
            render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(screen.getByText('85점')).toBeInTheDocument();
            expect(screen.getByText(/내 표현:/)).toBeInTheDocument();
            expect(screen.getByText(/추천 표현:/)).toBeInTheDocument();
            expect(screen.getByText('Great expression!')).toBeInTheDocument();
        });

        it('pronunciationScore가 없으면 배지를 렌더링하지 않아야 함', () => {
            const messages: ChatMessage[] = [
                { role: 'user', content: 'Hello' },
            ];
            render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(screen.queryByText(/점/)).not.toBeInTheDocument();
        });

        it('pronunciationIdeal이 없으면 "추천 표현" 항목을 렌더링하지 않아야 함', () => {
            const messages: ChatMessage[] = [
                {
                    role: 'user',
                    content: 'Hello',
                    pronunciationScore: 60,
                    pronunciationSpoken: 'Hello',
                    pronunciationFeedback: 'Good try!',
                },
            ];
            render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(screen.getByText('60점')).toBeInTheDocument();
            expect(screen.queryByText(/추천 표현:/)).not.toBeInTheDocument();
        });

        it('assistant 메시지에는 평가 배지를 렌더링하지 않아야 함', () => {
            const messages: ChatMessage[] = [
                {
                    role: 'assistant',
                    content: 'Hello there!',
                    pronunciationScore: 90, // assistant에 잘못 설정된 경우도 표시 안 함
                } as ChatMessage,
            ];
            render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(screen.queryByText('90점')).not.toBeInTheDocument();
        });
    });

    describe('점수 레벨 CSS 클래스', () => {
        it.each([
            [90, 'excellent'],
            [85, 'excellent'],
            [75, 'good'],
            [65, 'good'],
            [55, 'fair'],
            [45, 'fair'],
            [30, 'poor'],
            [0,  'poor'],
        ])('score %i → score-%s CSS 클래스 적용', (score, expectedLevel) => {
            const messages: ChatMessage[] = [
                {
                    role: 'user',
                    content: 'test',
                    pronunciationScore: score,
                    pronunciationSpoken: 'test',
                },
            ];
            const { container } = render(<ConversationPanelV2 messages={messages} isStreaming={false} />);

            expect(container.querySelector(`.score-${expectedLevel}`)).toBeInTheDocument();
        });
    });
});
