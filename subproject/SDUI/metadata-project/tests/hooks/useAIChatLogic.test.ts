import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { useAIChatLogic } from '@/lib/hooks/useAIChatLogic';
import { useSSEStreamV2 } from '@/lib/hooks/useSSEStreamV2';
import { useAudioRecorder } from '@/lib/hooks/useAudioRecorder';
import api from '@/services/axios';
import { aiService } from '@/services/aiService';

// Mock dependencies
jest.mock('@/lib/hooks/useSSEStreamV2');
jest.mock('@/lib/hooks/useAudioRecorder');
jest.mock('@/services/axios');
jest.mock('@/services/aiService');

describe('useAIChatLogic Hook', () => {
    const mockStream = jest.fn();
    const mockAbort = jest.fn();
    const mockStartRecording = jest.fn();
    const mockStopRecording = jest.fn();
    const mockCancelRecording = jest.fn();
    const mockResetState = jest.fn();

    beforeEach(() => {
        jest.clearAllMocks();
        
        (useSSEStreamV2 as jest.Mock).mockReturnValue({
            stream: mockStream,
            abort: mockAbort,
        });

        (useAudioRecorder as jest.Mock).mockReturnValue({
            state: 'idle',
            startRecording: mockStartRecording,
            stopRecording: mockStopRecording,
            cancelRecording: mockCancelRecording,
            resetState: mockResetState,
        });
    });

    it('should initialize with welcome message', () => {
        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt',
            welcomeMessage: 'Hello traveler'
        }));

        expect(result.current.messages).toHaveLength(1);
        expect(result.current.messages[0].content).toBe('Hello traveler');
    });

    it('should calculate userMessageCount correctly', async () => {
        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt'
        }));

        act(() => {
            result.current.setMessages([
                { role: 'assistant', content: 'Hi' },
                { role: 'user', content: 'Hello' },
                { role: 'user', content: 'How are you?' }
            ]);
        });

        expect(result.current.userMessageCount).toBe(2);
    });

    it('should trigger onGoalAchieved on 10th user message', () => {
        const onGoalAchieved = jest.fn();
        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt',
            onGoalAchieved
        }));

        act(() => {
            const messages = Array(10).fill({ role: 'user', content: 'test' });
            result.current.setMessages(messages);
        });

        expect(onGoalAchieved).toHaveBeenCalled();
    });

    it('should start recording with correct mode', () => {
        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt'
        }));

        act(() => {
            result.current.handleStartRecording('ko');
        });

        expect(mockStartRecording).toHaveBeenCalled();
    });
});

describe('useAIChatLogic — 표현 평가 (pronunciation)', () => {
    const mockStream = jest.fn();
    const mockAbort = jest.fn();
    const mockResetState = jest.fn();

    // onAudioReady 콜백을 캡처하기 위한 변수
    let capturedOnAudioReady: ((blob: Blob) => Promise<void>) | undefined;

    beforeEach(() => {
        jest.clearAllMocks();

        (useSSEStreamV2 as jest.Mock).mockReturnValue({
            stream: mockStream,
            abort: mockAbort,
        });

        // onAudioReady 캡처
        (useAudioRecorder as jest.Mock).mockImplementation(({ onAudioReady }) => {
            capturedOnAudioReady = onAudioReady;
            return {
                state: 'idle',
                startRecording: jest.fn(),
                stopRecording: jest.fn(),
                cancelRecording: jest.fn(),
                resetState: mockResetState,
            };
        });

        // URL.createObjectURL mock
        global.URL.createObjectURL = jest.fn().mockReturnValue('blob:mock-url');
    });

    it('General Mic 모드에서 STT 후 checkPronunciation을 호출해야 함', async () => {
        // STT 응답 mock
        (api.post as jest.Mock).mockResolvedValue({
            data: { data: { text: "It's a beautiful day" } },
        });
        // 표현 평가 mock
        (aiService.checkPronunciation as jest.Mock).mockResolvedValue({
            score: 85,
            feedback: 'Great expression!',
            idealExpression: 'What a beautiful day it is!',
        });

        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt',
        }));

        // General Mic 모드로 녹음 시작
        act(() => {
            result.current.handleStartRecording('en');
        });

        // 녹음 완료 시뮬레이션
        const mockBlob = new Blob(['audio'], { type: 'audio/webm' });
        await act(async () => {
            await capturedOnAudioReady?.(mockBlob);
        });

        expect(aiService.checkPronunciation).toHaveBeenCalledWith("It's a beautiful day", 'en');

        const userMsg = result.current.messages.find(m => m.role === 'user');
        expect(userMsg?.pronunciationScore).toBe(85);
        expect(userMsg?.pronunciationFeedback).toBe('Great expression!');
        expect(userMsg?.pronunciationIdeal).toBe('What a beautiful day it is!');
        expect(userMsg?.pronunciationSpoken).toBe("It's a beautiful day");
    });

    it('한국어 번역 모드에서는 checkPronunciation을 호출하지 않아야 함', async () => {
        // STT 응답: 한국어
        (api.post as jest.Mock)
            .mockResolvedValueOnce({ data: { data: { text: '오늘 날씨가 좋네요' } } }) // STT
            .mockResolvedValueOnce({ data: { data: 'The weather is nice today' } });    // 번역

        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt',
        }));

        // 한국어 입력 모드로 시작
        act(() => {
            result.current.handleStartRecording('ko');
        });

        const mockBlob = new Blob(['audio'], { type: 'audio/webm' });
        await act(async () => {
            await capturedOnAudioReady?.(mockBlob);
        });

        // 번역 모드(wasTranslated=true)이므로 표현 평가 호출 안 됨
        expect(aiService.checkPronunciation).not.toHaveBeenCalled();

        const userMsg = result.current.messages.find(m => m.role === 'user');
        expect(userMsg?.pronunciationScore).toBeUndefined();
    });

    it('checkPronunciation 실패 시 메시지는 정상 추가되어야 함 (에러 무시)', async () => {
        (api.post as jest.Mock).mockResolvedValue({
            data: { data: { text: 'Hello there' } },
        });
        (aiService.checkPronunciation as jest.Mock).mockRejectedValue(
            new Error('GPT evaluation failed')
        );

        const { result } = renderHook(() => useAIChatLogic({
            language: 'en',
            systemPrompt: 'Test prompt',
        }));

        act(() => {
            result.current.handleStartRecording('en');
        });

        const mockBlob = new Blob(['audio'], { type: 'audio/webm' });
        await act(async () => {
            await capturedOnAudioReady?.(mockBlob);
        });

        // 에러가 있어도 user 메시지는 추가됨
        const userMsg = result.current.messages.find(m => m.role === 'user');
        expect(userMsg).toBeDefined();
        expect(userMsg?.content).toBe('Hello there');
        // 평가 데이터는 없음
        expect(userMsg?.pronunciationScore).toBeUndefined();
    });
});
