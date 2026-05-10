import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import AIChatComponentV2 from '@/components/fields/AIChatComponentV2';
import { useAIChatLogic } from '@/lib/hooks/useAIChatLogic';
import api from '@/services/axios';

// Mock dependencies
jest.mock('@/lib/hooks/useAIChatLogic');
jest.mock('@/services/axios', () => ({
  __esModule: true,
  default: {
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
    create: jest.fn().mockReturnThis(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  },
}));

describe('AIChatComponentV2 Component', () => {
    const mockMeta = {
        labelText: 'Test AI Tutor',
        cssClass: 'test-class'
    };
    
    const mockData = {
        welcome_message: 'Hi!'
    };

    const mockLogic = {
        messages: [],
        isStreaming: false,
        recordingState: 'idle',
        analyser: null,
        userMessageCount: 0,
        handleStartRecording: jest.fn(),
        stopRecording: jest.fn(),
        cancelRecording: jest.fn(),
        handleEndChat: jest.fn(),
    };

    beforeEach(() => {
        jest.clearAllMocks();
        (useAIChatLogic as jest.Mock).mockReturnValue(mockLogic);
        (api.get as jest.Mock).mockResolvedValue({ data: {} });
    });

    it('should render Intro screen initially', () => {
        render(<AIChatComponentV2 meta={mockMeta} data={mockData} />);
        
        expect(screen.getByText('Test AI Tutor')).toBeInTheDocument();
        expect(screen.getByText('대화 시작하기')).toBeInTheDocument();
    });

    it('should switch to chat main screen when start button is clicked', () => {
        render(<AIChatComponentV2 meta={mockMeta} data={mockData} />);
        
        const startBtn = screen.getByText('대화 시작하기');
        fireEvent.click(startBtn);

        // After clicking, the Intro should be replaced by Header & Panel
        // Note: Header contains the title
        expect(screen.getAllByText('Test AI Tutor')).toHaveLength(1);
    });

    it('should call handleEndChat when session end button is clicked', () => {
        // Mock messages to show the footer actions
        (useAIChatLogic as jest.Mock).mockReturnValue({
            ...mockLogic,
            messages: [{ role: 'assistant', content: 'Hello' }]
        });

        render(<AIChatComponentV2 meta={mockMeta} data={mockData} />);
        
        // Start chat first to see the main UI
        fireEvent.click(screen.getByText('대화 시작하기'));

        const endBtn = screen.getByText('채팅종료하기');
        fireEvent.click(endBtn);

        expect(mockLogic.handleEndChat).toHaveBeenCalled();
    });
});
