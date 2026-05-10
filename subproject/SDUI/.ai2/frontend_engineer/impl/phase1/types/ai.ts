// DESTINATION: metadata-project/lib/types/ai.ts

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface ChatRequest {
    messages: ChatMessage[];
    language: 'en' | 'ko';
}

export type RecordingState = 'idle' | 'recording' | 'processing';
export type ConversationState = 'idle' | 'ai_speaking' | 'user_turn' | 'processing';

export interface AIChatConfig {
    mic_btn_label: string;
    submit_btn_label: string;
    end_btn_label: string;
    welcome_message: string;
    language: 'en' | 'ko';
    required_tier: string;
    upgrade_message: string;
}

export interface InterviewStartRequest {
    resumeText?: string;
    resumeImageBase64?: string; // Phase 2: 이미지 업로드 (백엔드 패치 필요)
    language?: 'ko' | 'en';
}

export type ResumeInputType = 'text' | 'pdf' | 'image';

export interface AIInterviewConfig {
    resume_placeholder: string;
    start_btn_label: string;
    answer_btn_label: string;
    mic_btn_label: string;
    end_btn_label: string;
    language: 'ko' | 'en';
    required_tier: string;
    upgrade_message: string;
}
