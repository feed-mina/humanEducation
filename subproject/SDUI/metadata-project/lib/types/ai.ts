export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    audioUrl?: string;              // 사용자가 녹음한 본인 목소리 URL
    translation?: string;           // 한국어 번역
    originalText?: string;          // 한국어로 말하기 모드에서 원본 한국어 텍스트
    pronunciationScore?: number;    // 표현 품질 점수 (0~100)
    pronunciationSpoken?: string;   // 사용자가 실제로 발화한 텍스트 (STT 결과)
    pronunciationIdeal?: string;    // GPT가 생성한 이상적 표현
    pronunciationFeedback?: string; // 피드백 메시지
}

export interface ChatRequest {
    messages: ChatMessage[];
    language: string; // 'en' | 'ko' | 'ja' 등 확장 가능하게 string으로 변경
}

export type RecordingState = 'idle' | 'recording' | 'processing';
export type ConversationState = 'idle' | 'ai_speaking' | 'user_turn' | 'processing';

/**
 * SDUI Meta 인터페이스
 */
export interface AIChatMeta {
    labelText?: string;
    label_text?: string;
    cssClass?: string;
    css_class?: string;
    actionType?: string;
    action_type?: string;
    placeholder?: string;
    isReadonly?: boolean;
    is_readonly?: boolean;
    // 추가적인 제어용 메타데이터 (언어 설정 등)
    target_language?: string; 
    system_prompt_template?: string;
}

/**
 * SDUI Data 인터페이스 (query_master 등에 설정된 값)
 */
export interface AIChatConfig {
    mic_btn_label?: string;
    submit_btn_label?: string;
    end_btn_label?: string;
    welcome_message?: string;
    language?: string;
    required_tier?: string;
    upgrade_message?: string;
}

/**
 * AIChat 전용 Props (DynamicEngine용)
 */
export interface AIChatComponentProps {
    meta: AIChatMeta;
    data?: AIChatConfig;
    [key: string]: any;
}

// 면접 관련 타입 (유지)
export interface InterviewStartRequest {
    resumeText?: string;
    resumeImageBase64?: string;
    language?: string;
}

export type ResumeInputType = 'text' | 'pdf' | 'image';

export interface AIInterviewConfig {
    resume_placeholder: string;
    start_btn_label: string;
    answer_btn_label: string;
    mic_btn_label: string;
    end_btn_label: string;
    language: string;
    required_tier: string;
    upgrade_message: string;
}
