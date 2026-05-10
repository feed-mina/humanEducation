// DESTINATION: metadata-project/components/fields/ai/ConversationPanel.tsx
'use client';

import React, { useEffect, useRef } from 'react';
import { ChatMessage } from '@/lib/types/ai';

interface ConversationPanelProps {
    messages: ChatMessage[];
    isStreaming: boolean;
}

export default function ConversationPanel({ messages, isStreaming }: ConversationPanelProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="conversation-panel">
            {messages.map((msg, i) => (
                <div key={i} className={`message-bubble message-${msg.role}`}>
                    <span className="message-avatar">
                        {msg.role === 'assistant' ? '🤖' : '👤'}
                    </span>
                    <div className="message-content">
                        {msg.content}
                        {isStreaming && i === messages.length - 1 && msg.role === 'assistant' && (
                            <span className="streaming-cursor">▌</span>
                        )}
                    </div>
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    );
}
