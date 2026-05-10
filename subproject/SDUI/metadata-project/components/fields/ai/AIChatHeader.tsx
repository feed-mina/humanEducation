'use client';

import React from 'react';

interface AIChatHeaderProps {
    title: string;
    userMessageCount: number;
}

export default function AIChatHeader({ title, userMessageCount }: AIChatHeaderProps) {
    const progress = Math.min((userMessageCount / 10) * 100, 100);
    const isGoalReached = userMessageCount >= 10;

    return (
        <div className="ai-chat-header">
            <div className="ai-header-content">
                <div className="ai-header-title-row">
                    <h2 className="ai-header-title">{title}</h2>
                </div>
                
                <div className="ai-gauge-wrapper">
                    <div className="ai-gauge-container">
                        <div 
                            className="ai-gauge-fill" 
                            style={{ 
                                width: `${progress}%`,
                                background: isGoalReached ? 'linear-gradient(90deg, #FFD700, #FFA000)' : undefined
                            }} 
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
