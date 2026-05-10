import React from 'react';

export default function InterviewIcon({ color = "#1e293b", width = "80px", height = "80px" }: { color?: string, width?: string, height?: string }) {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width, height, display: 'block' }}>
            <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
            <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
            <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            <rect x="14" y="14" width="8" height="6" rx="1" strokeWidth="1" />
            <line x1="16" y1="17" x2="20" y2="17" strokeWidth="1" />
        </svg>
    );
}
