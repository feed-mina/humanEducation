import React from 'react';

export default function SpeakerIcon({ color = "currentColor", width = "16px", height = "16px" }: { color?: string, width?: string, height?: string }) {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width, height }}>
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill={color} />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        </svg>
    );
}
