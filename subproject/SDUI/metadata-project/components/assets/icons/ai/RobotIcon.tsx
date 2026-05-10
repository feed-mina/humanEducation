import React from 'react';

export default function RobotIcon({ width = "32px", height = "32px" }: { width?: string, height?: string }) {
    return (
        <svg viewBox="0 0 100 100" style={{ width, height }}>
            <rect x="20" y="30" width="60" height="50" rx="10" fill="#E0E7FF" stroke="#4F46E5" strokeWidth="3" />
            <rect x="35" y="45" width="10" height="10" rx="5" fill="#4F46E5" />
            <rect x="55" y="45" width="10" height="10" rx="5" fill="#4F46E5" />
            <path d="M40 65 Q50 75 60 65" stroke="#4F46E5" strokeWidth="3" fill="none" strokeLinecap="round" />
            <rect x="45" y="20" width="10" height="10" rx="2" fill="#818CF8" />
            <line x1="50" y1="30" x2="50" y2="25" stroke="#4F46E5" strokeWidth="2" />
        </svg>
    );
}
