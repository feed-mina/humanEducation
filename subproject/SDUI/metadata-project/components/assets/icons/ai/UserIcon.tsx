import React from 'react';

export default function UserIcon({ color = "#6366F1", width = "32px", height = "32px" }: { color?: string, width?: string, height?: string }) {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width, height }}>
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
        </svg>
    );
}
