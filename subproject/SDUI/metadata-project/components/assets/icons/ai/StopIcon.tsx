import React from 'react';

export default function StopIcon({ color = "white", width = "32px", height = "32px" }: { color?: string, width?: string, height?: string }) {
    return (
        <svg viewBox="0 0 24 24" fill={color} style={{ width, height, display: 'block' }}>
            <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
    );
}
