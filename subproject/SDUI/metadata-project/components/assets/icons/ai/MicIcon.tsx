import React from 'react';

export default function MicIcon({ color = "#3F51B5", width = "40px", height = "40px" }: { color?: string, width?: string, height?: string }) {
    return (
        <svg viewBox="0 0 24 24" fill={color} style={{ width, height, display: 'block' }}>
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
        </svg>
    );
}
