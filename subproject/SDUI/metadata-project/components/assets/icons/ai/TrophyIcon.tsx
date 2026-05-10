import React from 'react';

export default function TrophyIcon({ width = "7rem", height = "7rem" }: { width?: string, height?: string }) {
    return (
        <div style={{ fontSize: width, display: 'inline-block' }}>🏆</div>
    );
}
