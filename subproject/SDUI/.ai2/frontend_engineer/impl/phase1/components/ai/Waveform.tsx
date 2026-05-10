// DESTINATION: metadata-project/components/fields/ai/Waveform.tsx
'use client';

import React, { useEffect, useRef } from 'react';

interface WaveformProps {
    analyser: AnalyserNode | null;
    isActive: boolean;
}

export default function Waveform({ analyser, isActive }: WaveformProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rafRef = useRef<number>(0);

    useEffect(() => {
        if (!isActive || !analyser) {
            cancelAnimationFrame(rafRef.current);
            const canvas = canvasRef.current;
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx?.clearRect(0, 0, canvas.width, canvas.height);
            }
            return;
        }

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            rafRef.current = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#4f8ef7';
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        };

        draw();
        return () => cancelAnimationFrame(rafRef.current);
    }, [analyser, isActive]);

    return (
        <canvas
            ref={canvasRef}
            className="waveform-canvas"
            width={200}
            height={48}
        />
    );
}
