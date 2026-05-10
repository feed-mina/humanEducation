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

        let phase = 0;

        const draw = () => {
            rafRef.current = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            // 평균 볼륨 계산
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
            const average = sum / bufferLength;
            const volume = average / 128; // 0 ~ 1.0 (대략)

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 3개의 레이어로 Siri 스타일 연출
            const layers = [
                { color: 'rgba(74, 144, 226, 0.4)', amplitude: 20 * volume, speed: 0.1 },
                { color: 'rgba(142, 68, 173, 0.3)', amplitude: 15 * volume, speed: 0.15 },
                { color: 'rgba(74, 144, 226, 0.8)', amplitude: 25 * volume, speed: 0.08 }
            ];

            layers.forEach(layer => {
                ctx.beginPath();
                ctx.lineWidth = 2;
                ctx.strokeStyle = layer.color;

                for (let x = 0; x < canvas.width; x++) {
                    const y = canvas.height / 2 + 
                        Math.sin(x * 0.03 + phase * layer.speed) * layer.amplitude * 
                        Math.sin(x * 0.01); // 끝부분을 모아주는 효과
                    
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });

            phase += 1;
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
