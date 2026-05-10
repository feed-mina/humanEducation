// DESTINATION: metadata-project/lib/hooks/useAudioRecorder.ts
import { useCallback, useRef, useState } from 'react';
import { RecordingState } from '@/lib/types/ai';

interface UseAudioRecorderOptions {
    onAudioReady: (blob: Blob) => void;
    onAnalyser?: (analyser: AnalyserNode) => void;
}

export function useAudioRecorder({ onAudioReady, onAnalyser }: UseAudioRecorderOptions) {
    const [state, setState] = useState<RecordingState>('idle');
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    const startRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // AudioContext: 웨이브폼 시각화용 (브라우저당 최대 6개 제한, ref로 단일 인스턴스 유지)
            audioContextRef.current = new AudioContext();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            const analyser = audioContextRef.current.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            onAnalyser?.(analyser);

            chunksRef.current = [];
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                onAudioReady(blob);
                // 마이크 스트림 정리
                stream.getTracks().forEach(t => t.stop());
                audioContextRef.current?.close();
                audioContextRef.current = null;
                setState('processing');
            };

            recorder.start();
            setState('recording');
        } catch {
            alert('마이크 권한이 필요합니다. 브라우저 설정에서 허용해주세요.');
            setState('idle');
        }
    }, [onAudioReady, onAnalyser]);

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    const resetState = useCallback(() => setState('idle'), []);

    return { state, startRecording, stopRecording, resetState };
}
