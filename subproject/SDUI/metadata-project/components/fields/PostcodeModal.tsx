'use client';
import React, { useState, useEffect } from 'react';
import DaumPostcodeEmbed from 'react-daum-postcode';

// OPEN_POSTCODE 액션이 발생하면 window 커스텀 이벤트로 이 모달을 열고,
// 선택 완료 후 콜백으로 formData를 업데이트한다.
export default function PostcodeModal() {
    const [isOpen, setIsOpen] = useState(false);
    const [onComplete, setOnComplete] = useState<((data: any) => void) | null>(null);

    useEffect(() => {
        const handleOpen = (e: Event) => {
            const { detail } = e as CustomEvent;
            setOnComplete(() => detail.onComplete);
            setIsOpen(true);
        };
        window.addEventListener('open-postcode-modal', handleOpen);
        return () => window.removeEventListener('open-postcode-modal', handleOpen);
    }, []);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black bg-opacity-50 p-4">
            <div className="relative w-full max-w-lg bg-white rounded-lg shadow-xl overflow-hidden">
                <div className="p-4 border-b flex justify-between items-center bg-gray-50">
                    <span className="font-bold">주소 검색</span>
                    <button
                        onClick={() => setIsOpen(false)}
                        className="text-gray-500 hover:text-black text-xl leading-none"
                    >
                        &times;
                    </button>
                </div>
                <DaumPostcodeEmbed
                    onComplete={(data) => {
                        onComplete?.(data);
                        setIsOpen(false);
                    }}
                    style={{ height: '450px' }}
                />
            </div>
        </div>
    );
}
