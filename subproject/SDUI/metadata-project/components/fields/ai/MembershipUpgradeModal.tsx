'use client';

import React from 'react';
import { useRouter } from 'next/navigation';

interface MembershipUpgradeModalProps {
    isOpen: boolean;
    message: string;
    onClose: () => void;
}

export default function MembershipUpgradeModal({ isOpen, message, onClose }: MembershipUpgradeModalProps) {
    const router = useRouter();

    if (!isOpen) return null;

    const handleUpgrade = () => {
        onClose();
        router.push('/view/MEMBERSHIP_SHOP_PAGE');
    };

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="modal-box" onClick={e => e.stopPropagation()}>
                <p className="modal-message">{message}</p>
                <div className="modal-actions">
                    <button className="content-btn" onClick={handleUpgrade}>
                        멤버십 업그레이드
                    </button>
                    <button className="admin-back-btn" onClick={onClose}>
                        닫기
                    </button>
                </div>
            </div>
        </div>
    );
}
