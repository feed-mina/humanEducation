import { useState, useEffect } from 'react';

export const useDeviceType = () => {
    const [isMobile, setIsMobile] = useState(true); // 모바일 우선(SSR hydration 플래시 방지)

    useEffect(() => {
        const checkMobile = () => {
            setIsMobile(window.innerWidth < 1000); // 768px 미만을 모바일로 간주
        };

        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    return { isMobile, deviceClass: isMobile ? 'is-mobile' : 'is-pc' };
};