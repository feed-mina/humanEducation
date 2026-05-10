// hooks/useRenderCount.ts
import { useRef, useEffect } from 'react';
import { useParams } from 'next/navigation';

export const useRenderCount = (componentName: string) => {
    const count = useRef(1);
    const params = useParams();
    const screenId = params?.screenId;

    useEffect(() => {
        count.current += 1;

        // 테스트 환경에서만 글로벌 객체에 저장
        if (typeof window !== 'undefined' && process.env.NODE_ENV === 'test') {
            if (!(window as any).__componentRenderCounts__) {
                (window as any).__componentRenderCounts__ = {};
            }
            (window as any).__componentRenderCounts__[componentName] = count.current;
        }
    });

    // 프로덕션 환경에서는 console.log 활성화 (디버깅용)
    if (process.env.NODE_ENV !== 'test') {
        // console.log(`[Screen: ${screenId}] ${componentName} 렌더링 횟수: ${count.current}`);
    }

    return count.current;
};