import { useMemo, useEffect, useRef } from 'react';
import { ChatMessage } from '@/lib/types/ai';

interface UseGoalTrackerProps {
    messages: ChatMessage[];
    goalCount?: number;
    onGoalAchieved?: () => void;
}

/**
 * 목표 달성(사용자 메시지 횟수) 추적 훅
 * useAIChatLogic에서 분리하여 SRP 준수
 */
export function useGoalTracker({
    messages,
    goalCount = 10,
    onGoalAchieved,
}: UseGoalTrackerProps) {
    const hasTriggeredGoalRef = useRef(false);

    const userMessageCount = useMemo(
        () => messages.filter((m) => m.role === 'user').length,
        [messages]
    );

    useEffect(() => {
        if (userMessageCount === goalCount && !hasTriggeredGoalRef.current) {
            onGoalAchieved?.();
            hasTriggeredGoalRef.current = true;
        }
    }, [userMessageCount, goalCount, onGoalAchieved]);

    const reset = () => {
        hasTriggeredGoalRef.current = false;
    };

    return { userMessageCount, reset };
}
