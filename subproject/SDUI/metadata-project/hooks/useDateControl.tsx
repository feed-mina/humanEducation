import {useCallback, useState} from 'react';

// 초기 날짜를 받아서 날짜 관련 모든 기능을 반환하는 훅
export const useDateControl = (initialData?: any) => {
    // 1. 날짜 상태 관리
    const [date, setDate] = useState<Date>(() =>
        initialData ? new Date(initialData) : new Date()
    );

    // 2. 모달 상태 관리
    const [isOpen, setIsOpen] = useState(false);
    const openCalendar = useCallback(() => setIsOpen(true), []);
    const closeCalendar = useCallback(() => setIsOpen(false), []);

    // 3. 순수 날짜 변경 함수 (시간 보존)
    const updateDateOnly = useCallback((newDate: Date) => {
        setDate((prev) => {
            const updated = new Date(newDate);
            updated.setHours(prev.getHours());
            updated.setMinutes(prev.getMinutes());
            return updated;
        });
        setIsOpen(false); // 날짜 선택 후 달력 닫기
    }, []);

    // 4. 순수 시간 변경 함수 (날짜 보존)
    const updateTimeOnly = useCallback((newDate: Date) => {
        setDate(newDate);
    }, []);

    // 5. 유틸리티: 분 더하기
    const addMinutes = useCallback((mins: number) => {
        setDate((prev) => {
            const newDate = new Date(prev);
            newDate.setMinutes(prev.getMinutes() + mins);
            return newDate;
        });
    }, []);

    return {
        date,
        setDate, // 필요하면 직접 제어
        isOpen,
        openCalendar,
        closeCalendar,
        updateDateOnly, // 달력 선택 시 사용
        updateTimeOnly, // 휠/입력 시 사용
        addMinutes      // 퀵 버튼 시 사용
    };
};