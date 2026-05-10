import {useCallback, useState} from 'react';

// 초기 날짜를 인자로 받습니다.
export const useCalendar = (initialDate: Date) => {
    // 1. 달력 모달의 열림/닫힘 상태
    const [isOpen, setIsOpen] = useState(false);

    // 2. 현재 선택된 날짜 상태 (초기값은 부모가 준 값)
    const [date, setDate] = useState<Date>(initialDate);

    // 3. 모달 제어 함수들
    const openCalendar = () => setIsOpen(true);
    const closeCalendar = () => setIsOpen(false);

    // 4. 날짜 변경 핸들러 (핵심 로직)
    const handleDateChange = useCallback((newDate: Date | null) => {
        // 라이브러리가 null을 줄 수도 있으므로 방어 코드 작성
        if (!newDate) return;

        setDate((prevDate) => {
            // 중요: 기존(prevDate)의 시간 정보를 새 날짜(newDate)에 이식합니다.
            const updatedDate = new Date(newDate);
            updatedDate.setHours(prevDate.getHours());
            updatedDate.setMinutes(prevDate.getMinutes());
            return updatedDate;
        });

        // 날짜 선택 후 달력을 닫습니다.
        setIsOpen(false);
    }, []);

    // 5. 외부에서 시간을 변경했을 때(휠 스크롤 등) 동기화하는 함수
    const updateTime = useCallback((newTimeDate: Date) => {
        setDate(newTimeDate);
    }, []);

    return {
        date,           // 현재 날짜 객체
        isOpen,         // 달력 열림 여부
        openCalendar,   // 달력 열기 함수
        closeCalendar,  // 달력 닫기 함수
        handleDateChange, // 달력에서 날짜 클릭 시 실행할 함수
        updateTime      // 휠로 시간 변경 시 실행할 함수
    };
};