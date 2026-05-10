import {useCallback, useEffect, useRef, useState} from 'react';

const ITEM_HEIGHT = 50;

// 이 훅이 작동하기 위해 필요한 재료들 (DateControl에서 받아옴)
interface UseTimeWheelProps {
    date: Date;
    updateTimeOnly: (date: Date) => void; // 시간 변경 함수
    onChange?: (id: string, value: string) => void; // 부모 알림
    id: string;
}

export const useTimeWheel = ({date, updateTimeOnly, onChange, id}: UseTimeWheelProps) => {
    // 1. Refs (DOM 접근용)
    const hourRef = useRef<HTMLDivElement>(null);
    const minuteRef = useRef<HTMLDivElement>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const isInitialized = useRef(false);

    // 2. 입력 모드 상태 (UI 전용)
    const [isInputMode, setIsInputMode] = useState(false);
    const [inputValues, setInputValues] = useState({hour: '00', minute: '00'});

    // 3. 부모에게 변경 알림 (공통)
    const notifyChange = useCallback((newDate: Date) => {
        if (onChange) onChange(id, newDate.toISOString());
    }, [onChange, id]);

    // 4. 스크롤 위치 동기화 (UI -> DOM)
    const syncScrollToData = useCallback((targetDate: Date) => {
        if (!hourRef.current || !minuteRef.current) return;

        const h = targetDate.getHours();
        const m = targetDate.getMinutes();

        // 약간의 지연을 주어 UI 렌더링 후 이동
        setTimeout(() => {
            if (hourRef.current) hourRef.current.scrollTo({top: h * ITEM_HEIGHT, behavior: 'smooth'});
            if (minuteRef.current) minuteRef.current.scrollTo({top: m * ITEM_HEIGHT, behavior: 'smooth'});
        }, 50);
    }, []);

    // 5. 초기화 및 모드 전환 시 스크롤 보정
    useEffect(() => {
        if (isInitialized.current && isInputMode) return; // 입력 모드 중엔 무시

        // 초기화거나 입력 모드가 끝났을 때 스크롤 맞춤
        if (!isInitialized.current) {
            isInitialized.current = true;
            if (onChange) onChange(id, date.toISOString()); // 초기값 전송
        }

        // 휠이 화면에 있을 때만 스크롤 이동
        if (!isInputMode) {
            setTimeout(() => syncScrollToData(date), 0);
        }
    }, [date, isInputMode, onChange, id, syncScrollToData]);


    // --- 이벤트 핸들러 ---

    // 휠 스크롤 핸들러
    const handleScroll = (type: 'hour' | 'minute') => {
        if (timerRef.current) clearTimeout(timerRef.current);

        timerRef.current = setTimeout(() => {
            const ref = type === 'hour' ? hourRef.current : minuteRef.current;
            if (!ref) return;

            const scrollTop = ref.scrollTop;
            const value = Math.round(scrollTop / ITEM_HEIGHT);

            ref.scrollTo({top: value * ITEM_HEIGHT, behavior: 'smooth'});

            const newDate = new Date(date);
            if (type === 'hour') newDate.setHours(value);
            else newDate.setMinutes(value);

            updateTimeOnly(newDate); // 데이터 업데이트
            notifyChange(newDate);   // 부모 알림
        }, 150);
    };

    // 클릭 시 입력 모드 진입
    const handleWheelClick = () => {
        setInputValues({
            hour: date.getHours().toString().padStart(2, '0'),
            minute: date.getMinutes().toString().padStart(2, '0')
        });
        setIsInputMode(true);
    };

    // 입력값 변경
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'hour' | 'minute') => {
        const val = e.target.value;
        if (!/^\d{0,2}$/.test(val)) return;
        setInputValues(prev => ({...prev, [type]: val}));
    };

    // 입력 확정 및 종료
    const handleInputConfirm = () => {
        let h = parseInt(inputValues.hour || '0', 10);
        let m = parseInt(inputValues.minute || '0', 10);

        // 유효성 검사 (0~23, 0~59)
        h = Math.min(23, Math.max(0, isNaN(h) ? 0 : h));
        m = Math.min(59, Math.max(0, isNaN(m) ? 0 : m));

        const newDate = new Date(date);
        newDate.setHours(h);
        newDate.setMinutes(m);

        updateTimeOnly(newDate);
        notifyChange(newDate);
        setIsInputMode(false); // 휠 모드로 복귀 -> useEffect가 스크롤 맞춤
    };

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
        if (e.relatedTarget && (e.relatedTarget as HTMLElement).classList.contains('time-input')) return;
        handleInputConfirm();
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') handleInputConfirm();
    };

    return {
        // Refs
        hourRef,
        minuteRef,
        // State
        isInputMode,
        inputValues,
        // Handlers
        handleScroll,
        handleWheelClick,
        handleInputChange,
        handleBlur,
        handleKeyDown,
        // Utils
        notifyChange,
        syncScrollToData
    };
};