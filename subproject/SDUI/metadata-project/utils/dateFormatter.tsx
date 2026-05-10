export const dateFormatter = () => {
    const formatGoalDate = (dateStr: string) => {
        if (!dateStr) return "";
        const originalDate = new Date(dateStr);
        // 목표시간 보다 10분 일찍 보인다
        const displayDate = new Date(originalDate.getTime() - 10 * 60000);
        const month = String(displayDate.getMonth() + 1).padStart(2, '0');
        const day = String(displayDate.getDate()).padStart(2, '0');
        const hours = String(displayDate.getHours()).padStart(2, '0');
        const minutes = String(displayDate.getMinutes()).padStart(2, '0');

        const dayNames = ['일요일', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일'];
        const dayName = dayNames[displayDate.getDay()];

        return `${month}월 ${day}일${dayName}`;
    };

// [유틸] time포맷함수 (오전/오후 HH:MM)
    const formatTimePretty = (dateStr: string) => {
        if (!dateStr) return "";
        const originalDate = new Date(dateStr);
        // 목표시간 보다 10분 일찍 보인다
        const displayDate = new Date(originalDate.getTime() - 10 * 60000);
        const hours = displayDate.getHours();
        const minutes = displayDate.getMinutes();
        const ampm = hours >= 12 ? '오후' : '오전';
        const displayHours = hours % 12 || 12;
        const displayMinutes = minutes < 10 ? `0${minutes}` : minutes;
        return `${ampm} ${displayHours}시 ${displayMinutes}분`;
    };
// [유틸] 날짜만 보여주는함수 (YYYY-MM-DD)
    const formatDateOnly = (dateStr: string) => {
        if (!dateStr) return "";
        return dateStr.split("T")[0];
    }
    return {formatGoalDate, formatTimePretty, formatDateOnly};
};
