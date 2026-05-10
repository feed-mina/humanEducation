// [유틸] 목표 날짜 포맷 (MM-DD HH:MM 요일) - 요청하신 포맷!
import {memo} from "react";


interface ButtonProps {
    onClick: () => void;
}

export const ArrivalButton = memo(({onClick}: ButtonProps) => {
    // console.log("버튼은 안 움직이고 가만히 있는다");
    return (
        <button className="arrival-button" onClick={onClick}>
            도착 완료
        </button>
    );
});
ArrivalButton.displayName = "ArrivalButton";
