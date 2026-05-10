import React, { useEffect, useRef } from 'react';
//  window 객체에 naver가 있음을 타입스크립트에게 알려주는 선언이야.
declare global {
    interface Window {
        naver: any;
    }
}
interface MapViewProps {
    id: string;
    meta: any;
    data: any;
}

// 2. 인자에 타입을 명시해서 TS7031 에러를 해결해.
const MapView = ({ id, meta, data }: MapViewProps) => {
    // 1. 지도가 그려질 HTML 엘리먼트를 참조하기 위한 변수야.
    const mapRef = useRef<HTMLDivElement>(null);
    // 2. 네이버 지도 객체를 담아둘 변수야.
    const naverMap = useRef<any>(null);

    useEffect(() => {
        // 3. window.naver로 접근해서 인식을 도와줘.
        if (window.naver && window.naver.maps && mapRef.current) {
            const location = new window.naver.maps.LatLng(data.lat, data.lng);

            naverMap.current = new window.naver.maps.Map(mapRef.current, {
                center: location,
                zoom: 15,
            });

            new window.naver.maps.Marker({
                position: location,
                map: naverMap.current,
            });
        }
    }, [data]);
    // 6. 서버에서 데이터(data)가 바뀔 때마다 지도를 다시 그려.

    return (
        <div id={id} ref={mapRef} style={{ width: '100%', height: '400px' }} />
    );
};

export default MapView;