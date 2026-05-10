import React, { useEffect, useRef } from 'react';

// window 객체에 네이버 지도 정의
declare global { interface Window { naver: any; } }

const AdminMapView = ({ workersData }: { workersData: any[] }) => {
    const mapRef = useRef<HTMLDivElement>(null);
    const naverMap = useRef<any>(null);
    // 마커들을 저장해둘 공간 (Key: userSqno, Value: Marker 객체)
    const markersRef = useRef<Map<number, any>>(new Map());

    useEffect(() => {
        // 1. 지도 초기화 (최초 1회)
        if (!naverMap.current && window.naver && mapRef.current) {
            naverMap.current = new window.naver.maps.Map(mapRef.current, {
                center: new window.naver.maps.LatLng(37.5665, 126.9780),
                zoom: 12,
            });
        }

        // 2. 아주머니 데이터(workersData)가 변할 때마다 마커 업데이트
        workersData.forEach((worker) => {
            const { userSqno, lat, lng, status } = worker;
            const newPos = new window.naver.maps.LatLng(lat, lng);

            if (markersRef.current.has(userSqno)) {
                // 기존 마커가 있다면 위치만 이동
                const marker = markersRef.current.get(userSqno);
                marker.setPosition(newPos);
                // 긴급 상황(HELP)이면 마커 색상 변경 로직 추가 가능
            } else {
                // 새로운 아주머니라면 마커 신규 생성
                const newMarker = new window.naver.maps.Marker({
                    position: newPos,
                    map: naverMap.current,
                    title: `작업자 ${userSqno}`,
                    icon: {
                        content: `<div style="padding:5px; background:${status === 'HELP' ? 'red' : 'blue'}; color:white; border-radius:50%;">${userSqno}</div>`,
                        anchor: new window.naver.maps.Point(11, 11),
                    }
                });
                markersRef.current.set(userSqno, newMarker);
            }
        });
    }, [workersData]);

    return <div ref={mapRef} style={{ width: '100%', height: '600px' }} />;
};