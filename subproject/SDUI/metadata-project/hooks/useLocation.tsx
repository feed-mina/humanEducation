import {useEffect, useState} from "react";
import {useRouter} from "next/navigation";
import axios from "@/services/axios";
import { useAuth } from "@/context/AuthContext";

//  @@@@ usePageMetadata 역할 : 메타데이터가져오기 , 원본 데이터 가져오기 , 가져온 데이터를 pageData로 담아줌, 로딩중인지 전체 개수가 몇개인지 같은 페이지의 전역 상태를 관리
export const usePageMetadata = (screenId: string) => {
    // 위치 정보를 가져오는 기본 로직 예시
    const getLocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    // console.log("위도:", lat, "경도:", lng);
                    // 여기서 서버로 전송하는 API 호출
                },
                (error) => {
                    // console.error("위치 정보를 가져오는 데 실패했습니다.", error);
                }
            );
        } else {
            // console.log("이 브라우저에서는 Geolocation을 지원하지 않습니다.");
        }
    };


};