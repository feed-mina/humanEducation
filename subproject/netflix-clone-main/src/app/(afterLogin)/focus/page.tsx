'use client';
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useOnboardingStore } from "@/store/onboarding-store";
import MapView from "@/components/kride/MapView";
import ItineraryPanel from "@/components/kride/ItineraryPanel";

// 임시 일정 데이터 (K-Ride FastAPI 연동 전)
const MOCK_ITINERARY = {
  itinerary: [
    {
      morning: {
        places: [
          { name: "경복궁", description: "조선시대 정궁" },
          { name: "북촌한옥마을", description: "전통 한옥 거리" },
        ],
      },
      afternoon: {
        places: [
          { name: "인사동", description: "전통 문화 거리" },
          { name: "광장시장", description: "전통 시장 먹거리" },
        ],
      },
    },
    {
      morning: {
        places: [
          { name: "남산타워", description: "서울 전경 조망" },
          { name: "이태원", description: "다국적 문화 거리" },
        ],
      },
      afternoon: {
        places: [
          { name: "홍대", description: "K-culture 성지" },
          { name: "연남동", description: "감성 카페 거리" },
        ],
      },
    },
    {
      morning: {
        places: [
          { name: "강남 COEX", description: "도심 쇼핑몰" },
        ],
      },
      afternoon: {
        places: [
          { name: "잠실 롯데월드", description: "테마파크" },
          { name: "석촌호수", description: "벚꽃 명소" },
        ],
      },
    },
  ],
  center: [37.5665, 126.978] as [number, number],
  markers: [
    { lat: 37.5796, lng: 126.977, name: "경복궁", index: 0 },
    { lat: 37.5823, lng: 126.9836, name: "북촌한옥마을", index: 1 },
    { lat: 37.5744, lng: 126.9856, name: "인사동", index: 2 },
    { lat: 37.5697, lng: 126.9996, name: "광장시장", index: 3 },
  ],
};

export default function FocusPage() {
  const router = useRouter();
  const { duration } = useOnboardingStore();

  useEffect(() => {
    if (!duration) router.replace("/browse");
  }, [duration, router]);

  if (!duration) return null;

  const mapData = {
    center: MOCK_ITINERARY.center,
    markers: MOCK_ITINERARY.markers,
    zoom: 13,
  };

  const panelData = {
    duration,
    itinerary: MOCK_ITINERARY.itinerary,
  };

  return (
    <div className="flex h-screen bg-black overflow-hidden">
      {/* 지도 — 60% */}
      <div className="w-[60%] h-full">
        <MapView id="focus-map" meta={{}} data={mapData} />
      </div>

      {/* 일정 패널 — 40% */}
      <div className="w-[40%] h-full flex flex-col bg-gray-950 border-l border-gray-800">
        <div className="px-5 py-4 border-b border-gray-800 flex items-center justify-between flex-shrink-0">
          <div>
            <h2 className="text-white font-bold text-lg">AI 추천 여행 일정</h2>
            <p className="text-gray-400 text-xs mt-0.5">GraphRAG 기반 맞춤 추천</p>
          </div>
          <button
            onClick={() => router.push("/my-list")}
            className="text-gray-400 hover:text-white text-sm transition-colors"
          >
            ← 요약
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <ItineraryPanel id="focus-panel" meta={{}} data={panelData} />
        </div>
      </div>
    </div>
  );
}
