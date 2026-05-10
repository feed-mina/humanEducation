'use client';
import { useRouter } from "next/navigation";
import DynamicEngine from "@/components/dynamic-engine/DynamicEngine";
import { usePageHook } from "@/components/dynamic-engine/hooks/usePageHook";
import { SCREEN_IDS } from "@/components/dynamic-engine/screenMap";
import { Metadata } from "@/components/dynamic-engine/type";

const INTRO5_METADATA: Metadata[] = [
  {
    componentId: "intro5_root",
    component_id: "intro5_root",
    componentType: "GROUP",
    groupId: "intro5_root",
    group_id: "intro5_root",
    parentGroupId: null,
    groupDirection: "COLUMN",
    cssClass: "min-h-screen bg-black flex flex-col items-center justify-center px-8 gap-10",
    children: [
      {
        componentId: "intro5_title",
        component_id: "intro5_title",
        componentType: "TEXT",
        parentGroupId: "intro5_root",
        cssClass: "text-2xl font-bold text-white text-center",
        labelText: "여행 예산을 설정해주세요",
        children: null,
      },
      {
        componentId: "intro5_sub",
        component_id: "intro5_sub",
        componentType: "TEXT",
        parentGroupId: "intro5_root",
        cssClass: "text-gray-400 text-sm text-center",
        labelText: "1인 기준 총 여행 경비예요",
        children: null,
      },
      {
        componentId: "budget_slider",
        component_id: "budget_slider",
        componentType: "DUAL_RANGE_SLIDER",
        parentGroupId: "intro5_root",
        cssClass: "w-full max-w-md",
        children: null,
      },
    ],
  },
];

export default function Intro5Page() {
  const router = useRouter();
  const { formData, handleChange, handleAction } = usePageHook(
    SCREEN_IDS.INTRO5,
    INTRO5_METADATA,
    {}
  );

  return (
    <div className="min-h-screen bg-black flex flex-col">
      <div className="flex-1">
        <DynamicEngine
          metadata={INTRO5_METADATA}
          screenId={SCREEN_IDS.INTRO5}
          pageData={{}}
          formData={formData}
          onChange={handleChange}
          onAction={handleAction}
        />
      </div>

      <div className="fixed bottom-0 left-0 right-0 bg-black border-t border-gray-800 px-6 py-6 z-50">
        <button
          onClick={() => router.push("/my-list")}
          className="w-full py-4 bg-red-600 text-white font-bold text-lg rounded-full hover:bg-red-700 transition-colors"
        >
          AI 여행 추천 받기 →
        </button>
      </div>
    </div>
  );
}
