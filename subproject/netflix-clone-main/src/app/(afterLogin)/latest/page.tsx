'use client';
import { useRouter } from "next/navigation";
import DynamicEngine from "@/components/dynamic-engine/DynamicEngine";
import { usePageHook } from "@/components/dynamic-engine/hooks/usePageHook";
import { SCREEN_IDS } from "@/components/dynamic-engine/screenMap";
import { Metadata } from "@/components/dynamic-engine/type";
import { useOnboardingStore } from "@/store/onboarding-store";

const REGION_LIST = [
  { id: 1,  name: "서울",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 2,  name: "경기",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 3,  name: "강원",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 4,  name: "충북",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 5,  name: "충남",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 6,  name: "경북",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 7,  name: "경남",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 8,  name: "전북",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 9,  name: "전남",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 10, name: "제주",  imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
];

const buildMetadata = (): Metadata[] => [
  {
    componentId: "intro3_root",
    component_id: "intro3_root",
    componentType: "GROUP",
    groupId: "intro3_root",
    group_id: "intro3_root",
    parentGroupId: null,
    groupDirection: "COLUMN",
    cssClass: "min-h-screen bg-black flex flex-col px-6 py-10 gap-6",
    children: [
      {
        componentId: "intro3_title",
        component_id: "intro3_title",
        componentType: "TEXT",
        parentGroupId: "intro3_root",
        cssClass: "text-2xl font-bold text-white",
        labelText: "어느 지역을 여행하고 싶으신가요?",
        children: null,
      },
      {
        componentId: "intro3_sub",
        component_id: "intro3_sub",
        componentType: "TEXT",
        parentGroupId: "intro3_root",
        cssClass: "text-gray-400 text-sm",
        labelText: "최대 5곳까지 선택할 수 있어요",
        children: null,
      },
      {
        componentId: "region_grid",
        component_id: "region_grid",
        componentType: "GROUP",
        groupId: "region_grid",
        group_id: "region_grid",
        parentGroupId: "intro3_root",
        groupDirection: "ROW",
        cssClass: "grid grid-cols-4 gap-4 pb-24",
        refDataId: "regionList",
        ref_data_id: "regionList",
        children: [
          {
            componentId: "region_card",
            component_id: "region_card",
            componentType: "SELECTION_CARD",
            parentGroupId: "region_grid",
            cssClass: "square",
            children: null,
          },
        ],
      },
    ],
  },
];

export default function LatestPage() {
  const router = useRouter();
  const { selectedRegions } = useOnboardingStore();
  const metadata = buildMetadata();

  const { formData, handleChange, handleAction } = usePageHook(
    SCREEN_IDS.INTRO3,
    metadata,
    {}
  );

  return (
    <div className="min-h-screen bg-black flex flex-col">
      <div className="flex-1">
        <DynamicEngine
          metadata={metadata}
          screenId={SCREEN_IDS.INTRO3}
          pageData={{ regionList: REGION_LIST }}
          formData={formData}
          onChange={handleChange}
          onAction={handleAction}
        />
      </div>

      <div className="fixed bottom-0 left-0 right-0 bg-black border-t border-gray-800 px-6 py-4 flex items-center justify-between z-50">
        <span className="text-gray-400 text-sm">
          {selectedRegions.length} / 5 선택됨
        </span>
        <button
          onClick={() => router.push("/intro4")}
          disabled={selectedRegions.length < 1}
          className="px-8 py-3 bg-red-600 text-white font-bold rounded-full disabled:opacity-40 disabled:cursor-not-allowed hover:bg-red-700 transition-colors"
        >
          다음 →
        </button>
      </div>
    </div>
  );
}
