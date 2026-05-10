'use client';
import { useRouter } from "next/navigation";
import DynamicEngine from "@/components/dynamic-engine/DynamicEngine";
import { usePageHook } from "@/components/dynamic-engine/hooks/usePageHook";
import { SCREEN_IDS } from "@/components/dynamic-engine/screenMap";
import { Metadata } from "@/components/dynamic-engine/type";
import { useOnboardingStore, TravelPurpose } from "@/store/onboarding-store";

const PURPOSE_ITEMS: { key: TravelPurpose; label: string }[] = [
  { key: "food",     label: "맛집 탐방" },
  { key: "kculture", label: "K-컬처" },
  { key: "nature",   label: "자연 힐링" },
  { key: "history",  label: "역사 문화" },
  { key: "shopping", label: "쇼핑" },
  { key: "rest",     label: "휴식" },
];

const buildMetadata = (): Metadata[] => [
  {
    componentId: "intro4_root",
    component_id: "intro4_root",
    componentType: "GROUP",
    groupId: "intro4_root",
    group_id: "intro4_root",
    parentGroupId: null,
    groupDirection: "COLUMN",
    cssClass: "min-h-screen bg-black flex flex-col px-6 py-10 gap-6",
    children: [
      {
        componentId: "intro4_title",
        component_id: "intro4_title",
        componentType: "TEXT",
        parentGroupId: "intro4_root",
        cssClass: "text-2xl font-bold text-white",
        labelText: "여행 목적을 알려주세요",
        children: null,
      },
      {
        componentId: "intro4_sub",
        component_id: "intro4_sub",
        componentType: "TEXT",
        parentGroupId: "intro4_root",
        cssClass: "text-gray-400 text-sm",
        labelText: "복수 선택 가능해요",
        children: null,
      },
      {
        componentId: "purpose_grid",
        component_id: "purpose_grid",
        componentType: "GROUP",
        groupId: "purpose_grid",
        group_id: "purpose_grid",
        parentGroupId: "intro4_root",
        groupDirection: "COLUMN",
        cssClass: "flex flex-col gap-3 pb-24",
        refDataId: "purposeList",
        ref_data_id: "purposeList",
        children: [
          {
            componentId: "purpose_card",
            component_id: "purpose_card",
            componentType: "PURPOSE_CARD",
            parentGroupId: "purpose_grid",
            children: null,
          },
        ],
      },
    ],
  },
];

export default function Intro4Page() {
  const router = useRouter();
  const { purposes } = useOnboardingStore();
  const metadata = buildMetadata();

  const { formData, handleChange, handleAction } = usePageHook(
    SCREEN_IDS.INTRO4,
    metadata,
    {}
  );

  const purposePageData = PURPOSE_ITEMS.map((p) => ({
    purposeKey: p.key,
    name: p.label,
  }));

  return (
    <div className="min-h-screen bg-black flex flex-col">
      <div className="flex-1">
        <DynamicEngine
          metadata={metadata}
          screenId={SCREEN_IDS.INTRO4}
          pageData={{ purposeList: purposePageData }}
          formData={formData}
          onChange={handleChange}
          onAction={handleAction}
        />
      </div>

      <div className="fixed bottom-0 left-0 right-0 bg-black border-t border-gray-800 px-6 py-4 flex items-center justify-between z-50">
        <span className="text-gray-400 text-sm">
          {purposes.length}개 선택됨
        </span>
        <button
          onClick={() => router.push("/intro5")}
          disabled={purposes.length < 1}
          className="px-8 py-3 bg-red-600 text-white font-bold rounded-full disabled:opacity-40 disabled:cursor-not-allowed hover:bg-red-700 transition-colors"
        >
          다음 →
        </button>
      </div>
    </div>
  );
}
