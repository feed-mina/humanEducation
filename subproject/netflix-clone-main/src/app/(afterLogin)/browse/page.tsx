'use client';
import DynamicEngine from "@/components/dynamic-engine/DynamicEngine";
import { usePageHook } from "@/components/dynamic-engine/hooks/usePageHook";
import { SCREEN_IDS } from "@/components/dynamic-engine/screenMap";
import { Metadata } from "@/components/dynamic-engine/type";

const INTRO1_METADATA: Metadata[] = [
  {
    componentId: "intro1_root",
    component_id: "intro1_root",
    componentType: "GROUP",
    groupId: "intro1_root",
    group_id: "intro1_root",
    parentGroupId: null,
    groupDirection: "COLUMN",
    cssClass: "min-h-screen bg-black flex flex-col items-center justify-center gap-8 px-6",
    children: [
      {
        componentId: "intro1_title",
        component_id: "intro1_title",
        componentType: "TEXT",
        parentGroupId: "intro1_root",
        cssClass: "text-3xl font-bold text-white text-center",
        labelText: "어떤 여행을 떠나실 건가요?",
        children: null,
      },
      {
        componentId: "intro1_sub",
        component_id: "intro1_sub",
        componentType: "TEXT",
        parentGroupId: "intro1_root",
        cssClass: "text-gray-400 text-base text-center",
        labelText: "여행 기간을 선택해주세요",
        children: null,
      },
      {
        componentId: "intro1_buttons",
        component_id: "intro1_buttons",
        componentType: "GROUP",
        groupId: "intro1_buttons",
        group_id: "intro1_buttons",
        parentGroupId: "intro1_root",
        groupDirection: "COLUMN",
        cssClass: "flex flex-col gap-4 w-full max-w-xs",
        children: [
          {
            componentId: "btn_day",
            component_id: "btn_day",
            componentType: "DURATION_BUTTON",
            parentGroupId: "intro1_buttons",
            actionType: "SET_DURATION",
            labelText: "당일치기",
            children: null,
          },
          {
            componentId: "btn_1n2d",
            component_id: "btn_1n2d",
            componentType: "DURATION_BUTTON",
            parentGroupId: "intro1_buttons",
            actionType: "SET_DURATION",
            labelText: "1박 2일",
            children: null,
          },
          {
            componentId: "btn_2n3d",
            component_id: "btn_2n3d",
            componentType: "DURATION_BUTTON",
            parentGroupId: "intro1_buttons",
            actionType: "SET_DURATION",
            labelText: "2박 3일",
            children: null,
          },
        ],
      },
    ],
  },
];

export default function BrowsePage() {
  const { formData, handleChange, handleAction } = usePageHook(
    SCREEN_IDS.INTRO1,
    INTRO1_METADATA,
    {}
  );

  return (
    <DynamicEngine
      metadata={INTRO1_METADATA}
      screenId={SCREEN_IDS.INTRO1}
      pageData={{}}
      formData={formData}
      onChange={handleChange}
      onAction={handleAction}
    />
  );
}
