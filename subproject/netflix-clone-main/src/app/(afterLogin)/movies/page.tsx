'use client';
import { useRouter } from "next/navigation";
import DynamicEngine from "@/components/dynamic-engine/DynamicEngine";
import { usePageHook } from "@/components/dynamic-engine/hooks/usePageHook";
import { SCREEN_IDS } from "@/components/dynamic-engine/screenMap";
import { Metadata } from "@/components/dynamic-engine/type";
import { useOnboardingStore } from "@/store/onboarding-store";

const ARTIST_LIST = [
  { id: 1,  name: "BTS",       imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 2,  name: "블랙핑크",   imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 3,  name: "뉴진스",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 4,  name: "아이브",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 5,  name: "세븐틴",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 6,  name: "에스파",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 7,  name: "엔믹스",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 8,  name: "스트레이키즈", imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 9,  name: "트와이스",   imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 10, name: "레드벨벳",   imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 11, name: "GOT7",      imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 12, name: "2PM",       imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 13, name: "샤이니",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 14, name: "소녀시대",   imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 15, name: "엑소",       imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 16, name: "몬스타엑스", imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 17, name: "있지",       imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 18, name: "르세라핌",   imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 19, name: "케플러",     imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
  { id: 20, name: "제로베이스원", imageUrl: "https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg" },
];

const buildMetadata = (): Metadata[] => [
  {
    componentId: "intro2_root",
    component_id: "intro2_root",
    componentType: "GROUP",
    groupId: "intro2_root",
    group_id: "intro2_root",
    parentGroupId: null,
    groupDirection: "COLUMN",
    cssClass: "min-h-screen bg-black flex flex-col px-6 py-10 gap-6",
    children: [
      {
        componentId: "intro2_title",
        component_id: "intro2_title",
        componentType: "TEXT",
        parentGroupId: "intro2_root",
        cssClass: "text-2xl font-bold text-white",
        labelText: "좋아하는 아이돌/배우를 선택해주세요",
        children: null,
      },
      {
        componentId: "intro2_sub",
        component_id: "intro2_sub",
        componentType: "TEXT",
        parentGroupId: "intro2_root",
        cssClass: "text-gray-400 text-sm",
        labelText: "최대 5명까지 선택할 수 있어요",
        children: null,
      },
      {
        componentId: "artist_grid",
        component_id: "artist_grid",
        componentType: "GROUP",
        groupId: "artist_grid",
        group_id: "artist_grid",
        parentGroupId: "intro2_root",
        groupDirection: "ROW",
        cssClass: "grid grid-cols-4 gap-4 pb-24",
        refDataId: "artistList",
        ref_data_id: "artistList",
        children: [
          {
            componentId: "artist_card",
            component_id: "artist_card",
            componentType: "SELECTION_CARD",
            parentGroupId: "artist_grid",
            cssClass: "circle",
            children: null,
          },
        ],
      },
    ],
  },
];

export default function MoviesPage() {
  const router = useRouter();
  const { selectedArtists } = useOnboardingStore();
  const metadata = buildMetadata();

  const { formData, handleChange, handleAction } = usePageHook(
    SCREEN_IDS.INTRO2,
    metadata,
    {}
  );

  return (
    <div className="min-h-screen bg-black flex flex-col">
      <div className="flex-1">
        <DynamicEngine
          metadata={metadata}
          screenId={SCREEN_IDS.INTRO2}
          pageData={{ artistList: ARTIST_LIST }}
          formData={formData}
          onChange={handleChange}
          onAction={handleAction}
        />
      </div>

      <div className="fixed bottom-0 left-0 right-0 bg-black border-t border-gray-800 px-6 py-4 flex items-center justify-between z-50">
        <span className="text-gray-400 text-sm">
          {selectedArtists.length} / 5 선택됨
        </span>
        <button
          onClick={() => router.push("/latest")}
          disabled={selectedArtists.length < 1}
          className="px-8 py-3 bg-red-600 text-white font-bold rounded-full disabled:opacity-40 disabled:cursor-not-allowed hover:bg-red-700 transition-colors"
        >
          다음 →
        </button>
      </div>
    </div>
  );
}
