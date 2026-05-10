'use client';
import { useRouter } from "next/navigation";
import DynamicEngine from "@/engine/DynamicEngine";
import { usePageHook } from "@/engine/hooks/usePageHook";
import { useUiScreen } from "@/engine/hooks/useUiScreen";
import { SCREEN_IDS } from "@/engine/screenMap";

export default function Intro5Page() {
  const router = useRouter();
  const { data: metadata = [], isLoading } = useUiScreen(SCREEN_IDS.INTRO5);
  const { formData, handleChange, handleAction } = usePageHook(SCREEN_IDS.INTRO5, metadata, {});

  return (
    <div className="min-h-screen bg-black flex flex-col">
      <div className="flex-1">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-white">로딩 중...</div>
          </div>
        ) : (
          <DynamicEngine
            metadata={metadata}
            screenId={SCREEN_IDS.INTRO5}
            pageData={{}}
            formData={formData}
            onChange={handleChange}
            onAction={handleAction}
          />
        )}
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
