'use client';
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useOnboardingStore } from "@/store/onboarding-store";

const DURATION_LABEL: Record<string, string> = {
  day: "당일치기",
  onenight: "1박 2일",
  twonight: "2박 3일",
};

const PURPOSE_LABEL: Record<string, string> = {
  food: "맛집 탐방",
  kculture: "K-컬처",
  nature: "자연 힐링",
  history: "역사 문화",
  shopping: "쇼핑",
  rest: "휴식",
};

export default function MyListPage() {
  const router = useRouter();
  const { duration, selectedArtists, selectedRegions, purposes, budget } =
    useOnboardingStore();

  useEffect(() => {
    if (!duration) router.replace("/browse");
  }, [duration, router]);

  if (!duration) return null;

  return (
    <div className="min-h-screen bg-black text-white px-6 py-10 flex flex-col gap-8 max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold">나의 여행 요약</h1>

      {/* 여행 기간 */}
      <section className="bg-gray-900 rounded-2xl p-5 flex flex-col gap-2">
        <p className="text-gray-400 text-xs uppercase tracking-widest">여행 기간</p>
        <p className="text-xl font-semibold">{DURATION_LABEL[duration]}</p>
      </section>

      {/* 좋아하는 아티스트 */}
      <section className="bg-gray-900 rounded-2xl p-5 flex flex-col gap-3">
        <p className="text-gray-400 text-xs uppercase tracking-widest">선택한 아티스트</p>
        <div className="flex flex-wrap gap-2">
          {selectedArtists.map((a) => (
            <span key={a.id} className="px-3 py-1 bg-red-900 text-red-300 rounded-full text-sm font-medium">
              {a.name}
            </span>
          ))}
        </div>
      </section>

      {/* 여행 지역 */}
      <section className="bg-gray-900 rounded-2xl p-5 flex flex-col gap-3">
        <p className="text-gray-400 text-xs uppercase tracking-widest">선택한 지역</p>
        <div className="flex flex-wrap gap-2">
          {selectedRegions.map((r) => (
            <span key={r.id} className="px-3 py-1 bg-gray-700 text-gray-200 rounded-full text-sm font-medium">
              {r.name}
            </span>
          ))}
        </div>
      </section>

      {/* 여행 목적 */}
      <section className="bg-gray-900 rounded-2xl p-5 flex flex-col gap-3">
        <p className="text-gray-400 text-xs uppercase tracking-widest">여행 목적</p>
        <div className="flex flex-wrap gap-2">
          {purposes.map((p) => (
            <span key={p} className="px-3 py-1 bg-gray-700 text-gray-200 rounded-full text-sm font-medium">
              {PURPOSE_LABEL[p]}
            </span>
          ))}
        </div>
      </section>

      {/* 예산 */}
      <section className="bg-gray-900 rounded-2xl p-5 flex flex-col gap-2">
        <p className="text-gray-400 text-xs uppercase tracking-widest">여행 예산</p>
        <p className="text-xl font-semibold">
          ₩{budget.min.toLocaleString("ko-KR")} ~ ₩{budget.max.toLocaleString("ko-KR")}
        </p>
      </section>

      {/* AI 추천 배너 */}
      <button
        onClick={() => router.push("/focus")}
        className="w-full py-6 rounded-2xl bg-gradient-to-r from-red-700 to-red-500 flex flex-col items-center gap-2 hover:from-red-600 hover:to-red-400 transition-all shadow-lg shadow-red-900/40"
      >
        <span className="text-2xl font-bold">✨ AI 여행 일정 보기</span>
        <span className="text-red-200 text-sm">GraphRAG 기반 맞춤 추천 →</span>
      </button>
    </div>
  );
}
