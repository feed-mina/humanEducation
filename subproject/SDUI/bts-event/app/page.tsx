/**
 * BTS 이벤트 메인 페이지 — SDUI 엔진 기반
 * 
 * 기존 page.tsx의 하드코딩 UI 조합을 DynamicEngine으로 대체합니다.
 * UI 구조는 data/screens/BTS_EVENT_MAIN.json에서 정의됩니다.
 * 
 * 기존 코드는 page.original.tsx에 백업되어 있습니다.
 */
"use client";

import { Suspense, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { DynamicEngine, MetadataProvider, EventStateProvider, useEventState } from "@/engine";
import { translations } from "@/data/translations";
import { Bell, Heart } from "lucide-react";

/**
 * 페이지 초기화 로직 (딥링크, 자동 공지 등)
 * EventStateProvider 안에서만 동작합니다.
 */
function PageInitializer() {
  const searchParams = useSearchParams();
  const { setTab, setShowNotice, lang } = useEventState();

  // 딥링크: ?tab=chat 등
  useEffect(() => {
    const t = searchParams.get("tab") as "map" | "chat" | "board";
    if (t && ["map", "chat", "board"].includes(t)) {
      setTab(t);
    }
  }, [searchParams, setTab]);

  // 첫 방문 시 공지 자동 표시
  useEffect(() => {
    if (!searchParams.get("tab")) {
      const timer = setTimeout(() => setShowNotice(true), 1500);
      return () => clearTimeout(timer);
    }
  }, [searchParams, setShowNotice]);

  return null;
}

/**
 * 헤더 컴포넌트 — DynamicEngine과 별도로 렌더링
 * (탭 바, 언어 토글, 후원 버튼은 상태와 깊이 연결되어 있어서
 *  메타데이터 기반보다 직접 렌더링이 안정적입니다)
 */
function AppHeader() {
  const { lang, setLang, tab, setTab, setShowSupport } = useEventState();
  const t = translations[lang];

  const TABS: { key: "map" | "chat" | "board"; label: string }[] = [
    { key: "map", label: t.map },
    { key: "chat", label: t.chat },
    { key: "board", label: t.board },
  ];

  const LANGS = [
    { key: "ko" as const, label: "🇰🇷" },
    { key: "en" as const, label: "🇺🇸" },
    { key: "ja" as const, label: "🇯🇵" },
  ];

  return (
    <header className="app-header">
      <h1>{t.title}</h1>
      <div className="flex items-center gap-2">
        <div className="tab-bar">
          {TABS.map(({ key, label }) => (
            <button
              key={key}
              className={`tab-btn ${tab === key ? "active" : ""}`}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {LANGS.map(({ key, label }) => (
            <button
              key={key}
              className={`tab-btn px-2 py-1 text-xs ${lang === key ? "active" : ""}`}
              onClick={() => setLang(key)}
              title={key.toUpperCase()}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowSupport(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-pink-100 text-pink-500 rounded-full text-xs font-bold hover:bg-pink-200 transition-all active:scale-95"
        >
          <Heart size={14} fill="currentColor" />
          {lang === "ko" ? "커피 후원" : lang === "ja" ? "応援" : "Support"}
        </button>
      </div>
    </header>
  );
}

/**
 * 플로팅 공지 버튼 — 지도 탭에서만 표시
 */
function FloatingNoticeButton() {
  const { tab, setShowNotice, lang } = useEventState();
  const t = translations[lang];

  if (tab !== "map") return null;

  return (
    <button
      className="floating-notice-btn"
      onClick={() => setShowNotice(true)}
    >
      <Bell size={14} fill="white" />
      {t.traffic}
    </button>
  );
}

/**
 * 메인 콘텐츠 — DynamicEngine이 JSON 메타데이터에서 컴포넌트를 렌더링
 */
function MainContent() {
  return (
    <MetadataProvider screenId="BTS_EVENT_MAIN">
      <DynamicEngine />
    </MetadataProvider>
  );
}

function HomePageContent() {
  return (
    <EventStateProvider>
      <PageInitializer />
      <AppHeader />
      <MainContent />
    </EventStateProvider>
  );
}

export default function HomePage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-screen bg-[#1a1a2e] text-white">
          💜 로딩 중...
        </div>
      }
    >
      <HomePageContent />
    </Suspense>
  );
}
