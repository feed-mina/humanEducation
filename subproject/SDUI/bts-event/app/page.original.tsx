"use client";

import dynamic from "next/dynamic";
import { useState, useEffect } from "react";
import LayerFilter, { type Layer } from "@/components/Map/LayerFilter";
import InfoPanel from "@/components/InfoPanel";
import LangToggle, { type Lang } from "@/components/LangToggle";
import GuestChat from "@/components/Chat/GuestChat";
import FanBoard from "@/components/Board/FanBoard";
import NoticeModal from "@/components/NoticeModal";
import StatusCard from "@/components/StatusCard";
import LivePip from "@/components/LivePip";
import CheerMode from "@/components/CheerMode";
import { Bell, Heart } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import SupportModal from "@/components/SupportModal";
import { translations } from "@/data/translations";

// LeafletMap must be loaded client-side only (no SSR)
const LeafletMap = dynamic(() => import("@/components/Map/LeafletMap"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-[#1a1a2e] text-white/50">
      🗺️ 지도 로딩 중...
    </div>
  ),
});

type Tab = "map" | "chat" | "board";

function HomePageContent() {
  const searchParams = useSearchParams();
  const [tab, setTab] = useState<Tab>("map");
  const [lang, setLang] = useState<Lang>("ko");
  const [activeLayer, setActiveLayer] = useState<Layer | null>("subway");
  const [showNotice, setShowNotice] = useState(false);
  const [showCheer, setShowCheer] = useState(false);
  const [showSupport, setShowSupport] = useState(false);

  // Handle Query Params for Deep Linking
  useEffect(() => {
    const t = searchParams.get("tab") as Tab;
    if (t && ["map", "chat", "board"].includes(t)) {
      setTab(t);
    }
  }, [searchParams]);

  // Auto-show notice on first load (only if no specific tab is requested)
  useEffect(() => {
    if (!searchParams.get("tab")) {
      const timer = setTimeout(() => setShowNotice(true), 1500);
      return () => clearTimeout(timer);
    }
  }, [searchParams]);

  const t = translations[lang];

  const TABS: { key: Tab; label: string }[] = [
    { key: "map",   label: t.map },
    { key: "chat",  label: t.chat },
    { key: "board", label: t.board },
  ];

  return (
    <div className="flex flex-col h-full relative font-sans">
      {/* Cheer Mode Overlay */}
      {showCheer && <CheerMode lang={lang} onClose={() => setShowCheer(false)} />}

      {/* Header */}
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
          <LangToggle lang={lang} onChange={setLang} />
          <button 
            onClick={() => setShowSupport(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-pink-100 text-pink-500 rounded-full text-xs font-bold hover:bg-pink-200 transition-all active:scale-95"
          >
            <Heart size={14} fill="currentColor" />
            {lang === 'ko' ? '커피 후원' : lang === 'ja' ? '応援' : 'Support'}
          </button>
        </div>
      </header>

      {/* Layer Filter (Sticky below Header on Map Tab) */}
      {tab === "map" && (
        <LayerFilter active={activeLayer} onSelect={setActiveLayer} lang={lang} />
      )}

      {/* Main content */}
      <main className="flex-1 overflow-hidden relative">
        {tab === "map" && (
          <div className="flex flex-col h-full">
            <div className="flex-1 overflow-hidden relative">
              <LeafletMap activeLayer={activeLayer} lang={lang} />
              
              {/* Floating UI on Map */}
              <StatusCard lang={lang} />
              <button 
                className="floating-notice-btn"
                onClick={() => setShowNotice(true)}
              >
                <Bell size={14} fill="white" />
                {t.traffic}
              </button>
              
              <LivePip />
            </div>
            <InfoPanel onCheer={() => setShowCheer(true)} lang={lang} />
          </div>
        )}

        {tab === "chat" && <GuestChat lang={lang} />}
        {tab === "board" && <FanBoard lang={lang} initialPostId={searchParams.get("id")} />}
      </main>

      {/* Modals */}
      {showNotice && (
        <NoticeModal 
          lang={lang} 
          onClose={() => setShowNotice(false)} 
          title={t.traffic}
        />
      )}

      {showSupport && (
        <SupportModal 
          lang={lang} 
          onClose={() => setShowSupport(false)} 
        />
      )}
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-screen bg-[#1a1a2e] text-white">
        💜 로딩 중...
      </div>
    }>
      <HomePageContent />
    </Suspense>
  )
}
