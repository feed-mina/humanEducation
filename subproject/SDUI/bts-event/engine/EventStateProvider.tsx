/**
 * EventStateProvider — 이벤트 페이지 전역 상태 관리
 * 
 * bts-event의 컴포넌트들은 lang, tab, modal 상태 등을
 * 부모에서 공유하므로, SDUI 엔진에서 이를 Context로 관리합니다.
 * 
 * DynamicEngine이 컴포넌트를 렌더링할 때 이 Context에서
 * 필요한 상태를 주입합니다.
 */
"use client";

import React, { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { type Lang } from "@/components/LangToggle";
import { type Layer } from "@/components/Map/LayerFilter";

type Tab = "map" | "chat" | "board";

interface EventState {
  // 언어
  lang: Lang;
  setLang: (lang: Lang) => void;
  // 탭
  tab: Tab;
  setTab: (tab: Tab) => void;
  // 레이어 필터
  activeLayer: Layer | null;
  setActiveLayer: (layer: Layer | null) => void;
  // 모달
  showNotice: boolean;
  setShowNotice: (v: boolean) => void;
  showCheer: boolean;
  setShowCheer: (v: boolean) => void;
  showSupport: boolean;
  setShowSupport: (v: boolean) => void;
}

const EventStateContext = createContext<EventState | null>(null);

export function EventStateProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("ko");
  const [tab, setTab] = useState<Tab>("map");
  const [activeLayer, setActiveLayer] = useState<Layer | null>("subway");
  const [showNotice, setShowNotice] = useState(false);
  const [showCheer, setShowCheer] = useState(false);
  const [showSupport, setShowSupport] = useState(false);

  return (
    <EventStateContext.Provider
      value={{
        lang, setLang,
        tab, setTab,
        activeLayer, setActiveLayer,
        showNotice, setShowNotice,
        showCheer, setShowCheer,
        showSupport, setShowSupport,
      }}
    >
      {children}
    </EventStateContext.Provider>
  );
}

export function useEventState() {
  const ctx = useContext(EventStateContext);
  if (!ctx) throw new Error("useEventState는 EventStateProvider 안에서 사용되어야 합니다.");
  return ctx;
}
