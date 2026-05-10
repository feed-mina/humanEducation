/**
 * MetadataProvider — 화면 메타데이터 Context Provider
 * 
 * JSON 파일에서 메타데이터를 로드합니다.
 * DB 전환 시 queryFn의 fetch URL만 변경하면 됩니다:
 *   JSON: /api/ui/${screenId} → 로컬 JSON 파일 읽기
 *   DB:   /api/ui/${screenId} → Spring Boot API 프록시
 */
"use client";

import React, { createContext, useContext, useMemo, ReactNode } from "react";
import { usePathname } from "next/navigation";
import { SCREEN_MAP, DEFAULT_SCREEN_ID } from "./screenMap";
import { Metadata, ScreenMetadataResponse } from "./type";
import { useQuery } from "@tanstack/react-query";

interface MetadataContextType {
  menuTree: Metadata[];
  isLoading: boolean;
  screenId: string;
  error: Error | null;
}

const MetadataContext = createContext<MetadataContextType | undefined>(undefined);

interface MetadataProviderProps {
  children: ReactNode;
  screenId?: string; // 직접 주입 (선택)
}

export function MetadataProvider({ children, screenId: propScreenId }: MetadataProviderProps) {
  const pathname = usePathname();

  // screenId 결정: 직접 주입 > URL 매핑 > 기본값
  const finalScreenId = useMemo(() => {
    if (propScreenId) return propScreenId;
    return SCREEN_MAP[pathname] || DEFAULT_SCREEN_ID;
  }, [propScreenId, pathname]);

  // 메타데이터 가져오기
  // DB 전환 시 이 queryFn만 수정하면 됩니다
  const { data, isLoading, error } = useQuery<Metadata[]>({
    queryKey: ["metadata", finalScreenId],
    queryFn: async () => {
      const res = await fetch(`/api/ui/${finalScreenId}`);
      if (!res.ok) throw new Error(`메타데이터 로드 실패: ${finalScreenId}`);
      const result: ScreenMetadataResponse = await res.json();
      return result.data || [];
    },
    staleTime: 1000 * 60 * 5, // 5분 캐시
    enabled: !!finalScreenId,
  });

  const contextValue = useMemo(() => ({
    menuTree: data || [],
    isLoading,
    screenId: finalScreenId,
    error: error as Error | null,
  }), [data, isLoading, finalScreenId, error]);

  return (
    <MetadataContext.Provider value={contextValue}>
      {children}
    </MetadataContext.Provider>
  );
}

export function useMetadata() {
  const ctx = useContext(MetadataContext);
  if (!ctx) throw new Error("useMetadata는 MetadataProvider 안에서 사용되어야 합니다.");
  return ctx;
}
