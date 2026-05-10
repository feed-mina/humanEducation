import { create } from "zustand";
import { persist } from "zustand/middleware";
// 온보딩 페이지에 사용함

// 여행기간 (당일치기, 1박2일, 2박3일)
export type TravelDuration = "day" | "onenight" | "twonight";
// 여행 목적
export type TravelPurpose =
  | "food"
  | "kculture"
  | "nature"
  | "history"
  | "shopping"
  | "rest";


export interface ContentItem {
  id: number;
  name: string;
  imageUrl: string;
}
// 여행비용
export interface BudgetRange {
  min: number;
  max: number;
}
// 온보딩 상태
interface OnboardingState {
  duration: TravelDuration | null;
  selectedArtists: ContentItem[];
  selectedRegions: ContentItem[];
  purposes: TravelPurpose[];
  budget: BudgetRange;
  setDuration: (d: TravelDuration) => void;
  toggleArtist: (item: ContentItem) => void;
  toggleRegion: (item: ContentItem) => void;
  togglePurpose: (p: TravelPurpose) => void;
  setBudget: (range: BudgetRange) => void;
  reset: () => void;
}

const DEFAULT_BUDGET: BudgetRange = { min: 30000, max: 2000000 };

export const useOnboardingStore = create<OnboardingState>()(
  persist(
    (set) => ({
      duration: null,
      selectedArtists: [],
      selectedRegions: [],
      purposes: [],
      budget: DEFAULT_BUDGET,
// 여행기간 정하는 상태
      setDuration: (d) => set({ duration: d }),
// 좋아하는 아티스트 고르기 
      toggleArtist: (item) =>
        set((state) => {
          const exists = state.selectedArtists.some((a) => a.id === item.id);
          if (exists) {
            return {
              selectedArtists: state.selectedArtists.filter((a) => a.id !== item.id),
            };
          }
          if (state.selectedArtists.length >= 5) return state;
          return { selectedArtists: [...state.selectedArtists, item] };
        }),
// 가고싶은 여행지 고르기 
      toggleRegion: (item) =>
        set((state) => {
          const exists = state.selectedRegions.some((r) => r.id === item.id);
          if (exists) {
            return {
              selectedRegions: state.selectedRegions.filter((r) => r.id !== item.id),
            };
          }
          if (state.selectedRegions.length >= 5) return state;
          return { selectedRegions: [...state.selectedRegions, item] };
        }),
// 여행을 가는 이유 고르기
      togglePurpose: (p) =>
        set((state) => {
          const exists = state.purposes.includes(p);
          return {
            purposes: exists
              ? state.purposes.filter((x) => x !== p)
              : [...state.purposes, p],
          };
        }),

      setBudget: (range) => set({ budget: range }),

      reset: () =>
        set({
          duration: null,
          selectedArtists: [],
          selectedRegions: [],
          purposes: [],
          budget: DEFAULT_BUDGET,
        }),
    }),
    { name: "kride-onboarding" }
  )
);
