'use client';
import CardImage from "./atoms/CardImage";
import CardLabel from "./atoms/CardLabel";
import CheckIndicator from "./atoms/CheckIndicator";
import { useOnboardingStore } from "@/store/onboarding-store";

interface Props {
  id: string;
  meta: any;
  data: any;
  onAction?: (meta: any, data?: any) => void;
}
// 복합컴포넌트 
export default function SelectionCard({ id, meta, data, onAction }: Props) {
  const { selectedArtists, selectedRegions, toggleArtist, toggleRegion } =
    useOnboardingStore();

  const mode = meta?.cssClass?.includes("circle") ? "circle" : "square";
  const isArtist = mode === "circle";
  const selected = isArtist
    ? selectedArtists.some((a) => a.id === data?.id)
    : selectedRegions.some((r) => r.id === data?.id);

  const maxReached = isArtist
    ? selectedArtists.length >= 5
    : selectedRegions.length >= 5;
  const disabled = maxReached && !selected;

  const handleClick = () => {
    if (disabled) return;
    if (isArtist) {
      toggleArtist(data);
    } else {
      toggleRegion(data);
    }
    onAction?.(meta, data);
  };

  return (
    <div
      className={`selection-card relative flex flex-col items-center gap-1 cursor-pointer transition-opacity ${
        disabled ? "opacity-40 cursor-not-allowed" : ""
      }`}
      onClick={handleClick}
    >
      <div className={`relative w-24 h-24 ${mode === "circle" ? "rounded-full" : "rounded-lg"} overflow-hidden`}>
        <CardImage id={id} meta={{ ...meta, cssClass: mode }} data={data} />
        <CheckIndicator id={id} meta={meta} data={data} selected={selected} />
      </div>
      <CardLabel id={id} meta={meta} data={data} />
    </div>
  );
}
