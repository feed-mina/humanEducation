'use client';
import { useOnboardingStore } from "@/store/onboarding-store";
import type { TravelDuration } from "@/store/onboarding-store";

const LABEL_TO_VALUE: Record<string, TravelDuration> = {
  "당일치기": "day",
  "1박 2일": "onenight",
  "2박 3일": "twonight",
};

interface Props {
  id: string;
  meta: any;
  data: any;
  onAction?: (meta: any, data?: any) => void;
}

export default function DurationButton({ id, meta, data, onAction }: Props) {
  const { duration } = useOnboardingStore();
  const label = meta?.labelText || meta?.label_text || data?.label || "";
  const value = LABEL_TO_VALUE[label] || label;
  const isSelected = duration === value;

  const handleClick = () => {
    onAction?.(meta, { value });
  };

  return (
    <button
      id={id}
      onClick={handleClick}
      className={`duration-btn px-8 py-4 text-lg font-bold rounded-full border-2 transition-all
        ${isSelected
          ? "bg-red-600 border-red-600 text-white"
          : "bg-transparent border-red-600 text-red-500 hover:bg-red-600 hover:text-white"
        }`}
    >
      {label}
    </button>
  );
}
