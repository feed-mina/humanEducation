'use client';
import PurposeIcon from "./atoms/PurposeIcon";
import { useOnboardingStore, TravelPurpose } from "@/store/onboarding-store";

const PURPOSE_LABELS: Record<TravelPurpose, string> = {
  food: "맛집 탐방",
  kculture: "K-컬처",
  nature: "자연 힐링",
  history: "역사 문화",
  shopping: "쇼핑",
  rest: "휴식",
};

interface Props {
  id: string;
  meta: any;
  data: any;
  onAction?: (meta: any, data?: any) => void;
}

export default function PurposeCard({ id, meta, data, onAction }: Props) {
  const { purposes, togglePurpose } = useOnboardingStore();
  const purposeKey = (data?.purposeKey || meta?.cssClass || "") as TravelPurpose;
  const label = PURPOSE_LABELS[purposeKey] || meta?.labelText || "";
  const selected = purposes.includes(purposeKey);

  const handleClick = () => {
    togglePurpose(purposeKey);
    onAction?.(meta, { value: purposeKey });
  };

  return (
    <button
      id={id}
      onClick={handleClick}
      className={`purpose-card flex items-center gap-3 px-5 py-4 rounded-xl border-2 w-full transition-all
        ${selected
          ? "bg-red-900 border-red-600 text-white"
          : "bg-gray-900 border-gray-700 text-gray-300 hover:border-red-600"
        }`}
    >
      <PurposeIcon id={id} meta={{ ...meta, cssClass: purposeKey }} data={data} />
      <span className="font-medium text-base">{label}</span>
    </button>
  );
}
