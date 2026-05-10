import { type Lang } from "../LangToggle";

const PAGE_URL = "https://bts-gwanghwamun.vercel.app";

const TEXTS = {
  ko: { label: "🟢 LINE 공유", text: "💜 BTS 광화문 현장 지도 — 24h 카페·충전·구급·지하철 루트" },
  en: { label: "🟢 Share on LINE", text: "💜 BTS Gwanghwamun Live Map — 24h Cafe, Charging, First Aid, Subway" },
  ja: { label: "🟢 LINE 共有", text: "💜 BTS 光化門ライブマップ — 24h カフェ・充電・救急・地下鉄ルート" }
};

export default function LineShare({ lang }: { lang: Lang }) {
  const t = TEXTS[lang];
  const lineUrl = `https://social-plugins.line.me/lineit/share?url=${encodeURIComponent(PAGE_URL)}&text=${encodeURIComponent(t.text)}`;

  return (
    <a
      href={lineUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="info-btn line"
    >
      {t.label}
    </a>
  );
}
