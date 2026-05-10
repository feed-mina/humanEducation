"use client";

export type Lang = "ko" | "en" | "ja";

const LANGS: { key: Lang; label: string }[] = [
  { key: "ko", label: "🇰🇷" },
  { key: "en", label: "🇺🇸" },
  { key: "ja", label: "🇯🇵" },
];

interface Props {
  lang: Lang;
  onChange: (lang: Lang) => void;
}

export default function LangToggle({ lang, onChange }: Props) {
  return (
    <div className="flex gap-1">
      {LANGS.map(({ key, label }) => (
        <button
          key={key}
          className={`tab-btn px-2 py-1 text-xs ${lang === key ? "active" : ""}`}
          onClick={() => onChange(key)}
          title={key.toUpperCase()}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
