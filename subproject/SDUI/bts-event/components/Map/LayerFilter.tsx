import { translations } from "@/data/translations";
import { type Lang } from "../LangToggle";

export type Layer = "cafe" | "charging" | "emergency" | "subway" | "restroom";

interface Props {
  active: Layer | null;
  onSelect: (layer: Layer) => void;
  lang: Lang;
}

export default function LayerFilter({ active, onSelect, lang }: Props) {
  const t = translations[lang].layers;

  const LAYERS: { key: Layer; label: string; emoji: string }[] = [
    { key: "cafe",      label: t.cafe,      emoji: "☕" },
    { key: "charging",  label: t.charging,  emoji: "🔋" },
    { key: "emergency", label: t.emergency, emoji: "🏥" },
    { key: "subway",    label: t.subway,    emoji: "🚇" },
    { key: "restroom",  label: t.restroom,  emoji: "🚻" },
  ];

  return (
    <div className="layer-bar overflow-x-auto no-scrollbar pb-1">
      <div className="flex gap-2 min-w-max">
        {LAYERS.map(({ key, label, emoji }) => (
          <button
            key={key}
            className={`layer-btn shrink-0 ${active === key ? "active" : ""}`}
            onClick={() => onSelect(key)}
          >
            {emoji} {label}
          </button>
        ))}
      </div>
    </div>
  );
}
