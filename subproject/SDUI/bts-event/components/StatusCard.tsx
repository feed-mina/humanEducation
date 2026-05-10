"use client";

import { CloudRain, Thermometer, ExternalLink, Wind, AlertTriangle } from "lucide-react";
import { useState, useEffect } from "react";
import { type WeatherData } from "@/app/api/weather/route";

interface Props {
  lang: string;
}

const DUST_MAP: Record<string, { ko: string; en: string; ja: string; color: string }> = {
  G:  { ko: "좋음",   en: "Good",   ja: "良い",   color: "text-blue-400" },
  B:  { ko: "보통",   en: "Fair",   ja: "普通",   color: "text-green-400" },
  W:  { ko: "나쁨",   en: "Bad",    ja: "悪い",   color: "text-orange-400" },
  VW: { ko: "매우나쁨", en: "V.Bad", ja: "非常に悪い", color: "text-red-400" },
};

function skyEmoji(skySttus: number, pty: number): string {
  if (pty === 1) return "🌧️";
  if (pty === 2) return "🌨️";
  if (pty === 3) return "❄️";
  if (skySttus === 3) return "⛅";
  if (skySttus === 4) return "☁️";
  return "☀️";
}

export default function StatusCard({ lang }: Props) {
  const [showWeather, setShowWeather] = useState(false);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [btsCount, setBtsCount] = useState<number | null>(null);
  const [mainCount, setMainCount] = useState<number | null>(null);

  useEffect(() => {
    // 날씨 + 교통 건수 병렬 fetch
    fetch("/api/weather")
      .then((r) => r.json())
      .then((d) => { if (d.data) setWeather(d.data); })
      .catch(() => {});

    fetch("/api/traffic")
      .then((r) => r.json())
      .then((d) => {
        if (d.btsCount != null) setBtsCount(d.btsCount);
        if (d.mainCount != null) setMainCount(d.mainCount);
      })
      .catch(() => {});
  }, []);

  const t = {
    ko: {
      congestion: "매우 혼잡 🔥 (퇴장 중)",
      weatherTitle: "실시간 날씨",
      source: "기상청 실시간 상세정보",
      fallback: "⚠️ 화면이 보이지 않으면 아래 버튼을 클릭해 주세요.",
      openNew: "기상청 새창열기",
      close: "닫기",
      btsLabel: (n: number) => `BTS 통제 ${n}건`,
      allLabel: (n: number) => `전체 사고·통제 ${n}건`,
    },
    en: {
      congestion: "Very Crowded 🔥 (Exiting)",
      weatherTitle: "Live Weather",
      source: "Source: Weather.go.kr",
      fallback: "⚠️ If blank, click the button below.",
      openNew: "Open Weather Site",
      close: "Close",
      btsLabel: (n: number) => `${n} BTS closures`,
      allLabel: (n: number) => `${n} active incidents`,
    },
    ja: {
      congestion: "非常に混雑 🔥 (退場中)",
      weatherTitle: "ライブ天気",
      source: "詳細: 気象庁",
      fallback: "⚠️ 表示されない場合は下のボタンから開いてください。",
      openNew: "公式サイトを開く",
      close: "閉じる",
      btsLabel: (n: number) => `BTS規制 ${n}件`,
      allLabel: (n: number) => `規制 ${n}件`,
    },
  };

  const text = (t as any)[lang] || t.ko;
  const dust = weather ? (DUST_MAP[weather.curPm10G] ?? null) : null;
  const emoji = weather ? skyEmoji(weather.skySttus, weather.pty) : "☀️";
  const temp = weather?.tmprt ?? "—";
  const rainChance = weather != null ? `${weather.pop}%` : "—%";
  const weatherUrl = "https://www.weather.go.kr/w/m/index.do";

  return (
    <>
      <div
        className="status-card cursor-pointer hover:scale-105 active:scale-95 transition-all border-l-4 border-bts-purple-light"
        onClick={() => setShowWeather(true)}
        title={text.weatherTitle}
      >
        {/* 날씨 행 */}
        <div className="status-item text-white">
          <span className="text-base leading-none">{emoji}</span>
          <Thermometer size={13} className="text-orange-400" />
          <span>{temp}°C</span>
          <span className="opacity-40 mx-1">|</span>
          <CloudRain size={13} className="text-blue-300" />
          <span>{rainChance}</span>
          {dust && (
            <>
              <span className="opacity-40 mx-1">|</span>
              <Wind size={12} className={dust.color} />
              <span className={`text-[10px] ${dust.color}`}>
                {(dust as any)[lang] ?? dust.ko}
              </span>
            </>
          )}
        </div>

        {/* 혼잡도 행 */}
        <div className="status-item mt-1">
          <span className="text-purple-100 text-[11px]">{text.congestion}</span>
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse ml-1 shrink-0" />
        </div>

        {/* 사고·통제 건수 배지 */}
        {btsCount != null && btsCount > 0 && (
          <div className="status-item mt-1 gap-1.5">
            <AlertTriangle size={11} className="text-red-400 shrink-0" />
            <span className="text-[10px] text-red-400">{text.btsLabel(btsCount)}</span>
            {mainCount != null && mainCount > btsCount && (
              <span className="text-[9px] text-gray-500">
                / {text.allLabel(mainCount)}
              </span>
            )}
          </div>
        )}

        <div className="text-[9px] text-gray-500 mt-1 flex items-center gap-1">
          <ExternalLink size={8} />
          {text.source}
        </div>
      </div>

      {/* 날씨 상세 모달 */}
      {showWeather && (
        <div className="modal-overlay" onClick={() => setShowWeather(false)}>
          <div
            className="modal-content !max-w-[500px] !p-0 overflow-hidden flex flex-col h-[75vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header !mb-0 p-4 border-b border-border bg-bg-card flex items-center justify-between shrink-0">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                {emoji} {text.weatherTitle}
                {weather && (
                  <span className="text-sm font-normal text-gray-400">
                    {weather.tmprt}°C · {weather.wfkor}
                  </span>
                )}
              </h2>
              <button
                onClick={() => setShowWeather(false)}
                className="p-1 hover:bg-white/10 rounded-full"
              >
                <div className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-white transition-colors">
                  ✕
                </div>
              </button>
            </div>

            {/* 미세먼지 요약 바 */}
            {weather && (
              <div className="px-4 py-2 bg-black/30 flex gap-4 text-[11px] border-b border-white/10 shrink-0">
                <span className="text-gray-400">
                  PM10 <span className={DUST_MAP[weather.curPm10G]?.color ?? "text-white"}>
                    {weather.pm10}㎍ ({(DUST_MAP[weather.curPm10G] as any)?.[lang] ?? DUST_MAP[weather.curPm10G]?.ko})
                  </span>
                </span>
                <span className="text-gray-400">
                  PM2.5 <span className={DUST_MAP[weather.curPm25G]?.color ?? "text-white"}>
                    {weather.pm25}㎍ ({(DUST_MAP[weather.curPm25G] as any)?.[lang] ?? DUST_MAP[weather.curPm25G]?.ko})
                  </span>
                </span>
                <span className="text-gray-400">
                  💧 {weather.pop}%
                </span>
              </div>
            )}

            <div className="flex-1 bg-white relative">
              <iframe
                src={weatherUrl}
                className="w-full h-full border-none"
                title="Official Weather Info"
              />
              <div className="absolute inset-x-0 bottom-4 px-6 pointer-events-none">
                <div className="bg-black/70 backdrop-blur rounded-lg p-3 text-center border border-white/20 pointer-events-auto">
                  <p className="text-[10px] text-white/80 mb-2">{text.fallback}</p>
                </div>
              </div>
            </div>

            <div className="p-3 bg-bg-card border-t border-border flex justify-between items-center shrink-0">
              <a
                href={weatherUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-indigo-600 text-white text-xs font-bold rounded-lg flex items-center gap-2"
              >
                <ExternalLink size={12} />
                {text.openNew}
              </a>
              <button
                onClick={() => setShowWeather(false)}
                className="px-4 py-2 bg-gray-700 text-white text-xs font-semibold rounded-lg hover:bg-gray-600"
              >
                {text.close}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
