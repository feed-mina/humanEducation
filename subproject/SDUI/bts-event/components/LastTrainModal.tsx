"use client";

import { X, Train, Bus, MapPin, Clock, AlertTriangle } from "lucide-react";
import { translations } from "@/data/translations";
import { type Lang } from "./LangToggle";

interface Props {
  lang: Lang;
  onClose: () => void;
}

export default function LastTrainModal({ lang, onClose }: Props) {
  const t = (translations as any)[lang] || translations.ko;

  const subwayData = [
    { line: "1", station: "종각/시청", last: "23:45 (인천/신창/서동탄), 00:00 (구로/서울역)" },
    { line: "2", station: "시청/을지로입구", last: "23:55 (내선/외선), 00:15 (성수/신도림)" },
    { line: "3", station: "경복궁/안국", last: "23:50 (수서/오금), 00:05 (구파발/대화)" },
    { line: "5", station: "광화문역 (22:00 이후 오픈)", last: "24:55 (방화), 24:38 (마천)" },
  ];

  const busData = [
    { name: "N62 (심야)", route: "목동 ↔ 광화문 ↔ 면목", interval: "25~35분 (23:40~04:10)" },
    { name: "N75 (심야)", route: "진관 ↔ 광화문 ↔ 신림", interval: "25~35분 (23:30~03:30)" },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div 
        className="modal-content bg-bg-card p-0 overflow-hidden max-w-lg w-[90%] border border-bts-purple/30 animate-in zoom-in duration-300 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-5 bg-bts-gradient flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            {t.lastTrainTitle}
          </h2>
          <button onClick={onClose} className="p-1 hover:bg-white/20 rounded-full transition-colors">
            <X size={20} className="text-white" />
          </button>
        </div>

        <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto custom-scrollbar">
          {/* Urgent Warning */}
          <div className="bg-red-500/20 border border-red-500/30 p-4 rounded-2xl flex gap-3">
             <AlertTriangle className="text-red-500 shrink-0" size={20} />
             <div>
                <p className="text-red-400 font-bold text-sm mb-1">📢 {lang === 'ko' ? '공연 종료! 광화문역 이용 안내' : 'Show Over! Gwanghwamun Stn Info'}</p>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                   {lang === 'ko' 
                    ? '공연이 끝났습니다! 밤 10시까지 광화문역은 무정차 통과하며 출입구가 폐쇄되어 있습니다. 10시 이후 순차적으로 개방될 예정이니, 그전까지는 종각역(1호선)이나 경복궁역(3호선)을 이용해 주세요!' 
                    : 'The show is over! Gwanghwamun Stn is CLOSED till 22:00. It will open sequentially after 22:00. Use Jonggak (L1) or Gyeongbokgung (L3) till then.'}
                </p>
             </div>
          </div>

          {/* Subway Section */}
          <section>
            <h3 className="text-sm font-bold text-bts-purple-light uppercase tracking-wider mb-3 flex items-center gap-2">
              <Train size={16} /> Subway (지하철 막차)
            </h3>
            <div className="space-y-3">
              {subwayData.map((s, i) => (
                <div key={i} className={`p-3 bg-white/5 rounded-xl border ${s.line === '5' ? 'border-red-500/20 bg-red-500/5' : 'border-white/5'}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                       <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white line-bg-${s.line}`}>
                         {s.line}
                       </span>
                       <span className="text-xs font-bold text-white">{s.station}</span>
                    </div>
                    {s.line === '5' && <span className="text-[9px] px-1.5 py-0.5 bg-red-600 text-white rounded font-bold animate-pulse">22:00~</span>}
                  </div>
                  <div className="text-[11px] text-gray-400 flex items-center gap-1">
                    <Clock size={10} /> {s.last}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Bus Section */}
          <section>
            <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Bus size={16} /> Night Bus (심야버스)
            </h3>
            <div className="space-y-3">
              {busData.map((b, i) => (
                <div key={i} className="p-3 bg-white/5 rounded-xl border border-white/5">
                  <div className="text-xs font-bold text-blue-100 mb-1">{b.name}</div>
                  <div className="text-[10px] text-gray-400 mb-1 flex items-center gap-1">
                    <MapPin size={10} /> {b.route}
                  </div>
                  <div className="text-[10px] text-bts-purple-light flex items-center gap-1">
                    <Clock size={10} /> {b.interval}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        <div className="p-4 bg-white/5 border-t border-white/10 flex justify-end">
          <button 
            onClick={onClose}
            className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white text-xs font-bold rounded-full transition-all"
          >
            {t.close}
          </button>
        </div>
      </div>

      <style jsx>{`
        .line-bg-1 { background-color: #0052A4; }
        .line-bg-2 { background-color: #00A84D; }
        .line-bg-3 { background-color: #EF7C1C; }
        .line-bg-5 { background-color: #996CAC; }
      `}</style>
    </div>
  );
}
