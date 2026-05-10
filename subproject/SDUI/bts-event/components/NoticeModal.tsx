import { translations } from "@/data/translations";
import { type Lang } from "./LangToggle";
import { Info, AlertTriangle, RefreshCw, FileText } from "lucide-react";
import { useEffect, useState } from "react";
import { type TrafficItem } from "@/app/api/traffic/route";
import { type NoticeItem } from "@/app/api/notices/route";

interface Props {
  lang: Lang;
  onClose: () => void;
  title: string;
}

export default function NoticeModal({ lang, onClose, title }: Props) {
  const t = (translations as any)[lang] || translations.ko;

  const [traffic, setTraffic] = useState<TrafficItem[]>([]);
  const [loadingTraffic, setLoadingTraffic] = useState(true);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [notices, setNotices] = useState<NoticeItem[]>([]);

  const fetchTraffic = () => {
    setLoadingTraffic(true);
    fetch("/api/traffic")
      .then((r) => r.json())
      .then((data) => {
        setTraffic(data.rows ?? []);
        setFetchedAt(data.fetchedAt ?? null);
      })
      .catch(() => {})
      .finally(() => setLoadingTraffic(false));
  };

  useEffect(() => {
    fetchTraffic();
    // 공지 별도 fetch (캐시 5분, 독립적으로 호출)
    fetch("/api/notices")
      .then((r) => r.json())
      .then((data) => setNotices(data.rows ?? []))
      .catch(() => {});
  }, []);

  // BTS 관련 공지만 필터
  const btsNotices = notices.filter(
    (n) =>
      n.bdwrTtlNm.toUpperCase().includes("BTS") ||
      n.bdwrTtlNm.includes("광화문") ||
      n.bdwrTtlNm.includes("시청")
  );

  const btsItems = traffic.filter((item) =>
    item.accInfo?.toUpperCase().includes("BTS")
  );
  const otherItems = traffic.filter(
    (item) => !item.accInfo?.toUpperCase().includes("BTS")
  );

  const formatClrTime = (clrDt: string) => {
    // "2026-03-21 23:30" → "23:30"
    return clrDt?.slice(11, 16) ?? "";
  };

  const controlBadge = (accRoadYn: string) => {
    if (accRoadYn === "전체 통제")
      return (
        <span className="text-[9px] px-1.5 py-0.5 bg-red-600 text-white rounded font-bold shrink-0">
          전면통제
        </span>
      );
    if (accRoadYn === "부분 통제")
      return (
        <span className="text-[9px] px-1.5 py-0.5 bg-orange-500 text-white rounded font-bold shrink-0">
          부분통제
        </span>
      );
    return null;
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div 
        className="modal-content bg-[#1a1a2e] border-2 border-red-500/50 p-0 overflow-hidden max-w-md w-[90%] shadow-2xl animate-in fade-in zoom-in duration-300"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bg-red-600 p-4 flex items-center justify-between">
           <h2 className="text-white font-bold flex items-center gap-2">
             <AlertTriangle size={18} /> {lang === 'ko' ? '📢 긴급 교통 통제 공지' : 'Emergency Notice'}
           </h2>
           <button onClick={onClose} className="text-white/80 hover:text-white">✕</button>
        </div>

        <div className="p-6 space-y-4">
          <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-xl">
             <p className="text-red-400 font-bold text-lg mb-2">
               🚨 {lang === 'ko' ? '공연 종료 및 퇴장 통제 안내' : 'Show Ended - Exit Controls'}
             </p>
             <div className="space-y-2 text-sm text-gray-200">
                <p>• ⏱️ **{lang === 'ko' ? '현재 시간: 21:55 (공연 성료)' : 'Current Time: 21:55 (Show Over)'}**</p>
                <p>• 🚫 **{lang === 'ko' ? '내용: 관객 퇴장 및 주변 통제 지속 (23:30까지)' : 'Details: Exit Controls in Place (Until 23:30)'}**</p>
                <p>• 📍 **{lang === 'ko' ? '통제: 세종대로, 을지로, 소공로 일대 전면 통제' : 'Closure: Sejong-daero, Euljiro, Sogong-ro fully closed'}**</p>
             </div>
          </div>

          <div className="space-y-3">
             <h3 className="font-bold text-white flex items-center gap-2 text-sm italic">
               <Info size={14} className="text-bts-purple-light" /> {lang === 'ko' ? '대체 이용 가능한 역 (경복궁·종각 이용 권장)' : 'Alternative Stations (Jonggak/Gyeongbokgung)'}
             </h3>
             <div className="grid grid-cols-1 gap-2">
                <div className="p-3 bg-white/5 rounded-lg border border-white/10 flex justify-between items-center">
                   <span className="text-xs text-white">1호선 **종각역** (0.5km)</span>
                   <span className="text-[10px] text-gray-500">도보 8분</span>
                </div>
                <div className="p-3 bg-white/5 rounded-lg border border-white/10 flex justify-between items-center">
                   <span className="text-xs text-white">3호선 **경복궁역** (0.6km)</span>
                   <span className="text-[10px] text-gray-500">도보 10분</span>
                </div>
                <div className="p-3 bg-white/5 rounded-lg border border-white/10 flex justify-between items-center">
                   <span className="text-xs text-white">2호선 **을지로입구역** (0.8km)</span>
                   <span className="text-[10px] text-gray-500">도보 12분</span>
                </div>
             </div>
          </div>

          {/* Real-time Traffic Section — TOPIS Live Data */}
          <div className="space-y-2 pt-2 border-t border-white/10">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-amber-400 text-xs flex items-center gap-2 uppercase">
                📍 {lang === "ko" ? "실시간 사고·통제 (서울시 교통정보 시스템)" : lang === "ja" ? "リアルタイム規制 (ソウル市交通情報システム)" : "Live Traffic Controls (Seoul Traffic Info)"}
              </h3>
              <button
                onClick={fetchTraffic}
                disabled={loadingTraffic}
                className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-white transition-colors disabled:opacity-40"
              >
                <RefreshCw
                  size={10}
                  className={loadingTraffic ? "animate-spin" : ""}
                />
                {fetchedAt
                  ? new Date(fetchedAt).toLocaleTimeString("ko-KR", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : ""}
              </button>
            </div>

            {loadingTraffic ? (
              <div className="flex items-center gap-2 text-[11px] text-gray-500">
                <RefreshCw size={10} className="animate-spin" />
                {lang === "ko" ? "불러오는 중..." : "Loading..."}
              </div>
            ) : traffic.length === 0 ? (
              <p className="text-[11px] text-gray-500 italic">
                {lang === "ko"
                  ? "교통 정보를 불러오지 못했습니다."
                  : "Could not load traffic data."}
              </p>
            ) : (
              <div className="space-y-3">
                {/* BTS-related closures — highlighted */}
                {btsItems.length > 0 && (
                  <div>
                    <p className="text-[10px] font-bold text-red-400 uppercase mb-1.5">
                      🚨 BTS {lang === "ko" ? "행사 관련 통제" : "Event Closures"} ({btsItems.length})
                    </p>
                    <ul className="space-y-1.5">
                      {btsItems.map((item) => (
                        <li
                          key={item.accId}
                          className="text-[11px] bg-red-500/10 border border-red-500/20 rounded-lg px-2.5 py-1.5"
                        >
                          <div className="flex items-start justify-between gap-1.5">
                            <span className="text-gray-200 leading-snug flex-1">
                              {item.accInfo?.replace(/\r\n/g, " ").trim()}
                            </span>
                            {controlBadge(item.accRoadYn)}
                          </div>
                          {item.clrDt && (
                            <p className="text-[10px] text-gray-500 mt-0.5">
                              ⏱ {lang === "ko" ? "해제 예정" : "Until"} {formatClrTime(item.clrDt)}
                            </p>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Other current incidents */}
                {otherItems.length > 0 && (
                  <details className="group">
                    <summary className="text-[10px] text-gray-500 cursor-pointer select-none hover:text-gray-300 list-none flex items-center gap-1">
                      <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
                      {lang === "ko"
                        ? `기타 교통 통제 ${otherItems.length}건`
                        : `Other controls (${otherItems.length})`}
                    </summary>
                    <ul className="mt-1.5 space-y-1">
                      {otherItems.map((item) => (
                        <li
                          key={item.accId}
                          className="text-[11px] flex items-start gap-1.5 text-gray-400"
                        >
                          <span className="shrink-0 mt-0.5">
                            {item.accTypeNm === "공사" ? "🔧" : "📌"}
                          </span>
                          <span className="leading-snug">
                            [{item.accTypeNm}] {item.roadNm} —{" "}
                            {item.accInfo
                              ?.split("\n")[0]
                              .replace(/\r/g, "")
                              .trim()
                              .slice(0, 60)}
                            {(item.accInfo?.length ?? 0) > 60 ? "…" : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            )}
          </div>

          {/* TOPIS 공식 공지 섹션 */}
          {btsNotices.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-white/10">
              <h3 className="font-bold text-sky-400 text-xs flex items-center gap-2 uppercase">
                <FileText size={12} />
                {lang === "ko" ? "서울시 교통정보 시스템 공식 공지" : lang === "ja" ? "ソウル市交通情報システム公式通知" : "Seoul Traffic Info System — Official Notices"}
              </h3>
              <ul className="space-y-1.5">
                {btsNotices.map((n) => (
                  <li
                    key={n.bdwrSeq}
                    className="text-[11px] bg-sky-500/10 border border-sky-500/20 rounded-lg px-2.5 py-2"
                  >
                    <p className="text-sky-100 leading-snug">{n.bdwrTtlNm}</p>
                    <p className="text-[9px] text-gray-500 mt-0.5">{n.updateDate}</p>
                  </li>
                ))}
              </ul>
              <a
                href="https://topis.seoul.go.kr/notice/openNoticeBoard.do"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-[10px] text-sky-400/70 hover:text-sky-400 transition-colors"
              >
                <FileText size={9} />
                {lang === "ko" ? "서울시 교통정보 시스템 공지 전체 보기 →" : lang === "ja" ? "全ての通知を見る →" : "View all notices (Seoul Traffic Info) →"}
              </a>
            </div>
          )}

          <p className="text-[11px] text-gray-500 leading-relaxed italic border-l-2 border-bts-purple-light pl-2">
             ※ 광화문역은 22:00 이후 순차적으로 개방될 예정이나, 인파 밀집 시 지연될 수 있습니다. 귀가 인파를 위해 임시 열차가 증편될 예정이오니 안내에 따라 이동해 주세요.
          </p>
        </div>

        <div className="p-4 bg-white/5 border-t border-white/10 flex justify-center">
          <button 
            onClick={onClose}
            className="w-full py-3 bg-bts-purple text-white font-bold rounded-xl transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            {t.close}
          </button>
        </div>
      </div>
    </div>
  );
}
