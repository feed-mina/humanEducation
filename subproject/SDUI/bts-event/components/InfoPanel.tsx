import { Heart, Map, Share2, Link as LinkIcon, MapPin, Coffee } from "lucide-react";
import KakaoShare from "./Share/KakaoShare";
import LineShare from "./Share/LineShare";
import { type Lang } from "./LangToggle";
import LastTrainModal from "./LastTrainModal";
import { useState } from "react";
import { Train } from "lucide-react";

const PAGE_URL = "https://bts-gwanghwamun.vercel.app";

const TEXTS = {
  ko: {
    cctv: "📹 CCTV",
    topis: "🗺️ 서울시 교통정보 시스템",
    cheer: "치어 모드",
    share: "X 공유",
    copy: "📋 링크 복사",
    location: "📍 내 위치",
    support: "☕ 후원 💜",
    copySuccess: "링크가 복사되었습니다! 💜",
    geoNotSupported: "위치 공유가 지원되지 않는 브라우저입니다.",
    locationCopied: "위치 링크가 복사되었습니다.",
    tweet: "💜 BTS 광화문 현장 실시간 지도\n24h 카페·충전·구급·지하철 루트\n#BTS #방탄소년단 #BTS광화문 #ARMY #아미",
    lastHome: "🏠 귀가 안내",
    restroom: "🚻 화장실"
  },
  en: {
    cctv: "📹 CCTV",
    topis: "🗺️ Seoul Traffic Info System",
    cheer: "Cheer Mode",
    share: "𝕏 Share",
    copy: "📋 Copy Link",
    location: "📍 My Location",
    support: "☕ Support 💜",
    copySuccess: "Link copied to clipboard! 💜",
    geoNotSupported: "Geolocation is not supported by your browser.",
    locationCopied: "Location link copied to clipboard.",
    tweet: "💜 BTS Gwanghwamun Live Map\n24h Cafe, Charging, First Aid, Subway\n#BTS #ARMY #BTSGwanghwamun",
    lastHome: "🏠 Safe Home",
    restroom: "🚻 Restroom"
  },
  ja: {
    cctv: "📹 CCTV",
    topis: "🗺️ ソウル市交通情報システム",
    cheer: "応援モード",
    share: "𝕏 共有",
    copy: "📋 リンクコピー",
    location: "📍 現在地",
    support: "☕ サポート 💜",
    copySuccess: "リンクをコピーしました！ 💜",
    geoNotSupported: "お使いのブラウザは位置情報に対応していません。",
    locationCopied: "位置情報をコピーしました。",
    tweet: "💜 BTS 光化門ライブマップ\n24h カフェ・充電・救急・地下철ルート\n#BTS #ARMY #BTS光化門",
    lastHome: "🏠 帰り道案内",
    restroom: "🚻 お手洗い"
  }
};

interface Props {
  onCheer?: () => void;
  lang: Lang;
}

export default function InfoPanel({ onCheer, lang }: Props) {
  const [showLastTrain, setShowLastTrain] = useState(false);
  const t = TEXTS[lang];
  const KAKAOPAY_URL = process.env.NEXT_PUBLIC_KAKAOPAY_URL;
  const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(t.tweet)}&url=${encodeURIComponent(PAGE_URL)}`;

  function shareLocation() {
    if (!navigator.geolocation) {
      alert(t.geoNotSupported);
      return;
    }
    navigator.geolocation.getCurrentPosition((pos) => {
      const { latitude, longitude } = pos.coords;
      const text = `📍 My Location: https://map.kakao.com/?q=${latitude},${longitude}`;
      if (navigator.share) {
        navigator.share({ title: "My Location", text });
      } else {
        navigator.clipboard.writeText(text);
        alert(t.locationCopied);
      }
    });
  }

  function copyLink() {
    navigator.clipboard.writeText(PAGE_URL);
    alert(t.copySuccess);
  }

  return (
    <div className="info-panel !flex-wrap gap-2">
      {/* <a href="http://cctv.seoul.go.kr" target="_blank" rel="noopener noreferrer" className="info-btn">
        {t.cctv}
      </a> */}
      <a href="http://topis.seoul.go.kr" target="_blank" rel="noopener noreferrer" className="info-btn">
        {t.topis}
      </a>

      <button className="info-btn !bg-indigo-600/90 text-white" onClick={() => setShowLastTrain(true)}>
        <Train size={14} className="text-white" /> {t.lastHome}
      </button>

      <button className="info-btn support !bg-bts-purple-light" onClick={onCheer}>
        <Heart size={14} fill="currentColor" /> {t.cheer}
      </button>

      <KakaoShare lang={lang} />
      <LineShare lang={lang} />

      <a href={twitterUrl} target="_blank" rel="noopener noreferrer" className="info-btn">
        {t.share}
      </a>
      <button className="info-btn" onClick={copyLink}>
        {t.copy}
      </button>
      <button className="info-btn" onClick={shareLocation}>
        {t.location}
      </button>

      {KAKAOPAY_URL && (
        <a href={KAKAOPAY_URL} target="_blank" rel="noopener noreferrer" className="info-btn support">
          {t.support}
        </a>
      )}

      {showLastTrain && <LastTrainModal lang={lang} onClose={() => setShowLastTrain(false)} />}
    </div>
  );
}
