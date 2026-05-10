"use client";

declare global {
  interface Window {
    Kakao: any;
  }
}

import { type Lang } from "../LangToggle";

const TEXTS = {
  ko: { label: "💬 카카오톡 공유", title: "💜 BTS 광화문 현장 지도", desc: "24시간 카페 · 충전 · 구급 텐트 · 지하철 탈출 루트", btn: "지도 열기" },
  en: { label: "💬 Share on Kakao", title: "💜 BTS Gwanghwamun Live Map", desc: "24h Cafe, Charging, First Aid, Subway Exit Routes", btn: "Open Map" },
  ja: { label: "💬 カカオトーク共有", title: "💜 BTS 光化門ライブマップ", desc: "24h カフェ・充電・救急・地下鉄ルート", btn: "地図を開く" }
};

export default function KakaoShare({ lang }: { lang: Lang }) {
  const t = TEXTS[lang];
  const share = () => {
    if (!window.Kakao) return;
    if (!window.Kakao.isInitialized()) {
      window.Kakao.init(process.env.NEXT_PUBLIC_KAKAO_APP_KEY);
    }
    if (!window.Kakao.Share) return;

    window.Kakao.Share.sendDefault({
      objectType: "feed",
      content: {
        title: t.title,
        description: t.desc,
        imageUrl: "https://bts-gwanghwamun.vercel.app/og-image.png",
        link: {
          mobileWebUrl: "https://bts-gwanghwamun.vercel.app",
          webUrl: "https://bts-gwanghwamun.vercel.app",
        },
      },
      buttons: [
        {
          title: t.btn,
          link: {
            mobileWebUrl: "https://bts-gwanghwamun.vercel.app",
            webUrl: "https://bts-gwanghwamun.vercel.app",
          },
        },
      ],
    });
  };

  return (
    <button className="info-btn kakao" onClick={share}>
      {t.label}
    </button>
  );
}
