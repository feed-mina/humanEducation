import type { Metadata } from "next";
import Script from "next/script";
import { Analytics } from "@vercel/analytics/react";
import QueryProvider from "@/engine/QueryProvider";
import "./globals.css";

const KAKAO_APP_KEY = process.env.NEXT_PUBLIC_KAKAO_APP_KEY;
const SITE_URL = "https://bts-gwanghwamun.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "💜 BTS 광화문 현장 지도 | ARMY LIVE MAP",
    template: "%s | BTS 광화문 현장 지도"
  },
  description: "BTS 광화문 현장 실시간 정보: 24시간 카페, 핸드폰 충전소, 구급 텐트, 지하철 귀가 루트, 실시간 CCTV 링크 제공",
  keywords: ["BTS", "방탄소년단", "광화문", "BTS 광화문", "아미", "ARMY", "현장지도", "실시간정보", "24시간카페", "충전소"],
  authors: [{ name: "ARMY for ARMY" }],
  creator: "ARMY for ARMY",
  publisher: "BTS Gwanghwamun Live Map",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "💜 BTS 광화문 현장 지도 | ARMY LIVE MAP",
    description: "24시간 카페 · 핸드폰 충전 · 구급 텐트 · 지하철 귀가 루트 · 실시간 CCTV",
    url: SITE_URL,
    siteName: "BTS 광화문 현장 지도",
    locale: "ko_KR",
    type: "website",
    images: [
      {
        url: `${SITE_URL}/og-image.png`,
        width: 1200,
        height: 630,
        alt: "BTS 광화문 현장 지도 미리보기",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "💜 BTS 광화문 현장 지도 | ARMY LIVE MAP",
    description: "실시간 정보: 24h 카페, 충전, 구급, 지하철 루트",
    creator: "@yerinmin257722",
    images: [`${SITE_URL}/og-image.png`],
  },
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
    userScalable: false,
  },
  themeColor: "#6d28d9",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className="h-full">
      <head>
        <link
          rel="preconnect"
          href="https://fonts.googleapis.com"
        />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
        <link 
          rel="stylesheet" 
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
      </head>
      <body className="h-full flex flex-col overflow-hidden">
        <QueryProvider>
          {children}
        </QueryProvider>
        <Analytics />
        {/* Kakao JS SDK (window.Kakao) — for Share API */}
        <Script
          src="https://t1.kakaocdn.net/kakao_js_sdk/2.7.4/kakao.min.js"
          strategy="afterInteractive"
        />
      </body>
    </html>
  );
}
