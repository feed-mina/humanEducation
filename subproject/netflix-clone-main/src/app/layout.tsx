import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import React from "react";
import AuthSession from "@/app/_component/auth-session";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "K-Ride",
  description: "K-Culture 여행 AI 추천 서비스",
  manifest: "/manifest.json",
  themeColor: "#e50914",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthSession>{children}</AuthSession>
      </body>
    </html>
  );
}
