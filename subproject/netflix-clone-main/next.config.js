/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: [
      "image.tmdb.org",
      "occ-0-7129-3996.1.nflxso.net",
      "daisyui.com",
      // Supabase Storage (K-Ride 아티스트/지역 이미지)
      "zrprdknyxkqcwqqarqng.supabase.co",
    ],
  },
};

// next-pwa 설치 후 아래 주석 해제: npm install next-pwa
// const withPWA = require("next-pwa")({
//   dest: "public",
//   disable: process.env.NODE_ENV === "development",
// });
// module.exports = withPWA(nextConfig);

module.exports = nextConfig;
