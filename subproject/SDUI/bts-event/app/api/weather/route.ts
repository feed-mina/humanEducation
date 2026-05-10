import { NextResponse } from "next/server";

export interface WeatherData {
  tmprt: number;     // 기온 (°C)
  skySttus: number;  // 하늘상태: 1=맑음, 3=구름많음, 4=흐림
  pty: number;       // 강수형태: 0=없음, 1=비, 2=비/눈, 3=눈
  wfkor: string;     // 날씨 한국어 (e.g. "맑음")
  wfen: string;      // 날씨 영어 (e.g. "Clear")
  pop: number;       // 강수확률 (%)
  pm10: number;      // 미세먼지 (㎍/㎥)
  pm25: number;      // 초미세먼지 (㎍/㎥)
  curPm10G: string;  // 미세먼지 등급: G=좋음, B=보통, W=나쁨, VW=매우나쁨
  curPm25G: string;  // 초미세먼지 등급
  ws: string;        // 풍속 (m/s)
}

/** KST 기준 현재 날짜/시간 반환 */
function getSeoulDateTime() {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60_000;
  const kstMs = utcMs + 9 * 3_600_000;
  const d = new Date(kstMs);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  return { today: `${y}${m}${dd}`, currTime: `${hh}00` };
}

export async function GET() {
  try {
    const { today, currTime } = getSeoulDateTime();
    const body = new URLSearchParams({ today, currTime, regCd: "111121" });

    const res = await fetch("https://topis.seoul.go.kr/main/selectWeather.do", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Referer: "https://topis.seoul.go.kr/",
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
      },
      body: body.toString(),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const row: WeatherData | null = data.rows?.[0] ?? null;

    return NextResponse.json(
      { data: row, fetchedAt: new Date().toISOString() },
      { headers: { "Cache-Control": "public, max-age=300, stale-while-revalidate=60" } }
    );
  } catch (err) {
    console.error("[/api/weather] TOPIS fetch failed:", err);
    return NextResponse.json(
      { data: null, fetchedAt: null, error: true },
      { status: 200 }
    );
  }
}
