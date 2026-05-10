import { NextResponse } from "next/server";

export interface TrafficItem {
  accId: number;
  accTypeNm: string;
  accInfo: string;
  roadNm: string;
  linkNm: string;
  accRoadYn: string; // "전체 통제" | "부분 통제"
  occrDt: string;
  clrDt: string;
}

const HEADERS = {
  "Content-Type": "application/x-www-form-urlencoded",
  Referer: "https://topis.seoul.go.kr/",
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
};

export async function GET() {
  try {
    // /map/ 상세 데이터 + /main/ 건수 — 병렬 호출
    const [mapRes, mainRes] = await Promise.all([
      fetch("https://topis.seoul.go.kr/map/selectAccList.do", {
        method: "POST",
        headers: HEADERS,
        // body 불필요 — 기본으로 전체 현황 반환
      }),
      fetch("https://topis.seoul.go.kr/main/selectAccList.do", {
        method: "POST",
        headers: HEADERS,
        body: "sIngYn=Y", // 진행 중인 사고만
      }),
    ]);

    const [mapData, mainData] = await Promise.all([
      mapRes.json(),
      mainRes.json(),
    ]);

    const rows: TrafficItem[] = mapData.rows ?? [];
    const mainRows: { accInfo?: string }[] = mainData.rows ?? [];

    return NextResponse.json(
      {
        rows,                  // /map/ 상세 (NoticeModal용)
        mainCount: mainRows.length,                                              // 전체 진행 건수 (StatusCard 배지)
        btsCount: mainRows.filter((r) =>
          r.accInfo?.toUpperCase().includes("BTS")
        ).length,              // BTS 관련 건수 (StatusCard 배지)
        fetchedAt: new Date().toISOString(),
      },
      {
        headers: {
          "Cache-Control": "public, max-age=120, stale-while-revalidate=60",
        },
      }
    );
  } catch (err) {
    console.error("[/api/traffic] TOPIS fetch failed:", err);
    return NextResponse.json(
      { rows: [], mainCount: 0, btsCount: 0, fetchedAt: null, error: true },
      { status: 200 }
    );
  }
}
