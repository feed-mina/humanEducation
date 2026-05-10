import { NextResponse } from "next/server";

export interface NoticeItem {
  bdwrSeq: string;
  bdwrTtlNm: string;  // 제목 (★★ 등 특수문자 제거 후 반환)
  updateDate: string;
  bdwrDivCd: string;  // 분류: 0201=교통통제, 0202=버스, 0203=안전
}

export async function GET() {
  try {
    const body = new URLSearchParams({ mainBdwrRowNum: "10" });

    const res = await fetch("https://topis.seoul.go.kr/notice/selectNoticeList.do", {
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

    // bdwrCts는 대용량 HTML이라 제목만 반환
    const rows: NoticeItem[] = (data.rows ?? []).map((r: Record<string, string>) => ({
      bdwrSeq: r.bdwrSeq ?? "",
      bdwrTtlNm: (r.bdwrTtlNm ?? "").replace(/★+\s*/g, "").replace(/<<|>>/g, "").trim(),
      updateDate: r.updateDate ?? "",
      bdwrDivCd: r.bdwrDivCd ?? "",
    }));

    return NextResponse.json(
      { rows, fetchedAt: new Date().toISOString() },
      { headers: { "Cache-Control": "public, max-age=300, stale-while-revalidate=60" } }
    );
  } catch (err) {
    console.error("[/api/notices] TOPIS fetch failed:", err);
    return NextResponse.json(
      { rows: [], fetchedAt: null, error: true },
      { status: 200 }
    );
  }
}
