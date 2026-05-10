import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ sqlKey: string }> }
) {
  const { sqlKey } = await params;
  const searchParams = req.nextUrl.searchParams.toString();
  const url = `${BACKEND_URL}/api/execute/${sqlKey}${searchParams ? `?${searchParams}` : ""}`;

  try {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    // 백엔드 미연결 시 빈 목록 반환 (개발 환경 폴백)
    return NextResponse.json({ status: "success", data: [] }, { status: 200 });
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ sqlKey: string }> }
) {
  const { sqlKey } = await params;
  const body = await req.json();
  const url = `${BACKEND_URL}/api/execute/${sqlKey}`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { status: "error", message: "Backend unavailable" },
      { status: 503 }
    );
  }
}
