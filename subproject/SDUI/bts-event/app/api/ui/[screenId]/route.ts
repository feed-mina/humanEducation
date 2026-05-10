/**
 * API Route: GET /api/ui/[screenId]
 * 
 * data/screens/{screenId}.json 파일을 읽어서 메타데이터를 반환합니다.
 * 
 * DB 전환 시:
 *   이 라우트를 Spring Boot API 프록시로 교체하거나,
 *   직접 PostgreSQL에서 ui_metadata를 조회하도록 수정합니다.
 *   응답 형식 { data, success }는 동일하게 유지합니다.
 */
import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ screenId: string }> }
) {
  const { screenId } = await params;

  try {
    const filePath = path.join(process.cwd(), "data", "screens", `${screenId}.json`);
    const fileContent = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(fileContent);

    return NextResponse.json({
      data,
      success: true,
    });
  } catch (error) {
    // 파일이 없으면 빈 배열 반환 (DB 전환 시에도 동일한 에러 핸들링)
    console.error(`[API /ui/${screenId}] 메타데이터 파일 없음:`, error);
    return NextResponse.json(
      {
        data: [],
        success: false,
        error: `Screen '${screenId}' not found`,
      },
      { status: 404 }
    );
  }
}
