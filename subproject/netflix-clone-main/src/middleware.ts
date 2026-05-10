import { NextResponse } from "next/server";
import { auth } from "@/auth";
import type { NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
}

export const config = {
  matcher: [
    "/browse",
    "/movies",
    "/latest",
    "/intro4",
    "/intro5",
    "/my-list",
    "/focus",
    "/profile/account",
    "/profile/manage",
  ],
};
