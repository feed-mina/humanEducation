import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://yerin.duckdns.org'
  : 'http://localhost:8080';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const code = searchParams.get('code');

    // console.log('[Kakao Callback] Authorization code:', code);

    if (!code) {
      return NextResponse.redirect(new URL('/view/LOGIN_PAGE?error=no_code', request.url));
    }

    // 백엔드로 code 전달
    // state=mobile: 백엔드가 302 리다이렉트 대신 JSON을 반환하도록 강제 (KakaoController mobile 분기 활용)
    // console.log('[Kakao Callback] Sending code to backend:', `${BACKEND_URL}/api/kakao/callback`);
    const response = await fetch(`${BACKEND_URL}/api/kakao/callback?code=${code}&state=mobile`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // console.log('[Kakao Callback] Backend response status:', response.status);
    // console.log('[Kakao Callback] Backend response headers:', Object.fromEntries(response.headers));

    if (!response.ok) {
      const errorText = await response.text();
      // console.error('[Kakao Callback] Backend error status:', response.status);
      // console.error('[Kakao Callback] Backend error body:', errorText);
      return NextResponse.redirect(new URL('/view/LOGIN_PAGE?error=backend_error', request.url));
    }

    // 백엔드 응답 본문 읽기
    const responseText = await response.text();
    // console.log('[Kakao Callback] Backend response body:', responseText);

    // JSON 파싱 시도
    let data;
    try {
      data = responseText ? JSON.parse(responseText) : {};
    } catch (parseError: any) {
      // console.error('[Kakao Callback] JSON parse error:', parseError.message);
      // console.error('[Kakao Callback] Response text:', responseText);
      return NextResponse.redirect(new URL('/view/LOGIN_PAGE?error=parse_error', request.url));
    }

    // console.log('[Kakao Callback] Parsed data:', data);
    // console.log('[Kakao Callback] User role:', data.role);

    // 백엔드에서 Set-Cookie 헤더 추출
    const setCookieHeaders = response.headers.getSetCookie();
    // console.log('[Kakao Callback] Set-Cookie headers count:', setCookieHeaders.length);
    // console.log('[Kakao Callback] Set-Cookie headers:', setCookieHeaders);

    // Redirect URL 결정
    const redirectUrl = data.role === 'ROLE_GUEST'
      ? new URL('/view/ADDITIONAL_INFO_PAGE', request.url)
      : new URL('/view/MAIN_PAGE', request.url);

    // NextResponse 생성 및 쿠키 전달
    const nextResponse = NextResponse.redirect(redirectUrl);

    // 백엔드에서 받은 모든 Set-Cookie 헤더를 클라이언트로 전달
    setCookieHeaders.forEach(cookie => {
      nextResponse.headers.append('Set-Cookie', cookie);
    });

    // console.log('[Kakao Callback] Redirecting to:', redirectUrl.pathname);
    return nextResponse;

  } catch (error: any) {
    // console.error('[Kakao Callback] Error:', error);
    return NextResponse.redirect(
      new URL('/view/LOGIN_PAGE?error=internal_error', request.url)
    );
  }
}
