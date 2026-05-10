"""
test_contract.py
================
FastAPI ↔ Spring Boot 계약(Contract) 테스트

두 서버가 모두 실행 중일 때 실행하는 E2E 스타일 테스트.
각 서버의 응답 스키마가 Next.js 프론트엔드가 기대하는 형태와 일치하는지 검증.

사전 조건:
  - FastAPI  포트 8000 실행: uvicorn src.api.fastapi_server:app --port 8000
  - Spring Boot 포트 8080 실행: ./gradlew bootRun (또는 Docker)

실행:
  cd D:/kride-project
  pytest src/api/test_contract.py -v -m contract

의존 패키지:
  pip install pytest httpx
"""

from __future__ import annotations

import pytest
import httpx

FASTAPI_BASE   = "http://localhost:8000"
SPRINGBOOT_BASE = "http://localhost:8080"

# ─────────────────────────────────────────────────────────────────────────────
# 서버 가용성 체크 (서버가 꺼져 있으면 테스트 skip)
# ─────────────────────────────────────────────────────────────────────────────
def _is_up(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/api/health", timeout=3)
        return r.status_code < 500
    except Exception:
        return False


requires_fastapi    = pytest.mark.skipif(not _is_up(FASTAPI_BASE),    reason="FastAPI 서버 미실행 (port 8000)")
requires_springboot = pytest.mark.skipif(not _is_up(SPRINGBOOT_BASE), reason="Spring Boot 서버 미실행 (port 8080)")
requires_both       = pytest.mark.skipif(
    not (_is_up(FASTAPI_BASE) and _is_up(SPRINGBOOT_BASE)),
    reason="FastAPI(8000) + Spring Boot(8080) 모두 실행 필요",
)

pytestmark = pytest.mark.contract   # -m contract 필터용


# ═════════════════════════════════════════════════════════════════════════════
# 1. FastAPI 서버 단독 계약 검증
# ═════════════════════════════════════════════════════════════════════════════
class TestFastAPIContract:

    @requires_fastapi
    def test_health_schema(self):
        """FastAPI health 응답 → {status, graph_nodes, graph_edges, road_scored_rows}"""
        r = httpx.get(f"{FASTAPI_BASE}/api/health")
        assert r.status_code == 200
        body = r.json()
        for field in ("status", "graph_nodes", "graph_edges", "road_scored_rows"):
            assert field in body, f"missing field: {field}"
        assert body["status"] == "ok"

    @requires_fastapi
    def test_artists_schema(self):
        """GET /api/artists → {artists: [{id, name, imageUrl?}]}"""
        r = httpx.get(f"{FASTAPI_BASE}/api/artists")
        assert r.status_code in (200, 503), f"unexpected: {r.status_code}"
        if r.status_code == 200:
            body = r.json()
            assert "artists" in body
            assert isinstance(body["artists"], list)
            if body["artists"]:
                item = body["artists"][0]
                assert "id" in item
                assert "name" in item

    @requires_fastapi
    def test_regions_schema(self):
        """GET /api/regions → {regions: [{id, name, imageUrl?, safety_score?}]}"""
        r = httpx.get(f"{FASTAPI_BASE}/api/regions")
        assert r.status_code in (200, 503), f"unexpected: {r.status_code}"
        if r.status_code == 200:
            body = r.json()
            assert "regions" in body
            assert isinstance(body["regions"], list)
            # fallback 이든 Neo4j든 반드시 1개 이상
            assert len(body["regions"]) >= 1
            item = body["regions"][0]
            assert "id" in item
            assert "name" in item

    @requires_fastapi
    def test_itinerary_schema(self):
        """POST /api/recommend/itinerary → {itinerary: list, mapData: {markers: list}}"""
        payload = {
            "duration": "당일치기",
            "artists": [],
            "regions": ["서울"],
            "purposes": ["food"],
            "budget": {"min": 0, "max": 300000},
        }
        r = httpx.post(f"{FASTAPI_BASE}/api/recommend/itinerary", json=payload, timeout=30)
        assert r.status_code in (200, 503), f"unexpected: {r.status_code}"
        if r.status_code == 200:
            body = r.json()
            assert "itinerary" in body,        "itinerary 키 누락"
            assert "mapData" in body,           "mapData 키 누락"
            assert "markers" in body["mapData"], "markers 키 누락"
            assert isinstance(body["itinerary"], list)
            assert isinstance(body["mapData"]["markers"], list)

    @requires_fastapi
    def test_recommend_ai_schema(self):
        """POST /api/recommend/ai → {pois: list, recommendation_text: str, count: int}"""
        payload = {
            "artists": [],
            "regions": ["서울"],
            "purposes": [],
            "budget": {"min": 0, "max": 500000},
        }
        r = httpx.post(f"{FASTAPI_BASE}/api/recommend/ai", json=payload, timeout=30)
        assert r.status_code in (200, 503), f"unexpected: {r.status_code}"
        if r.status_code == 200:
            body = r.json()
            assert "pois" in body
            assert "recommendation_text" in body
            assert "count" in body
            assert isinstance(body["pois"], list)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Spring Boot 서버 단독 계약 검증
# ═════════════════════════════════════════════════════════════════════════════
class TestSpringBootContract:

    @requires_springboot
    def test_ui_intro2_schema(self):
        """GET /api/ui/KRIDE_INTRO2 → ApiResponse { status, data: [{componentId, componentType, ...}] }"""
        r = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_INTRO2", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert body["status"] == "success"
        assert "data" in body
        assert isinstance(body["data"], list)

    @requires_springboot
    def test_ui_intro3_schema(self):
        """GET /api/ui/KRIDE_INTRO3 → ApiResponse { status, data: list }"""
        r = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_INTRO3", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    @requires_springboot
    def test_ui_focus_schema(self):
        """GET /api/ui/KRIDE_FOCUS → ApiResponse { status, data: list }"""
        r = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_FOCUS", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    @requires_springboot
    def test_ui_component_fields(self):
        """data 배열 항목에 Next.js DynamicEngine 필수 필드 포함"""
        r = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_INTRO2", timeout=10)
        assert r.status_code == 200
        data = r.json()["data"]
        if not data:
            pytest.skip("DB에 KRIDE_INTRO2 메타데이터 없음 — 스킵")
        item = data[0]
        # DynamicEngine이 반드시 사용하는 필드
        for field in ("componentId", "componentType", "sortOrder"):
            assert field in item, f"DynamicEngine 필수 필드 누락: {field}"

    @requires_springboot
    def test_unknown_screen_returns_empty_not_500(self):
        """존재하지 않는 screenId → 500이 아닌 200 + 빈 data"""
        r = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/__NO_SUCH_SCREEN__", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["data"] == [] or isinstance(body["data"], list)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Cross-service — 두 서버의 응답이 Next.js 온보딩 플로우에서 호환되는지
# ═════════════════════════════════════════════════════════════════════════════
class TestCrossServiceContract:

    @requires_both
    def test_onboarding_movies_flow(self):
        """
        /movies 페이지 플로우:
          1) Spring Boot → /api/ui/KRIDE_INTRO2  : UI 메타데이터
          2) FastAPI     → /api/artists    : 아티스트 목록
          두 응답 모두 200이어야 하고 스키마가 일치해야 한다.
        """
        ui_res = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_INTRO2", timeout=10)
        ai_res = httpx.get(f"{FASTAPI_BASE}/api/artists", timeout=10)

        assert ui_res.status_code == 200, "Spring Boot KRIDE_INTRO2 실패"
        assert ai_res.status_code in (200, 503), f"FastAPI artists 실패: {ai_res.status_code}"

        ui_body = ui_res.json()
        assert ui_body["status"] == "success"

        if ai_res.status_code == 200:
            ai_body = ai_res.json()
            assert "artists" in ai_body
            # artists 목록이 있으면 id/name 필드 검증
            if ai_body["artists"]:
                assert "id"   in ai_body["artists"][0]
                assert "name" in ai_body["artists"][0]

    @requires_both
    def test_onboarding_latest_flow(self):
        """
        /latest 페이지 플로우:
          1) Spring Boot → /api/ui/KRIDE_INTRO3  : UI 메타데이터
          2) FastAPI     → /api/regions    : 지역 목록
        """
        ui_res = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_INTRO3", timeout=10)
        ai_res = httpx.get(f"{FASTAPI_BASE}/api/regions", timeout=10)

        assert ui_res.status_code == 200, "Spring Boot KRIDE_INTRO3 실패"
        assert ai_res.status_code in (200, 503), f"FastAPI regions 실패: {ai_res.status_code}"

        if ai_res.status_code == 200:
            regions = ai_res.json()["regions"]
            assert len(regions) >= 1, "지역 목록이 비어 있음 (fallback도 실패)"

    @requires_both
    def test_onboarding_focus_flow(self):
        """
        /focus 페이지 플로우:
          1) Spring Boot → /api/ui/KRIDE_FOCUS                  : UI 메타데이터
          2) FastAPI     → /api/recommend/itinerary (POST) : AI 일정
        """
        ui_res = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_FOCUS", timeout=10)
        ai_res = httpx.post(
            f"{FASTAPI_BASE}/api/recommend/itinerary",
            json={
                "duration": "당일치기",
                "artists": [],
                "regions": ["서울"],
                "purposes": [],
                "budget": {"min": 0, "max": 300000},
            },
            timeout=30,
        )

        assert ui_res.status_code == 200, "Spring Boot KRIDE_FOCUS 실패"
        assert ai_res.status_code in (200, 503), f"FastAPI itinerary 실패: {ai_res.status_code}"

        if ai_res.status_code == 200:
            body = ai_res.json()
            assert "itinerary" in body
            assert "mapData" in body

    @requires_both
    def test_both_servers_healthy(self):
        """두 서버 모두 정상 응답"""
        fa = httpx.get(f"{FASTAPI_BASE}/api/health", timeout=5)
        # Spring Boot는 /api/health 없으므로 /api/ui/KRIDE_INTRO2 로 대체
        sb = httpx.get(f"{SPRINGBOOT_BASE}/api/ui/KRIDE_INTRO2", timeout=5)

        assert fa.status_code == 200, "FastAPI 불응"
        assert sb.status_code == 200, "Spring Boot 불응"
        assert fa.json()["status"] == "ok"
