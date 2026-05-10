"""supabase_client.py — Supabase Full DB 클라이언트"""
from __future__ import annotations
import os
from supabase import create_client, Client

_client: Client | None = None

def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
    return _client


def get_all_artists() -> list[dict]:
    """artist 테이블 전체 조회 → {id, name, imageUrl}"""
    resp = get_client().table("artist").select("id, name, image_url").execute()
    return [
        {"id": row["id"], "name": row["name"], "imageUrl": row["image_url"]}
        for row in (resp.data or [])
    ]


def get_poi_details(poi_ids: list[str]) -> list[dict]:
    """poi 테이블에서 상세 정보 + 이미지 URL 조회"""
    resp = (
        get_client()
        .table("poi")
        .select("id, name, address, lat, lon, category, image_url, avg_cost")
        .in_("id", poi_ids)
        .execute()
    )
    return resp.data or []


def get_artist_poi_map(artist_ids: list[str]) -> list[dict]:
    """artist_poi 조인 테이블 — 아티스트별 촬영지 목록"""
    resp = (
        get_client()
        .table("artist_poi")
        .select("artist_id, poi_id")
        .in_("artist_id", artist_ids)
        .execute()
    )
    return resp.data or []