"use client";

import { useEffect, useRef } from "react";
import type { Layer } from "./LayerFilter";
import locationsData from "@/data/locations.json";

declare global {
  interface Window {
    kakao: any;
  }
}

interface Props {
  activeLayers: Set<Layer>;
  lang: "ko" | "en" | "ja";
}

const LAYER_EMOJI: Record<Layer, string> = {
  cafe:      "☕",
  charging:  "🔋",
  emergency: "🏥",
  subway:    "🚇",
  restroom:  "🚻",
};

function markerContent(emoji: string): string {
  return `<div class="map-marker">${emoji}</div>`;
}

function infoContent(title: string, memo: string): string {
  return `<div class="map-info"><div class="title">${title}</div><div class="memo">${memo}</div></div>`;
}

export default function KakaoMap({ activeLayers, lang }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef       = useRef<any>(null);
  const markersRef   = useRef<Record<Layer, any[]>>({
    cafe: [], charging: [], emergency: [], subway: [], restroom: [],
  });
  const infoWindowRef = useRef<any>(null);

  // Initialize map once
  useEffect(() => {
    const init = () => {
      if (!window.kakao || !containerRef.current) return;

      window.kakao.maps.load(() => {
        if (!containerRef.current) return;

        const map = new window.kakao.maps.Map(containerRef.current, {
          center: new window.kakao.maps.LatLng(37.5759, 126.9769),
          level: 4,
        });
        mapRef.current = map;
        infoWindowRef.current = new window.kakao.maps.CustomOverlay({
          zIndex: 10,
        });

        // Add zoom control
        map.addControl(
          new window.kakao.maps.ZoomControl(),
          window.kakao.maps.ControlPosition.RIGHT
        );

        // Static layers
        addStaticMarkers(map, "charging",  locationsData.charging);
        addStaticMarkers(map, "emergency", locationsData.emergency_tents);
        addStaticMarkers(map, "subway",    locationsData.subway_routes);
        addStaticMarkers(map, "restroom",  (locationsData as any).restrooms);

        // Cafe keyword search
        const ps = new window.kakao.maps.services.Places();
        ps.keywordSearch(
          "광화문 24시간 카페",
          (result: any[], status: string) => {
            if (status !== window.kakao.maps.services.Status.OK) return;
            const cafes = result.slice(0, 8).map((p: any) => {
              const overlay = createOverlay(
                map,
                LAYER_EMOJI.cafe,
                parseFloat(p.y),
                parseFloat(p.x),
                p.place_name,
                p.road_address_name || p.address_name
              );
              return overlay;
            });
            markersRef.current.cafe = cafes;
            // Apply current visibility
            const visible = activeLayers.has("cafe");
            cafes.forEach((o: any) => o.setMap(visible ? map : null));
          },
          {
            location: new window.kakao.maps.LatLng(37.5759, 126.9769),
            radius: 1200,
          }
        );
      });
    };

    // Poll until kakao SDK is available
    if (window.kakao) {
      init();
    } else {
      const timer = setInterval(() => {
        if (window.kakao) {
          clearInterval(timer);
          init();
        }
      }, 150);
      return () => clearInterval(timer);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Toggle layer visibility when activeLayers changes
  useEffect(() => {
    if (!mapRef.current) return;
    (Object.keys(markersRef.current) as Layer[]).forEach((layer) => {
      const visible = activeLayers.has(layer);
      markersRef.current[layer].forEach((overlay: any) =>
        overlay.setMap(visible ? mapRef.current : null)
      );
    });
  }, [activeLayers]);

  function addStaticMarkers(
    map: any,
    layer: Layer,
    items: Array<{ id: string; name: string; name_en?: string; lat: number; lng: number; memo_ko?: string; memo_en?: string; [key: string]: any }>
  ) {
    const overlays = items.map((item) => {
      const title = lang === "en" && item.name_en ? item.name_en : item.name;
      const memo  = lang === "en" ? (item.memo_en || "") : (item.memo_ko || "");
      return createOverlay(map, LAYER_EMOJI[layer], item.lat, item.lng, title, memo);
    });
    markersRef.current[layer] = overlays;
    // Show/hide based on initial activeLayers
    const visible = activeLayers.has(layer);
    overlays.forEach((o) => o.setMap(visible ? map : null));
  }

  function createOverlay(
    map: any,
    emoji: string,
    lat: number,
    lng: number,
    title: string,
    memo: string
  ): any {
    const position = new window.kakao.maps.LatLng(lat, lng);

    const markerEl = document.createElement("div");
    markerEl.innerHTML = markerContent(emoji);

    const overlay = new window.kakao.maps.CustomOverlay({
      map: null,
      position,
      content: markerEl,
      yAnchor: 1,
    });

    // Click: show info window
    markerEl.addEventListener("click", () => {
      if (!infoWindowRef.current) return;
      const existing = infoWindowRef.current.getMap();
      const isSame =
        infoWindowRef.current.getPosition()?.toString() === position.toString();

      if (existing && isSame) {
        infoWindowRef.current.setMap(null);
        return;
      }

      infoWindowRef.current.setContent(infoContent(title, memo));
      infoWindowRef.current.setPosition(position);
      infoWindowRef.current.setMap(map);
    });

    return overlay;
  }

  return (
    <div ref={containerRef} id="kakao-map" style={{ width: "100%", height: "100%" }} />
  );
}
