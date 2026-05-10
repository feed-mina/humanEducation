"use client";

import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap, ZoomControl } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Layer as MapLayer } from "./LayerFilter";
import locationsData from "@/data/locations.json";

// Fix leaflet icon issue in Next.js
const DefaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

// Custom Marker Creator
const createCustomIcon = (emoji: string) => {
  return L.divIcon({
    html: `<div class="map-marker">${emoji}</div>`,
    className: "custom-leaflet-marker",
    iconSize: [30, 30],
    iconAnchor: [15, 30],
  });
};

interface Props {
  activeLayer: MapLayer | null;
  lang: "ko" | "en" | "ja";
}

const LAYER_EMOJI: Record<MapLayer, string> = {
  cafe:      "☕",
  charging:  "🔋",
  emergency: "🏥",
  subway:    "🚇",
  restroom:  "🚻",
};

export default function LeafletMap({ activeLayer, lang }: Props) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) return <div className="h-full w-full bg-[#1a1a2e]" />;

  const center: [number, number] = [37.5759, 126.9769];

  return (
    <div className="absolute inset-0 w-full h-full bg-white">
      <MapContainer 
        center={center} 
        zoom={16} 
        className="h-full w-full"
        style={{ height: "100%", width: "100%" }}
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
        />
        <ZoomControl position="bottomright" />

        {/* 24h Cafes */}
        {activeLayer === "cafe" && locationsData.cafes?.map((item: any) => (
          <Marker 
            key={item.id} 
            position={[item.lat, item.lng]} 
            icon={createCustomIcon(LAYER_EMOJI.cafe)}
          >
            <Popup className="custom-popup">
              <div className="font-bold text-indigo-900">{lang === "en" ? item.name_en : item.name}</div>
              <div className="text-xs text-gray-600 mt-1">{lang === "en" ? item.memo_en : item.memo_ko}</div>
            </Popup>
          </Marker>
        ))}

        {/* Charging Stations */}
        {activeLayer === "charging" && locationsData.charging?.map((item: any) => (
          <Marker 
            key={item.id} 
            position={[item.lat, item.lng]} 
            icon={createCustomIcon(LAYER_EMOJI.charging)}
          >
            <Popup className="custom-popup">
              <div className="font-bold text-green-700">{lang === "en" ? item.name_en : item.name}</div>
              <div className="text-xs text-gray-600 mt-1">{lang === "en" ? item.memo_en : item.memo_ko}</div>
            </Popup>
          </Marker>
        ))}

        {/* Emergency Tents */}
        {activeLayer === "emergency" && locationsData.emergency_tents?.map((item: any) => (
          <Marker 
            key={item.id} 
            position={[item.lat, item.lng]} 
            icon={createCustomIcon(LAYER_EMOJI.emergency)}
          >
            <Popup className="custom-popup">
              <div className="font-bold text-red-600">{lang === "en" ? item.name_en : item.name}</div>
              <div className="text-xs text-gray-600 mt-1">{lang === "en" ? item.memo_en : item.memo_ko}</div>
            </Popup>
          </Marker>
        ))}

        {/* Subway Routes */}
        {activeLayer === "subway" && locationsData.subway_routes?.map((item: any) => (
          <Marker 
            key={item.id} 
            position={[item.lat, item.lng]} 
            icon={createCustomIcon(LAYER_EMOJI.subway)}
          >
            <Popup className="custom-popup">
              <div className="font-bold text-blue-800">{lang === "en" ? item.name_en : item.name}</div>
              <div className="text-xs text-gray-600 mt-1">{lang === "en" ? item.memo_en : item.memo_ko}</div>
            </Popup>
          </Marker>
        ))}

        {/* Public Restrooms */}
        {activeLayer === "restroom" && (locationsData as any).restrooms?.map((item: any) => (
          <Marker 
            key={item.id} 
            position={[item.lat, item.lng]} 
            icon={createCustomIcon(LAYER_EMOJI.restroom)}
          >
            <Popup className="custom-popup">
              <div className="font-bold text-amber-700">{lang === "en" ? item.name_en : item.name}</div>
              <div className="text-xs text-gray-600 mt-1">{lang === "en" ? item.memo_en : item.memo_ko}</div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>

      {/* Map Decoration for Premiere Feel */}
      <div className="absolute inset-0 pointer-events-none border-[1px] border-white/5 z-[400]" />
    </div>
  );
}
