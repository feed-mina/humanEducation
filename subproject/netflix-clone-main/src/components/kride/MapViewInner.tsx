'use client';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect } from "react";

export interface RouteMarker {
  lat: number;
  lng: number;
  name: string;
  index: number;
}

interface Props {
  center?: [number, number];
  markers?: RouteMarker[];
  zoom?: number;
}

export default function MapViewInner({
  center = [37.5665, 126.978],
  markers = [],
  zoom = 13,
}: Props) {
  useEffect(() => {
    // leaflet 기본 아이콘 경로 수동 설정 (Next.js 빌드 시 깨지는 현상 방지)
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: "/leaflet/marker-icon-2x.png",
      iconUrl: "/leaflet/marker-icon.png",
      shadowUrl: "/leaflet/marker-shadow.png",
    });
  }, []);

  const positions: [number, number][] = markers.map((m) => [m.lat, m.lng]);

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      className="w-full h-full"
      style={{ background: "#1a1a2e" }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      />
      {positions.length > 1 && (
        <Polyline positions={positions} color="#e50914" weight={3} />
      )}
      {markers.map((m) => (
        <Marker key={m.index} position={[m.lat, m.lng]}>
          <Popup>
            <span className="font-bold text-red-600">{m.index + 1}.</span> {m.name}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
