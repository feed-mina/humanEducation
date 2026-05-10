/**
 * componentMap — bts-event 전용 컴포넌트 매핑
 * 
 * DB 전환 시 ui_metadata.component_type 값과 동일한 키를 사용합니다.
 * 새 컴포넌트 추가 시 여기에 등록하면 JSON(또는 DB)에서 바로 사용 가능합니다.
 */

import dynamic from "next/dynamic";
import StatusCard from "@/components/StatusCard";
import InfoPanel from "@/components/InfoPanel";
import NoticeModal from "@/components/NoticeModal";
import LangToggle from "@/components/LangToggle";
import LivePip from "@/components/LivePip";
import CheerMode from "@/components/CheerMode";
import SupportModal from "@/components/SupportModal";
import LastTrainModal from "@/components/LastTrainModal";

// Leaflet은 SSR 불가 → dynamic import
const LeafletMap = dynamic(() => import("@/components/Map/LeafletMap"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-[#1a1a2e] text-white/50">
      🗺️ 지도 로딩 중...
    </div>
  ),
});

const LayerFilter = dynamic(() => import("@/components/Map/LayerFilter"), { ssr: false });
const GuestChat = dynamic(() => import("@/components/Chat/GuestChat"), { ssr: false });
const FanBoard = dynamic(() => import("@/components/Board/FanBoard"), { ssr: false });

/**
 * 컴포넌트 맵
 * 키: component_type (DB 컬럼 값)
 * 값: React 컴포넌트
 */
export const componentMap: Record<string, React.ComponentType<any>> = {
  // === 이벤트 전용 ===
  EVENT_MAP: LeafletMap,
  STATUS_CARD: StatusCard,
  INFO_PANEL: InfoPanel,
  NOTICE_MODAL: NoticeModal,
  LAYER_FILTER: LayerFilter,
  GUEST_CHAT: GuestChat,
  FAN_BOARD: FanBoard,
  LANG_TOGGLE: LangToggle,
  LIVE_PIP: LivePip,
  CHEER_MODE: CheerMode,
  SUPPORT_MODAL: SupportModal,
  LAST_TRAIN_MODAL: LastTrainModal,
};
