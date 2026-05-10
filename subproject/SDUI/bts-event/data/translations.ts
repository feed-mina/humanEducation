import { type Lang } from "@/components/LangToggle";
import { type Layer } from "@/components/Map/LayerFilter";

export const translations = {
  ko: {
    title: "💜 BTS 광화문",
    map: "🗺️ 지도",
    chat: "🤖 채팅",
    board: "✍️ 게시판",
    traffic: "실시간 교통상황",
    weatherTitle: "실시간 날씨 (기상청)",
    congestion: "혼잡 (여유로움) 💜",
    moderate: "혼잡 (보통) 💜",
    crowded: "매우 혼잡 🔥",
    source: "기상청 실시간 상세정보",
    layers: {
      cafe: "24h 카페",
      charging: "🔋 충전",
      emergency: "🏥 구급",
      subway: "🚇 지하철",
      restroom: "🚻 화장실"
    },
    boardReady: "팬 게시판 준비 중",
    boardDesc: "전 세계 ARMY가 함께하는 게시판이 곧 오픈됩니다.",
    notice: "알림",
    close: "닫기",
    write: "글쓰기",
    update: "수정",
    submit: "등록",
    empty: "게시물이 없습니다. 첫 번째 글을 작성해보세요!",
    category: "구분",
    placeholderTitle: "제목을 입력하세요",
    placeholderContent: "내용을 입력하세요",
    boardTabs: {
      ALL: "전체",
      REPORT: "현황제보",
      LOST: "분실물",
      CHEER: "응원하기"
    },
    boardFields: {
      title: "제목",
      content: "내용",
      location: "위치 태그",
      tags: "태그 (#뷔 #화장실 등)",
      image: "이미지 첨부"
    },
    boardSuccess: "게시글이 성공적으로 등록되었습니다! 💜",
    copySuccess: "링크가 클립보드에 복사되었습니다! 💜",
    lastTrain: "📢 막차 정보",
    lastTrainTitle: "💜 귀가 길 안내 (Safe Home)",
    restroomStatus: "현재 화장실 대기 현황"
  },
  en: {
    title: "💜 BTS Gwanghwamun",
    map: "🗺️ Map",
    chat: "🤖 Chat",
    board: "✍️ Board",
    traffic: "Live Traffic Info",
    weatherTitle: "Live Weather (KMA)",
    congestion: "Moderate 💜",
    moderate: "Normal 💜",
    crowded: "Very Crowded 🔥",
    source: "Source: Weather.go.kr",
    layers: {
      cafe: "24h Cafe",
      charging: "🔋 Charge",
      emergency: "🏥 First Aid",
      subway: "🚇 Subway",
      restroom: "🚻 Restroom"
    },
    boardReady: "Fan Board Coming Soon",
    boardDesc: "A board for ARMYs worldwide will be open soon.",
    notice: "Notice",
    close: "Close",
    write: "Write",
    update: "Edit",
    submit: "Post",
    empty: "No posts yet. Be the first to share!",
    category: "Category",
    placeholderTitle: "Enter title",
    placeholderContent: "Enter content",
    boardTabs: {
      ALL: "All",
      REPORT: "Status",
      LOST: "Lost & Found",
      CHEER: "Cheer"
    },
    boardFields: {
      title: "Title",
      content: "Content",
      location: "Location Tag",
      tags: "Tags (e.g., #V #BTS)",
      image: "Attach Image"
    },
    boardSuccess: "Post registered successfully! 💜",
    copySuccess: "Link copied to clipboard! 💜",
    lastTrain: "📢 Last Trains",
    lastTrainTitle: "💜 Safe Home Guide",
    restroomStatus: "Restroom Wait Status"
  },
  ja: {
    title: "💜 BTS 光化門",
    map: "🗺️ 地図",
    chat: "🤖 チャット",
    board: "✍️ 掲示板",
    traffic: "リアルタイム交通정보",
    weatherTitle: "リアルタイム天気 (気象庁)",
    congestion: "混雑 (余裕) 💜",
    moderate: "混雑 (普通) 💜",
    crowded: "非常に混雑 🔥",
    source: "詳細: 気象庁",
    layers: {
      cafe: "24h カフェ",
      charging: "🔋 充電",
      emergency: "🏥 救急",
      subway: "🚇 地下鉄",
      restroom: "🚻 お手洗い"
    },
    boardReady: "ファン掲示板準備中",
    boardDesc: "世界中のARMY가 참가할 수 있는 掲示판이 곧 오픈됩니다.",
    notice: "お知らせ",
    close: "閉じる",
    write: "書く",
    update: "編集",
    submit: "登録",
    empty: "投稿がありません。最初の投稿をしてみましょう！",
    category: "区分",
    placeholderTitle: "タイトルを入力してください",
    placeholderContent: "内容を入力してください",
    boardTabs: {
      ALL: "すべて",
      REPORT: "現況報告",
      LOST: "落し物",
      CHEER: "応援"
    },
    boardFields: {
      title: "タイトル",
      content: "内容",
      location: "位置タグ",
      tags: "タグ (例: #V #トイレ)",
      image: "画像添付"
    },
    boardSuccess: "投稿が完了しました！ 💜",
    copySuccess: "リンクがクリップボードにコピーされました！ 💜",
    lastTrain: "📢 終電情報",
    lastTrainTitle: "💜 安全な帰り道案内 (Safe Home)",
    restroomStatus: "お手洗い待ち状況"
  }
};
