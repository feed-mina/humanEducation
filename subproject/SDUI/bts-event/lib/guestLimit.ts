const KEY = "bts_guest_chat_count";
const MAX = 5;

export function getGuestChatCount(): number {
  if (typeof window === "undefined") return 0;
  return parseInt(localStorage.getItem(KEY) || "0", 10);
}

export function incrementGuestChatCount(): number {
  const next = getGuestChatCount() + 1;
  localStorage.setItem(KEY, String(next));
  return next;
}

export function hasGuestChatRemaining(): boolean {
  return getGuestChatCount() < MAX;
}

export function getGuestChatRemaining(): number {
  return Math.max(0, MAX - getGuestChatCount());
}
