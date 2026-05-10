/**
 * screenMap — URL 경로를 screenId로 매핑
 * DB 전환 시 ui_metadata.screen_id 값과 동일하게 유지합니다.
 */

export const SCREEN_MAP: Record<string, string> = {
  "/": "BTS_EVENT_MAIN",
  "/BTS_EVENT_MAIN": "BTS_EVENT_MAIN",
};

export const DEFAULT_SCREEN_ID = "BTS_EVENT_MAIN";
