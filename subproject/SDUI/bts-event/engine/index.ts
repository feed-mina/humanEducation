/**
 * engine/index.ts — 엔진 모듈 공개 API
 */
export { default as DynamicEngine } from "./DynamicEngine";
export { MetadataProvider, useMetadata } from "./MetadataProvider";
export { EventStateProvider, useEventState } from "./EventStateProvider";
export { componentMap } from "./componentMap";
export { SCREEN_MAP, DEFAULT_SCREEN_ID } from "./screenMap";
export type { Metadata, DynamicEngineProps, ScreenMetadataResponse } from "./type";
