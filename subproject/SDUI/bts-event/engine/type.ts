/**
 * SDUI 메타데이터 타입 정의
 * metadata-project의 type.ts 기반, bts-event에 맞게 확장
 * 
 * DB 전환 시 ui_metadata 테이블의 컬럼과 1:1 매핑됩니다.
 * snake_case(DB 컬럼) / camelCase(프론트) 모두 지원합니다.
 */

export interface Metadata {
  // === 식별 ===
  uiId?: number | string;
  componentId: string;
  component_id?: string;

  // === 컴포넌트 타입 (componentMap 키) ===
  componentType: string;
  component_type?: string;

  // === 그룹 계층 ===
  groupId?: string | null;
  group_id?: string | null;
  parentGroupId?: string | null;
  parent_group_id?: string | null;

  // === 데이터 바인딩 ===
  refDataId?: string;
  ref_data_id?: string;

  // === 표시 제어 ===
  isVisible?: boolean | string;
  is_visible?: boolean | string;

  // === 레이아웃 ===
  groupDirection?: "ROW" | "COLUMN";
  group_direction?: "ROW" | "COLUMN";
  cssClass?: string;
  css_class?: string;
  inlineStyle?: Record<string, string>;

  // === 액션 ===
  actionType?: string;
  action_type?: string;

  // === 필드 속성 ===
  labelText?: string;
  label_text?: string;
  placeholder?: string;
  isReadonly?: boolean | string;
  is_readonly?: boolean | string;

  // === 이벤트 전용: 컴포넌트별 커스텀 props ===
  props?: Record<string, any>;

  // === 트리 구조 ===
  children?: Metadata[] | null;

  // === 확장 필드 ===
  [key: string]: any;
}

export interface DynamicEngineProps {
  metadata: Metadata[];
  screenId: string;
  pageData: Record<string, any>;
  formData: Record<string, any>;
  setFormData?: (value: any | ((prev: any) => any)) => void;
  onChange: (id: string, value: any) => void;
  onAction: (meta: Metadata, data?: any) => void;
  activeModal?: string | null;
  closeModal?: () => void;
  [key: string]: any;
}

export interface ScreenMetadataResponse {
  data: Metadata[];
  success: boolean;
}
