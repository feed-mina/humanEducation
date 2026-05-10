// [메모] component_id 와 componentId 이렇게 있을때 server와 redis, db 부분을 고려해서 하나로 고정할수 있는지 알아봐주세요. 
export interface Metadata {
  componentId: string;
  component_id: string;
  componentType: string;
  component_type?: string;
  parentGroupId?: string | null;
  parent_group_id?: string | null;
  groupId?: string | null;
  group_id?: string | null;
  refDataId?: string;
  ref_data_id?: string;
  isVisible?: boolean | string;
  is_visible?: boolean | string;
  groupDirection?: "ROW" | "COLUMN";
  cssClass?: string;
  css_class?: string;
  inlineStyle?: any;
  actionType?: string;
  action_type?: string;
  actionUrl?: string;
  action_url?: string;
  placeholder?: string;
  uiId?: string;
  labelText?: string;
  label_text?: string;
  isReadonly?: boolean | string;
  is_readonly?: boolean | string;
  children?: Metadata[] | null;
  [key: string]: any;
}

export interface DynamicEngineProps {
  metadata: Metadata[];
  screenId: string;
  pageData: any;
  formData?: any;
  setFormData?: (value: any | ((prev: any) => any)) => void;
  onChange: (id: string, value: any) => void;
  onAction: (meta: Metadata, data?: any) => void;
  activeModal?: string | null;
  closeModal?: () => void;
  onConfirmModal?: () => void;
  [key: string]: any;
}
