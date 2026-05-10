// @@@@ 2026-02-07 Metadata 인인터페이스 타입 정의

export interface Metadata {
    componentId: string;
    component_id: string;
    componentType: string;
    component_type?: string;
    parentGroupId?: string | null;
    parent_group_id?: string | null;
    groupId?: string | null;
    group_id?: string | null; // 추가
    refDataId?: string;
    ref_data_id?: string; //
    isVisible?: boolean | string;
    is_visible?: boolean | string;
    groupDirection?: "ROW" | "COLUMN";
    cssClass?: string;
    css_class?: string;
    inlineStyle?: any;
    actionType?: string;
    action_type?: string;
    placeholder?: string;
    uiId?: string;
    labelText?: string;
    isReadonly?: boolean | string;
    is_readonly?: boolean | string;
    children?: Metadata[] | null;
}

export interface DynamicEngineProps {
    metadata: Metadata[];   // 트리 구조로 분석되기 전의 원본 메타데이터 배열
    screenId: string;       // @@@@ 현재 화면의 고유 식별자 추가
    pageData: any;          // 서버에서 가져온 실제 비즈니스 데이터
    formData: any;          // 사용자가 입력 중인 폼 데이터
    setFormData?: (value: any | ((prev: any) => any)) => void; // formData 업데이트 함수
    onChange: (id: string, value: any) => void;
    onAction: (meta: Metadata, data?: any) => void;// [추가] 모달 관련 프로퍼티 정의
    activeModal?: string | null;
    closeModal?: () => void;
    onConfirmModal?: () => void;
    pwType?: string;
    showPassword?: boolean;
    [key: string]: any;
}