'use client';

import React, { memo } from "react";
import { cn } from "@/components/utils/cn";

interface ButtonFieldProps {
    meta: any;
    data?: any;
    onAction: (meta: any, data?: any) => void;
    [key: string]: any;
}

const ButtonField: React.FC<ButtonFieldProps> = memo(({ meta, data, onAction, ...rest }) => {
    // 1. DOM에 직접 전달되면 안 되는 속성들을 분리 (거름망 역할)
    const {
        pwType,
        showPassword,
        className: _ignoredClassName,
        ...domSafeRest
    } = rest;

    // 2. 라벨 설정
    const label = meta?.labelText || meta?.label_text || "버튼";

    // 3. 읽기 전용 여부 판단
    const isReadOnly = meta?.isReadonly === true || meta?.isReadonly === "true" ||
        meta?.is_readonly === true || meta?.is_readonly === "true";

    // 4. 클래스 병합
    const mergedClassName = cn(
        "content-btn",
        rest.className,
        meta?.cssClass,
        meta?.css_class,
        isReadOnly && "is-readonly"
    );

    const handleAction = () => {
        if (!isReadOnly && onAction) {
            onAction(meta, data);
        }
    };

    return (
        <button
            {...domSafeRest} // 정제된 안전한 속성들만 전달
            type="button"
            className={mergedClassName}
            disabled={isReadOnly}
            onClick={handleAction}
        >
            {label}
        </button>
    );
});

ButtonField.displayName = "ButtonField";
export default ButtonField;