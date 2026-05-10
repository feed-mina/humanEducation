'use client';

import React, { useState, memo } from 'react';
import { cn } from "@/components/utils/cn";

const PasswordField = memo(({ meta, data, onChange, onAction, ...rest }: any) => {
    // 1. DOM에 전달되면 안 되는 속성들을 rest에서 미리 추출하여 분리함
    const {
        pwType,
        showPassword: _ignoredShowPassword, // 부모가 혹시 보낼지 모를 속성 차단
        className: _ignoredClassName,
        ...domSafeRest
    } = rest;

    const [showPassword, setShowPassword] = useState<boolean>(false);

    // 2. 데이터 매핑 키 및 읽기 전용 여부 판단
    const targetKey = meta?.componentId || meta?.ref_data_id || meta?.refDataId || "password";
    const isReadOnly = meta?.isReadonly === true || meta?.isReadonly === "true" ||
        meta?.is_readonly === true || meta?.is_readonly === "true";

    // 3. 인라인 스타일 안전하게 파싱
    let customStyle = {};
    try {
        customStyle = typeof meta?.inline_style === 'string'
            ? JSON.parse(meta.inline_style)
            : (meta?.inline_style || meta?.inlineStyle || {});
    } catch (e) {
        customStyle = {};
    }

    // 4. 클래스 병합
    const mergedClassName = cn(
        "password-field-input",
        meta?.cssClass,
        rest.className,
        isReadOnly && "is-readonly"
    );

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!isReadOnly && onChange) {
            onChange(targetKey, e.target.value);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !isReadOnly && onAction) {
            onAction(meta);
        }
    };

    return (
        <div className={cn("password-field-wrapper", isReadOnly && "read-only-wrapper")}
             style={{ position: 'relative', width: '100%' }}>
            <input
                {...domSafeRest} // 걸러진 안전한 속성들만 input에 전달
                type={showPassword ? 'text' : 'password'}
                placeholder={meta?.labelText || "비밀번호를 입력하세요"}
                className={mergedClassName}
                style={{
                    ...customStyle,
                    width: '100%',
                    paddingRight: isReadOnly ? '10px' : '70px',
                    boxSizing: 'border-box'
                }}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                readOnly={isReadOnly}
            />

            {!isReadOnly && (
                <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    style={{
                        position: 'absolute',
                        right: '10px',
                        top: '50%',
                        background: 'none',
                        border: 'none',
                        fontSize: '24px',
                        transform: 'translateY(-50%)',
                        cursor: 'pointer',
                        zIndex: 2,
                    }}
                >
                    {showPassword ? '👀' : '🕶️'}
                </button>
            )}
        </div>
    );
});

PasswordField.displayName = "PasswordField";
export default PasswordField;