'use client';

import React, { useState, memo } from 'react';
import { cn } from "@/components/utils/cn";

interface SelectFieldProps {
    id: string;
    meta: any;
    data: any;
    onChange: (id: string, value: string) => void;
    [key: string]: any;
}

const SelectField = memo(({ id, meta, data, onChange, ...rest }: SelectFieldProps) => {
    const [isDirect, setIsDirect] = useState(false);
    const options = ["naver.com", "gmail.com", "nate.com", "hanmail.net", "직접 입력"];

    // 1. 데이터 매핑 키 및 값 추출
    const targetKey = meta?.ref_data_id || meta?.refDataId || id;
    const value = (typeof data === 'string') ? data : (data?.[targetKey] || "");

    // 2. 읽기 전용 여부 판단
    const isReadOnly = meta?.isReadonly === true || meta?.isReadonly === "true" ||
        meta?.is_readonly === true || meta?.is_readonly === "true";

    // 3. 클래스 병합
    const containerClass = cn(
        "select-field-wrapper",
        meta?.css_class,
        rest.className,
        isReadOnly && "is-readonly"
    );

    const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value;
        const directMode = val === "직접 입력";
        setIsDirect(directMode);

        if (!isReadOnly) {
            onChange(targetKey, directMode ? "" : val);
        }
    };

    return (
        <div className={containerClass}>
            {meta?.labelText && <span className="select-label" style={{ fontWeight: 'bold' }}>{meta.labelText}</span>}

            {isReadOnly ? (
                // 읽기 전용 상태: 현재 선택된 값을 텍스트로 보여줌
                <div className="readonly-select-text">
                    {value || "이메일 미선택"}
                </div>
            ) : (
                // 편집 가능 상태
                <div className="select-input-group">
                    <select
                        {...rest}
                        id={id}
                        value={isDirect ? "직접 입력" : (options.includes(value) ? value : (value ? "직접 입력" : ""))}
                        className={cn("select-element", isReadOnly && "is-readonly")}
                        onChange={handleSelectChange}
                    >
                        <option value="">이메일 선택</option>
                        {options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>

                    {(isDirect || (!options.includes(value) && value)) && (
                        <input
                            type="text"
                            className="domain-input"
                            value={value}
                            placeholder="도메인 입력"
                            onChange={(e) => onChange(targetKey, e.target.value)}
                        />
                    )}
                </div>
            )}
        </div>
    );
});

SelectField.displayName = "SelectField";
export default SelectField;