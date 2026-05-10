import React, { memo } from 'react';
import {useRenderCount} from "@/components/DynamicEngine/hook/useRenderCount";
import '../../app/styles/TileSlot.css'

const DEFAULT_SLOT_DATA = { morning: '', lunch: '', evening: '' };
// @@@@ 2026-02-17 시간대별 범용 기록 컴포넌트
const TimeSlotRecord: React.FC<any> = ({ id, meta, data, onChange }) => {
    // 1. DB의 JSONB(component_props)에서 테마 설정을 가져옴
    const {
        title = "하루 일과",
        description = "시간대별로 기록해보세요.",
        placeholders = { morning: "아침 기록", lunch: "점심 기록", evening: "저녁 기록" }
    } = meta.component_props || {};

    // 2. 현재 저장된 데이터 (없으면 빈 객체)
    const slotData = data || DEFAULT_SLOT_DATA;
    const updateKey = meta.refDataId || meta.ref_data_id || id;

    const handleInputChange = (slot: string, value: string) => {
        // 3. 불변성을 유지하며 부모의 formData 업데이트
        onChange(updateKey, {
            morning: slotData.morning || '',
            lunch: slotData.lunch || '',
            evening: slotData.evening || '',
            [slot]: value
        });
    };

    const isVisible = meta.isVisible !== false && meta.isVisible !== "false";
    if (!isVisible) return null;

    const slots = [
        { key: 'morning', label: '아침' },
        { key: 'lunch', label: '점심' },
        { key: 'evening', label: '저녁' }
    ];

    return (<div className="time-slot-container">
            <h3 className="time-slot-title">{title}</h3>
            <p className="time-slot-description">{description}</p>
            <div className="time-slot-grid">
                {slots.map((slot) => (
                    <div key={slot.key} className="time-slot-wrapper">
                        <label className="time-slot-label">{slot.label}</label>
                        <input
                            type="text"
                            className="time-slot-input"
                            value={slotData[slot.key] || ''}
                            onChange={(e) => handleInputChange(slot.key, e.target.value)}
                            placeholder={placeholders[slot.key]}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
};

export default memo(TimeSlotRecord);