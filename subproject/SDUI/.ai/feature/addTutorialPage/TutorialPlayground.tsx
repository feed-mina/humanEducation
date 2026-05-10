'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';

// DynamicEngine을 동적으로 import하여 순환 참조 방지 및 클라이언트 전용 렌더링 보장
const DynamicEngine = dynamic(() => import('../DynamicEngine/DynamicEngine'), {
    ssr: false,
    loading: () => <div className="p-4 text-gray-400">엔진 로딩 중...</div>
});

// 초기 예제 데이터 (버튼 하나가 있는 상태)
const INITIAL_META = [
    {
        component_id: 'demo_title',
        component_type: 'TEXT',
        label_text: 'Hello, SDUI!',
        sort_order: 1,
        css_class: 'text-2xl font-bold text-gray-800 mb-4'
    },
    {
        component_id: 'demo_button',
        component_type: 'BUTTON',
        label_text: '클릭해보세요',
        action_type: 'TOAST', // 예시 액션
        sort_order: 2,
        css_class: 'px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition'
    }
];

export default function TutorialPlayground() {
    const [metaList, setMetaList] = useState<any[]>(INITIAL_META);
    const [selectedId, setSelectedId] = useState<string>('demo_button');
    const [showHistory, setShowHistory] = useState(false);
    const [historyList, setHistoryList] = useState<string[]>([]);
    const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
    const [isPreviewMode, setIsPreviewMode] = useState(false);

    // 현재 선택된 컴포넌트 찾기
    const selectedComponent = metaList.find(m => m.component_id === selectedId) || metaList[0];

    // 속성 변경 핸들러
    const handleAttributeChange = (key: string, value: string) => {
        setMetaList(prev => prev.map(item =>
            item.component_id === selectedId ? { ...item, [key]: value } : item
        ));
    };

    // 컴포넌트별 고유 속성(componentProps) 변경 핸들러
    const handlePropChange = (propKey: string, value: string) => {
        setMetaList(prev => prev.map(item => {
            if (item.component_id === selectedId) {
                const currentProps = item.componentProps || {};
                return { ...item, componentProps: { ...currentProps, [propKey]: value } };
            }
            return item;
        }));
    };

    // 새 컴포넌트 추가 핸들러
    const handleAddComponent = (type: string) => {
        const newId = `new_${type.toLowerCase()}_${Date.now()}`;

        let label = '새 컴포넌트';
        let css = '';
        let extraProps = {};

        if (type === 'BUTTON') {
            label = '새 버튼';
            css = 'px-4 py-2 bg-blue-500 text-white rounded-lg mt-2';
        } else if (type === 'TEXT') {
            label = '새 텍스트';
            css = 'text-base text-gray-600 mt-2';
        } else if (type === 'INPUT') {
            label = '새 입력창';
            css = 'w-full p-2 border border-gray-300 rounded mt-2 text-sm';
            extraProps = { componentProps: { placeholder: '내용을 입력하세요...' } };
        } else if (type === 'IMAGE') {
            label = '샘플 이미지';
            css = 'w-full h-40 object-cover rounded-lg mt-2 bg-gray-100';
            extraProps = { componentProps: { src: 'https://via.placeholder.com/400x200' } };
        }

        const newComponent = {
            component_id: newId,
            component_type: type,
            label_text: label,
            sort_order: metaList.length + 1,
            css_class: css,
            ...extraProps
        };
        setMetaList([...metaList, newComponent]);
        setSelectedId(newId);
    };

    // DB 저장 핸들러
    const handleSaveToDB = async () => {
        if (!confirm('현재 구성을 DB에 저장하시겠습니까?')) return;

        try {
            const response = await fetch('/api/ui/tutorial/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(metaList)
            });

            if (!response.ok) throw new Error('저장 실패');
            alert('성공적으로 저장되었습니다.');
        } catch (error) {
            console.error('Save failed:', error);
            alert('저장에 실패했습니다. 백엔드 로그를 확인하세요.');
        }
    };

    // 데모 페이지 이동 핸들러
    const handleViewDemo = () => {
        window.open('/view/TUTORIAL_DEMO', '_blank');
    };

    // 히스토리 목록 조회
    const handleOpenHistory = async () => {
        try {
            const response = await fetch('/api/ui/tutorial/history');
            const json = await response.json();
            // ApiResponse 구조에 따라 json.data 또는 json.body 등 확인 필요 (여기선 json.data 가정)
            if (json.data) {
                setHistoryList(json.data);
                setShowHistory(true);
            }
        } catch (error) {
            console.error('History fetch failed:', error);
            alert('히스토리 목록을 불러오지 못했습니다.');
        }
    };

    // 특정 버전 불러오기
    const handleLoadVersion = async (versionId: string) => {
        if (!confirm(`[${versionId}] 버전을 불러오시겠습니까?\n현재 작업 중인 내용은 사라집니다.`)) return;
        try {
            const response = await fetch(`/api/ui/tutorial/history/${versionId}`);
            const json = await response.json();
            if (json.data) {
                setMetaList(json.data);
                setShowHistory(false);
                alert('버전을 성공적으로 불러왔습니다.');
            }
        } catch (error) {
            console.error('Version load failed:', error);
            alert('버전 로드에 실패했습니다.');
        }
    };

    // 드래그 앤 드롭 핸들러: 시작
    const handleDragStart = (e: React.DragEvent, index: number) => {
        setDraggedIndex(index);
        e.dataTransfer.effectAllowed = "move";
    };

    // 드래그 앤 드롭 핸들러: 이동 중 (실시간 순서 변경)
    const handleDragOver = (e: React.DragEvent, index: number) => {
        e.preventDefault(); // 드롭 허용
        if (draggedIndex === null || draggedIndex === index) return;

        const newList = [...metaList];
        const draggedItem = newList[draggedIndex];

        // 배열 재배치
        newList.splice(draggedIndex, 1);
        newList.splice(index, 0, draggedItem);

        setDraggedIndex(index);
        setMetaList(newList);
    };

    // 드래그 앤 드롭 핸들러: 종료 (sort_order 확정)
    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDraggedIndex(null);

        // sort_order 재할당 (1부터 시작)
        setMetaList(prev => prev.map((item, idx) => ({
            ...item,
            sort_order: idx + 1
        })));
    };

    return (
        <div className="flex flex-col h-[calc(100vh-100px)] bg-white rounded-xl shadow-sm overflow-hidden border border-gray-200">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b bg-gray-50">
                <div>
                    <h2 className="text-lg font-bold text-gray-800">SDUI Playground</h2>
                    <p className="text-xs text-gray-500">좌측 UI를 수정하면 우측 JSON이 실시간으로 변경됩니다.</p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => handleAddComponent('BUTTON')}
                        className="px-3 py-1.5 text-xs font-semibold bg-white border border-gray-300 rounded hover:bg-gray-50"
                    >
                        + 버튼
                    </button>
                    <button
                        onClick={() => handleAddComponent('TEXT')}
                        className="px-3 py-1.5 text-xs font-semibold bg-white border border-gray-300 rounded hover:bg-gray-50"
                    >
                        + 텍스트
                    </button>
                    <button
                        onClick={() => handleAddComponent('INPUT')}
                        className="px-3 py-1.5 text-xs font-semibold bg-white border border-gray-300 rounded hover:bg-gray-50"
                    >
                        + 입력창
                    </button>
                    <button
                        onClick={() => handleAddComponent('IMAGE')}
                        className="px-3 py-1.5 text-xs font-semibold bg-white border border-gray-300 rounded hover:bg-gray-50"
                    >
                        + 이미지
                    </button>
                    <button
                        onClick={() => setIsPreviewMode(!isPreviewMode)}
                        className={`px-3 py-1.5 text-xs font-semibold border rounded transition-colors ${isPreviewMode
                                ? 'bg-gray-800 text-white border-gray-800 hover:bg-gray-700'
                                : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                            }`}
                    >
                        {isPreviewMode ? '편집 모드' : '미리보기'}
                    </button>
                    <button
                        onClick={handleSaveToDB}
                        className="px-3 py-1.5 text-xs font-semibold text-white bg-blue-600 border border-blue-600 rounded hover:bg-blue-700"
                    >
                        적용
                    </button>
                    <button
                        onClick={handleOpenHistory}
                        className="px-3 py-1.5 text-xs font-semibold text-gray-700 bg-yellow-100 border border-yellow-300 rounded hover:bg-yellow-200"
                    >
                        히스토리
                    </button>
                    <button
                        onClick={handleViewDemo}
                        className="px-3 py-1.5 text-xs font-semibold text-white bg-purple-600 border border-purple-600 rounded hover:bg-purple-700"
                    >
                        데모 보기
                    </button>
                    <button
                        onClick={() => setMetaList(INITIAL_META)}
                        className="px-3 py-1.5 text-xs font-semibold text-red-600 bg-white border border-red-200 rounded hover:bg-red-50"
                    >
                        초기화
                    </button>
                </div>
            </div>

            {/* Body: Split View */}
            <div className="flex flex-1 overflow-hidden">

                {/* Left: Preview (UI) */}
                <div className={`${isPreviewMode ? 'w-full' : 'w-1/2 border-r'} p-8 bg-gray-50 overflow-y-auto border-gray-200 transition-all duration-300`}>
                    <div className="mb-4 text-xs font-bold text-gray-400 uppercase tracking-wider">
                        Real-time Preview
                    </div>
                    <div className="bg-white p-6 rounded-xl shadow-sm min-h-[400px] border border-gray-100">
                        {/* DynamicEngine에 실시간 메타데이터 주입 */}
                        {/* 주의: DynamicEngine이 metadata prop을 지원해야 함. 
                            지원하지 않을 경우 useDynamicEngine 훅을 사용하여 treeData로 변환 필요 */}
                        <DynamicEngine
                            metadata={metaList}
                            pageData={{}}
                            formData={{}}
                        />
                    </div>
                </div>

                {/* Right: Editor & JSON */}
                {!isPreviewMode && (
                    <div className="w-1/2 flex flex-col bg-white">

                        {/* Controls */}
                        <div className="p-6 border-b border-gray-100 bg-white">
                            <div className="mb-4 text-xs font-bold text-gray-400 uppercase tracking-wider">
                                Component Properties
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-gray-500 mb-1">Layers (Drag to Reorder)</label>
                                    <ul className="w-full max-h-40 overflow-y-auto border border-gray-300 rounded bg-white">
                                        {metaList.map((m, index) => (
                                            <li
                                                key={m.component_id}
                                                draggable
                                                onDragStart={(e) => handleDragStart(e, index)}
                                                onDragOver={(e) => handleDragOver(e, index)}
                                                onDrop={handleDrop}
                                                onClick={() => setSelectedId(m.component_id)}
                                                className={`
                                                px-3 py-2 text-sm cursor-pointer border-b border-gray-100 last:border-0 flex items-center justify-between
                                                ${selectedId === m.component_id ? 'bg-blue-50 text-blue-600 font-medium' : 'hover:bg-gray-50 text-gray-700'}
                                                ${draggedIndex === index ? 'opacity-50 bg-gray-100' : ''}
                                            `}
                                            >
                                                <div className="flex items-center gap-2 truncate">
                                                    <span className="text-gray-400 cursor-move">☰</span>
                                                    <span>{m.component_id}</span>
                                                </div>
                                                <span className="text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded ml-2 shrink-0">{m.component_type}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-gray-500 mb-1">Label Text</label>
                                    <input
                                        type="text"
                                        value={selectedComponent?.label_text || ''}
                                        onChange={(e) => handleAttributeChange('label_text', e.target.value)}
                                        className="w-full p-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-green-500 outline-none"
                                    />
                                </div>
                                <div className="col-span-2">
                                    <label className="block text-xs font-medium text-gray-500 mb-1">CSS Class (Tailwind)</label>
                                    <input
                                        type="text"
                                        value={selectedComponent?.css_class || ''}
                                        onChange={(e) => handleAttributeChange('css_class', e.target.value)}
                                        className="w-full p-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-green-500 outline-none font-mono text-xs"
                                    />
                                </div>
                                {selectedComponent?.component_type === 'INPUT' && (
                                    <div className="col-span-2">
                                        <label className="block text-xs font-medium text-gray-500 mb-1">Placeholder</label>
                                        <input
                                            type="text"
                                            value={selectedComponent.componentProps?.placeholder || ''}
                                            onChange={(e) => handlePropChange('placeholder', e.target.value)}
                                            className="w-full p-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-green-500 outline-none"
                                        />
                                    </div>
                                )}
                                {selectedComponent?.component_type === 'IMAGE' && (
                                    <div className="col-span-2">
                                        <label className="block text-xs font-medium text-gray-500 mb-1">Image URL (src)</label>
                                        <input
                                            type="text"
                                            value={selectedComponent.componentProps?.src || ''}
                                            onChange={(e) => handlePropChange('src', e.target.value)}
                                            className="w-full p-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-green-500 outline-none font-mono text-xs"
                                        />
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* JSON View */}
                        <div className="flex-1 p-0 overflow-hidden flex flex-col bg-[#1e1e1e]">
                            <div className="px-4 py-2 bg-[#252526] text-xs font-bold text-gray-400 border-b border-[#333] flex justify-between">
                                <span>JSON Data (Server Response Simulation)</span>
                                <span className="text-green-500">● Live</span>
                            </div>
                            <div className="flex-1 overflow-auto p-4">
                                <pre className="text-xs font-mono text-[#d4d4d4] leading-relaxed whitespace-pre-wrap">
                                    {JSON.stringify(metaList, null, 2)}
                                </pre>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* History Modal */}
            {showHistory && (
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50">
                    <div className="bg-white rounded-lg shadow-xl w-96 max-h-[500px] flex flex-col">
                        <div className="flex items-center justify-between px-4 py-3 border-b">
                            <h3 className="font-bold text-gray-800">저장된 히스토리</h3>
                            <button onClick={() => setShowHistory(false)} className="text-gray-500 hover:text-gray-700">✕</button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-2">
                            {historyList.length === 0 ? (
                                <p className="text-center text-gray-400 py-4 text-sm">저장된 히스토리가 없습니다.</p>
                            ) : (
                                <ul className="space-y-1">
                                    {historyList.map((version) => (
                                        <li key={version}>
                                            <button
                                                onClick={() => handleLoadVersion(version)}
                                                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-blue-50 rounded border border-transparent hover:border-blue-100 transition-colors"
                                            >
                                                {version}
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}