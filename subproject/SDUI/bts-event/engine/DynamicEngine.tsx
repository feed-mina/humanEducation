/**
 * DynamicEngine — bts-event SDUI 렌더링 엔진
 * 
 * metadata-project의 DynamicEngine.tsx 패턴을 따릅니다:
 * 1. MetadataProvider에서 메타데이터 트리를 받음
 * 2. 트리를 순회하며 componentMap에서 컴포넌트를 조회
 * 3. EventStateProvider에서 공유 상태(lang, tab 등)를 주입
 * 4. 그룹/리피터 패턴 지원
 * 
 * DB 전환 시 이 파일은 수정 없이 그대로 사용됩니다.
 */
"use client";

import React, { useMemo, useState } from "react";
import { componentMap } from "./componentMap";
import { useDynamicEngine } from "./useDynamicEngine";
import { useMetadata } from "./MetadataProvider";
import { useEventState } from "./EventStateProvider";
import { Metadata } from "./type";

const DynamicEngine: React.FC = () => {
  const { menuTree, isLoading, screenId } = useMetadata();
  const eventState = useEventState();
  const [formData, setFormData] = useState<Record<string, any>>({});

  // pageData: EventState의 값을 pageData로 변환
  const pageData: Record<string, any> = useMemo(() => ({
    lang: eventState.lang,
    tab: eventState.tab,
    activeLayer: eventState.activeLayer,
    showNotice: eventState.showNotice,
    showCheer: eventState.showCheer,
    showSupport: eventState.showSupport,
  }), [eventState]);

  const { treeData, getComponentData } = useDynamicEngine(menuTree, pageData, formData);

  // onChange: formData 업데이트
  const handleChange = (id: string, value: any) => {
    setFormData((prev) => ({ ...prev, [id]: value }));
  };

  // onAction: 메타데이터의 actionType에 따라 상태 변경
  const handleAction = (meta: Metadata, data?: any) => {
    const actionType = (meta.actionType || meta.action_type || "").toUpperCase();

    switch (actionType) {
      case "SET_TAB":
        eventState.setTab(data?.tab || meta.props?.tab || "map");
        break;
      case "SET_LANG":
        eventState.setLang(data?.lang || meta.props?.lang || "ko");
        break;
      case "TOGGLE_NOTICE":
        eventState.setShowNotice(!eventState.showNotice);
        break;
      case "SHOW_NOTICE":
        eventState.setShowNotice(true);
        break;
      case "CLOSE_NOTICE":
        eventState.setShowNotice(false);
        break;
      case "TOGGLE_CHEER":
        eventState.setShowCheer(!eventState.showCheer);
        break;
      case "SHOW_CHEER":
        eventState.setShowCheer(true);
        break;
      case "CLOSE_CHEER":
        eventState.setShowCheer(false);
        break;
      case "TOGGLE_SUPPORT":
        eventState.setShowSupport(!eventState.showSupport);
        break;
      case "SHOW_SUPPORT":
        eventState.setShowSupport(true);
        break;
      case "CLOSE_SUPPORT":
        eventState.setShowSupport(false);
        break;
      default:
        console.warn(`[DynamicEngine] 알 수 없는 액션: ${actionType}`);
    }
  };

  /**
   * 메타데이터 노드에서 컴포넌트로 전달할 props를 구성합니다.
   * 각 컴포넌트는 자신이 필요한 props를 선택적으로 사용합니다.
   */
  const buildComponentProps = (node: Metadata): Record<string, any> => {
    const customProps = node.props || {};
    return {
      // 공통 상태
      lang: eventState.lang,
      tab: eventState.tab,
      activeLayer: eventState.activeLayer,
      // 콜백
      onChange: (lang: any) => eventState.setLang(lang),
      onCheer: () => eventState.setShowCheer(true),
      onClose: () => {
        // 모달 컴포넌트가 닫힐 때 적절한 상태를 false로
        const type = (node.componentType || node.component_type || "").toUpperCase();
        if (type.includes("NOTICE")) eventState.setShowNotice(false);
        else if (type.includes("CHEER")) eventState.setShowCheer(false);
        else if (type.includes("SUPPORT")) eventState.setShowSupport(false);
      },
      // 메타데이터에서 정의한 커스텀 props
      ...customProps,
    };
  };

  // 트리 순회 렌더링
  const renderNodes = (nodes?: Metadata[] | null, rowData: any = null): React.ReactNode => {
    if (!nodes) return null;

    return nodes.map((node) => {
      // 가시성 체크
      const isVisible = node.isVisible !== false && node.isVisible !== "false" &&
        node.is_visible !== false && node.is_visible !== "false";
      if (!isVisible) return null;

      const rawId = node.componentId || node.component_id || node.uiId;
      const uId = String(rawId);

      // 조건부 렌더링: showWhen 필드로 상태 기반 표시/숨기기
      if (node.props?.showWhen) {
        const condition = node.props.showWhen;
        const stateValue = (pageData as any)[condition.field];
        if (stateValue !== condition.equals) return null;
      }

      // 그룹 노드 처리
      const isGroup =
        (node.componentType || node.component_type || "").toUpperCase() === "GROUP" ||
        (node.children && node.children.length > 0);

      if (isGroup) {
        const classList: string[] = [];
        const refId = node.refDataId || node.ref_data_id;
        const isRepeater = !!refId;
        const cid = node.componentId || node.component_id;
        const customClass = node.cssClass || node.css_class;
        const directionClass = (node.groupDirection || node.group_direction) === "ROW"
          ? "flex flex-row" : "flex flex-col";

        if (cid) classList.push(`group-${cid}`);
        if (customClass) classList.push(customClass);
        classList.push(directionClass);

        const combinedClassName = [...new Set(classList)].filter(Boolean).join(" ").trim();
        const hasAction = !!(node.actionType || node.action_type);

        // 리피터
        if (isRepeater) {
          const list = pageData?.[refId];
          if (!list || !Array.isArray(list)) return null;
          return list.map((item: any, idx: number) => {
            const handleClick = hasAction ? () => handleAction(node, item) : undefined;
            return (
              <div
                key={`${uId}-${idx}`}
                className={combinedClassName}
                style={{ cursor: hasAction ? "pointer" : "default" }}
                onClick={handleClick}
              >
                {renderNodes(node.children, item)}
              </div>
            );
          });
        }

        // 일반 그룹
        const handleGroupClick = hasAction ? () => handleAction(node, rowData) : undefined;
        return (
          <div
            key={uId}
            className={combinedClassName}
            style={{ cursor: hasAction ? "pointer" : "default" }}
            onClick={handleGroupClick}
          >
            {renderNodes(node.children, rowData)}
          </div>
        );
      }

      // 리프 노드 (개별 컴포넌트)
      const typeKey = (node.componentType || node.component_type || "").toUpperCase();
      const Component = componentMap[typeKey];
      if (!Component) {
        console.warn(`[DynamicEngine] 등록되지 않은 컴포넌트: ${typeKey}`);
        return null;
      }

      const componentProps = buildComponentProps(node);
      const finalData = getComponentData(node, rowData);

      return (
        <Component
          key={uId}
          id={uId}
          meta={node}
          data={finalData}
          {...componentProps}
        />
      );
    });
  };

  // 로딩 상태
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full bg-[#1a1a2e] text-white/50">
        💜 로딩 중...
      </div>
    );
  }

  return (
    <div className="engine-container flex flex-col h-full relative font-sans">
      {renderNodes(treeData)}
    </div>
  );
};

export default DynamicEngine;
