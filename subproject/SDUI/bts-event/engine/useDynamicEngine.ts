/**
 * useDynamicEngine — 데이터 바인딩 훅
 * metadata-project의 useDynamicEngine.tsx 패턴과 동일한 우선순위:
 *   formData(사용자 입력) > rowData(리피터 행) > pageData(서버 데이터)
 */

import { Metadata } from "./type";

export const useDynamicEngine = (
  metadata: Metadata[],
  pageData: Record<string, any>,
  formData: Record<string, any>
) => {
  const treeData = metadata;

  /**
   * 컴포넌트에 전달할 데이터를 결정합니다.
   * 
   * 1순위: formData[refId] — 사용자가 입력 중인 데이터
   * 2순위: rowData — 리피터 내 개별 항목 데이터
   * 3순위: pageData[refId] — 서버에서 가져온 데이터
   * 4순위: pageData 전체 — fallback
   */
  const getComponentData = (node: Metadata, rowData: any) => {
    const refId = node.refDataId || node.ref_data_id;

    // 1순위: 사용자 입력값 (formData)
    if (refId && formData && formData[refId] !== undefined) {
      return formData[refId];
    }

    // 2순위: 리피터 행 데이터
    if (rowData) {
      return rowData;
    }

    // 3순위: 서버 데이터 (pageData[refId])
    if (refId && pageData && pageData[refId]) {
      const isRepeater = node.children && node.children.length > 0;
      // 단일 컴포넌트인데 배열로 감싸져 왔을 때 [0] 추출
      if (!isRepeater && Array.isArray(pageData[refId])) {
        return pageData[refId][0] || {};
      }
      return isRepeater ? pageData[refId] : pageData;
    }

    // 4순위: pageData 전체
    return pageData || {};
  };

  return { treeData, getComponentData };
};
