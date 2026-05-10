import { Metadata } from "./type";

export const useDynamicEngine = (metadata: Metadata[], pageData: any, formData: any) => {
  const treeData = metadata;

  const getComponentData = (node: Metadata, rowData: any) => {
    const refId = node.refDataId || node.ref_data_id;

    // 1순위: 사용자가 입력 중인 formData
    if (refId && formData && formData[refId] !== undefined) {
      return formData[refId];
    }

    // 2순위: Repeater 행 데이터
    if (rowData) {
      return rowData;
    }

    // 3순위: 서버에서 받은 pageData
    if (refId && pageData && pageData[refId]) {
      const isRepeater = node.children && node.children.length > 0;
      if (!isRepeater && Array.isArray(pageData[refId])) {
        return pageData[refId][0] || {};
      }
      return isRepeater ? pageData[refId] : pageData;
    }

    return pageData || {};
  };

  return { treeData, getComponentData };
};
