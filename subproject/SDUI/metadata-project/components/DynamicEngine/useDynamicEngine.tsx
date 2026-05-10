// 2026-02-07 추가 데이터 가공 및 트리 생성 로직
import {Metadata} from "@/components/DynamicEngine/type"

// @@@@ 2026-02-07 주석 추가
// useDynamicEngine 역할 : 메타데이터를 트리(부모-자식) 구조로 바꾸고 데이터를 매핑함, 데이터바인딩 (pageData에서 필드로 연결)
export const useDynamicEngine = (metadata: Metadata[], pageData: any, formData: any) => {

    const treeData = metadata;

    // @@@@ 데이터 추출 로직 개선: formData(입력값)가 있으면 최우선으로 반환
    const getComponentData = (node: Metadata, rowData: any) => {
        // refDataId는 서버와 공용하는 id
        const refId = node.refDataId || node.ref_data_id;

        // * 화면동작1 (ex 쓰기 페이지) : 사용자가 입력중인 데이터 (formData) 사용자가 키보드로 값을 치고 있다면 서버에서 가져온 옛날 데이터는 일단 무시하고 방금 입력한 값을 보여준다 , formData[refId]가 존재하면 바로 반환
        if (refId && formData && formData[refId] !== undefined) {
            return formData[refId];
        }
        // 데이터의 흐름을 단방향으로 유지하면서도 유연성도 갖음
        //* 화면동작2 (ex 목록페이지): 리피터 안,게시글 목록처럼 같은 모양일때 각 줄(row)마다 다른 내용이 들어가야한다.  리피터안에서 개별 항목을 그릴때 해당 행의 객체를 그대로 반환한다
        if (rowData) {
            return rowData;
        }
        // * 화면동작3 (ex 상세페이지): 서버에서 받아온 메타 데이터를 찾아온다. pageData 전체를 넘김

        if (refId && pageData && pageData[refId]) {
            // * 예외처리1: 배열 정규화 (서버에서 데이터 하나만 줬는데 굳이 배열로 감싸서 보냈을때 첫번째 [0]를 꺼낸다.)
            // * 리스트가 아닌 '단일 컴포넌트(TimeSlot 등)'인데 배열로 감싸져 왔다면 0번을 꺼내준다
            const isRepeater = node.children && node.children.length > 0;
            if (!isRepeater && Array.isArray(pageData[refId])) {
                return pageData[refId][0] || {};
            }
            // * 예외처리2: 리피터 판단 (자식 노드가 있으면 리스트로 볻고 배열 전체를 넘기고 아니면 단일 객체로 취급한다)
            return isRepeater ? pageData[refId] : pageData;
        }
        return pageData || {};
    };
    return {treeData, getComponentData};
};