/**
 * 트리 구조의 메타데이터를 일렬로 펴주는 함수
 * @param items - 메타데이터 배열
 * @returns 평면화된 메타데이터 배열
 */
export const flattenMetadata = (items: any[] = []): any[] => {
    let result: any[] = [];

    items.forEach(item => {
        result.push(item);
        if (item.children && item.children.length > 0) {
            result = result.concat(flattenMetadata(item.children));
        }
    });

    return result;
};