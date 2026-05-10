'use client';

import React, {useMemo} from "react";

interface PaginationProps {
    totalCount: number;
    pageSize: number;
    currentPage: number;
    onPageChange: (page: number) => void;
}

const Pagination: React.FC<PaginationProps> = ({totalCount, pageSize, currentPage, onPageChange}) => {
    // 전체페이지 수 계산
    const totalPages = Math.ceil(totalCount / pageSize);
    // @@@@ 2026-02-07 추가
    // 보여줄 페이지 번호 계산 (현재 페이지 기준 앞 뒤 2개씩만)
    const pageNumbers = useMemo(() => {
        if (totalPages <= 1) return [];

        const windowSize = 5; // 한 번에 보여줄 페이지 버큰 개수
        let start = Math.max(1, currentPage - Math.floor(windowSize / 2));
        const end = Math.min(totalPages, start + windowSize - 1);
        if (end - start < windowSize) {
            start = Math.max(1, end - windowSize + 1);
        }

        return Array.from({length: end - start + 1}, (_, i) => start + i);
    }, [currentPage, totalPages]);

    // 훅 호출이 모두 끝난 뒤에 리턴해야 에러가 안 난다
    if (totalPages <= 1) return null;
    return (
        /* 인라인 스타일 삭제, 클래스명 부여 */
        <div className="pagination-container">
            <button disabled={currentPage === 1} onClick={() => onPageChange(currentPage - 1)}>
                이전
            </button>
            {pageNumbers.map(page => (
                <button key={page} onClick={() => onPageChange(page)}
                        className={`page-num-btn ${currentPage === page ? 'active' : ''}`}>
                    {page}
                </button>
            ))}
            <button disabled={currentPage === totalPages} onClick={() => onPageChange(currentPage + 1)}>
                다음
            </button>
        </div>
    );
};

export default Pagination;