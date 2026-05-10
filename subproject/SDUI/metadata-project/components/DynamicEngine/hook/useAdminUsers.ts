import { useState, useCallback, useEffect } from 'react';
import api from '@/services/axios';

export interface AdminUser {
    userSqno: number;
    userId: string;
    email: string;
    role: string;
}

// 검색에 실제로 사용된 파라미터 (page 변경 시 keyword/role 유지를 위해 분리)
interface QueryParams {
    keyword: string;
    roleFilter: string;
    page: number;
}

const PAGE_SIZE = 10;

export function useAdminUsers() {
    // 실제 API 호출에 사용되는 파라미터 (검색 버튼 클릭 시 커밋)
    const [query, setQuery] = useState<QueryParams>({ keyword: '', roleFilter: '', page: 1 });

    // 입력 중인 live 값 (검색 버튼 클릭 전까지 query에 반영 안 됨)
    const [keyword, setKeyword] = useState('');
    const [roleFilter, setRoleFilter] = useState('');

    // 결과 데이터
    const [users, setUsers] = useState<AdminUser[]>([]);
    const [totalCount, setTotalCount] = useState(0);
    const [isLoading, setIsLoading] = useState(false);

    // 권한 변경용 상태
    const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
    const [newRole, setNewRole] = useState('ROLE_USER');

    // API 호출
    const fetchUsers = useCallback(async (kw: string, rf: string, pg: number) => {
        setIsLoading(true);
        try {
            const params = new URLSearchParams();
            if (kw) params.set('keyword', kw);
            if (rf) params.set('role', rf);
            params.set('page', String(pg));
            params.set('size', String(PAGE_SIZE));
            const res = await api.get(`/api/admin/users?${params.toString()}`);
            setUsers(res.data.list || []);
            setTotalCount(res.data.total || 0);
        } catch {
            // 401/403 처리는 axios 인터셉터에서 담당
        } finally {
            setIsLoading(false);
        }
    }, []);

    // query 상태가 변경될 때마다 API 재조회
    useEffect(() => {
        fetchUsers(query.keyword, query.roleFilter, query.page);
    }, [query, fetchUsers]);

    // 검색 버튼: 현재 입력값을 query에 반영, 페이지는 1로 리셋
    const handleSearch = useCallback(() => {
        setSelectedIds(new Set());
        setQuery({ keyword, roleFilter, page: 1 });
    }, [keyword, roleFilter]);

    // 권한 필터 즉시 적용 (선택 즉시 검색, 검색 버튼 불필요)
    const applyRoleFilter = useCallback((role: string) => {
        setRoleFilter(role);
        setSelectedIds(new Set());
        setQuery({ keyword, roleFilter: role, page: 1 });
    }, [keyword]);

    // 페이지 변경: keyword/roleFilter는 유지
    const handlePageChange = useCallback((page: number) => {
        setSelectedIds(new Set());
        setQuery(prev => ({ ...prev, page }));
    }, []);

    // 체크박스 토글 (최대 5명)
    const toggleSelect = useCallback((id: number) => {
        setSelectedIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                if (next.size >= 5) {
                    alert('최대 5명까지만 선택할 수 있습니다.');
                    return prev;
                }
                next.add(id);
            }
            return next;
        });
    }, []);

    // 권한 변경: confirm → PUT /api/admin/users/role → 목록 새로고침
    const handleRoleChange = useCallback(async () => {
        if (selectedIds.size === 0) {
            alert('권한을 변경할 회원을 선택해주세요.');
            return;
        }
        const userIds = Array.from(selectedIds);
        const roleLabel = newRole === 'ROLE_ADMIN' ? '관리자 (ADMIN)' : newRole === 'ROLE_GUEST' ? '게스트 (GUEST)' : '일반사용자 (USER)';
        const selectedUsers = users.filter(u => selectedIds.has(u.userSqno));
        const names = selectedUsers.map(u => u.userId).join(', ');

        const confirmed = window.confirm(
            `총 ${userIds.length}명의 회원 권한을 '${roleLabel}'(으)로 변경하시겠습니까?\n\n대상: ${names}`
        );
        if (!confirmed) {
            alert('권한 변경이 취소되었습니다.');
            return;
        }

        try {
            await api.put('/api/admin/users/role', { userIds, newRole });
            alert('권한이 성공적으로 변경되었습니다.');
            setSelectedIds(new Set());
            // 현재 query 상태로 목록 새로고침
            await fetchUsers(query.keyword, query.roleFilter, query.page);
        } catch {
            // axios 인터셉터에서 에러 처리
        }
    }, [selectedIds, newRole, users, query, fetchUsers]);

    return {
        // 데이터
        users,
        totalCount,
        isLoading,
        pageSize: PAGE_SIZE,
        currentPage: query.page,
        // 입력 상태
        keyword,
        setKeyword,
        roleFilter,
        setRoleFilter,
        // 권한 변경 상태
        selectedIds,
        newRole,
        setNewRole,
        // 핸들러
        handleSearch,
        applyRoleFilter,
        handlePageChange,
        toggleSelect,
        handleRoleChange,
    };
}
