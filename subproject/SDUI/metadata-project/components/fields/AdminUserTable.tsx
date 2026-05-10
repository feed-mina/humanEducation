'use client';

import React, { memo } from 'react';
import Pagination from '@/components/fields/Pagination';
import { useAdminUsers } from '@/components/DynamicEngine/hook/useAdminUsers';

// SDUI DynamicEngine이 전달하는 표준 props (meta, data, onChange, onAction)를 받지만,
// 이 컴포넌트는 자체 hook으로 모든 상태와 API 호출을 처리합니다.
const AdminUserTable: React.FC<any> = memo(() => {
    const {
        users, totalCount, isLoading, pageSize, currentPage,
        keyword, setKeyword,
        roleFilter, applyRoleFilter,
        selectedIds, newRole, setNewRole,
        handleSearch, handlePageChange, toggleSelect, handleRoleChange,
    } = useAdminUsers();

    return (
        <div className="admin-user-table-wrapper">

            {/* ── 툴바: 검색 + 권한 변경 ── */}
            <div className="admin-toolbar">
                <div className="admin-search-group">
                    <input
                        type="text"
                        className="admin-search-input inputfield-core"
                        placeholder="아이디 또는 이메일로 검색..."
                        value={keyword}
                        onChange={e => setKeyword(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleSearch()}
                    />
                    <select
                        className="admin-role-filter"
                        value={roleFilter}
                        onChange={e => applyRoleFilter(e.target.value)}
                    >
                        <option value="">전체 권한</option>
                        <option value="ROLE_GUEST">게스트</option>
                        <option value="ROLE_USER">일반사용자</option>
                        <option value="ROLE_ADMIN">관리자</option>
                    </select>
                    <button className="admin-search-btn content-btn" onClick={handleSearch}>
                        검색
                    </button>
                </div>

                <div className="admin-role-control">
                    <select
                        className="admin-role-select"
                        value={newRole}
                        onChange={e => setNewRole(e.target.value)}
                    >
                        <option value="ROLE_GUEST">게스트 (GUEST)</option>
                        <option value="ROLE_USER">일반사용자 (USER)</option>
                        <option value="ROLE_ADMIN">관리자 (ADMIN)</option>
                    </select>
                    <button className="admin-change-btn content-btn" onClick={handleRoleChange}>
                        선택된 회원 권한 변경
                    </button>
                </div>
            </div>

            {/* ── 선택 인원 표시 ── */}
            {selectedIds.size > 0 && (
                <div className="admin-selection-info">
                    {selectedIds.size}명 선택됨 (최대 5명)
                </div>
            )}

            {/* ── 회원 테이블 ── */}
            <table className="admin-user-table">
                <thead>
                    <tr>
                        <th style={{ width: '5%' }}></th>
                        <th style={{ width: '25%' }}>아이디</th>
                        <th style={{ width: '40%' }}>이메일</th>
                        <th style={{ width: '30%' }}>현재 권한</th>
                    </tr>
                </thead>
                <tbody>
                    {isLoading ? (
                        <tr>
                            <td colSpan={4} className="admin-table-empty">불러오는 중...</td>
                        </tr>
                    ) : users.length === 0 ? (
                        <tr>
                            <td colSpan={4} className="admin-table-empty">검색 결과가 없습니다.</td>
                        </tr>
                    ) : (
                        users.map(user => (
                            <tr
                                key={user.userSqno}
                                className={selectedIds.has(user.userSqno) ? 'admin-row-selected' : ''}
                                onClick={() => toggleSelect(user.userSqno)}
                            >
                                <td style={{ textAlign: 'center' }} onClick={e => e.stopPropagation()}>
                                    <input
                                        type="checkbox"
                                        checked={selectedIds.has(user.userSqno)}
                                        onChange={() => toggleSelect(user.userSqno)}
                                    />
                                </td>
                                <td>{user.userId}</td>
                                <td>{user.email}</td>
                                <td>
                                    <span className={`admin-role-badge ${user.role === 'ROLE_ADMIN' ? 'badge-admin' : user.role === 'ROLE_GUEST' ? 'badge-guest' : 'badge-user'}`}>
                                        {user.role === 'ROLE_ADMIN' ? '관리자' : user.role === 'ROLE_GUEST' ? '게스트' : '일반사용자'}
                                    </span>
                                </td>
                            </tr>
                        ))
                    )}
                </tbody>
            </table>

            {/* ── 페이징 (기존 Pagination 컴포넌트 재사용) ── */}
            <Pagination
                totalCount={totalCount}
                pageSize={pageSize}
                currentPage={currentPage}
                onPageChange={handlePageChange}
            />
        </div>
    );
});

AdminUserTable.displayName = 'AdminUserTable';
export default AdminUserTable;
