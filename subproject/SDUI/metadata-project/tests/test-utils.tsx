import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from '@/context/AuthContext';

const createTestQueryClient = () => new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
});

const renderWithProviders = (
    ui: ReactElement,
    options?: Omit<RenderOptions, 'wrapper'>
) => {
    const testQueryClient = createTestQueryClient();
    function Wrapper({ children }: { children: React.ReactNode }) {
        return (
            <QueryClientProvider client={testQueryClient}>
                <AuthProvider>
                    {children}
                </AuthProvider>
            </QueryClientProvider>
        );
    }
    return render(ui, { wrapper: Wrapper, ...options });
};

export * from '@testing-library/react';
export { renderWithProviders };

// 렌더 카운트 헬퍼 함수
export const getRenderCount = (componentName: string): number => {
    if (typeof window !== 'undefined' && (window as any).__componentRenderCounts__) {
        return (window as any).__componentRenderCounts__[componentName] || 0;
    }
    return 0;
};

export const resetRenderCounts = () => {
    if (typeof window !== 'undefined') {
        (window as any).__componentRenderCounts__ = {};
    }
};