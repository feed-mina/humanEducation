'use client' // 여기는 손님이 있는 '식탁' 구역이라고 알려주는 거예요

import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import {useState} from 'react';

export default function ReactQueryProvider({children}: { children: React.ReactNode }) {
    const [queryClient] = useState(() => new QueryClient());

    return (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );
}