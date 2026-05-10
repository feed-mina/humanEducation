import React, { ReactNode } from "react";
import ConditionalHeader from "./_component/conditional-header";
import ReactQueryProvider from "./_component/react-query-provider";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <ReactQueryProvider>
      <main>
        <ConditionalHeader />
        {children}
      </main>
    </ReactQueryProvider>
  );
}
