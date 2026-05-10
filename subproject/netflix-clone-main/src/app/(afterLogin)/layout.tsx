import React, { ReactNode } from "react";
import ConditionalHeader from "@/app/(afterLogin)/_component/conditional-header";
import ReactQueryProvider from "@/app/(afterLogin)/_component/react-query-provider";

export default function Layout({
  children,
  modal,
}: {
  children: ReactNode;
  modal: ReactNode;
}) {
  return (
    <ReactQueryProvider>
      <main>
        <ConditionalHeader />
        {children}
        {modal}
      </main>
    </ReactQueryProvider>
  );
}
