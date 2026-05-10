'use client';
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import RecordTimeComponent from "@/components/fields/RecordTimeComponent";
import {useDeviceType} from "@/hooks/useDeviceType";

export default function AppShell({ children }: { children: React.ReactNode }) {
    // 1. 훅에서 모바일 여부와 클래스명을 동시에 가져온다.
    const { isMobile, deviceClass } = useDeviceType();

    // 2. 가독성을 위해 PC 여부를 변수로 선언해둔다.
    const isPc = !isMobile;

    return (
        // 3. 훅에서 이미 정의된 deviceClass('is-pc' 또는 'is-mobile')를 그대로 사용한다.
        <div className={`app-wrapper ${deviceClass}`}>

            {/* 4. PC일 때는 Sidebar, 모바일일 때는 Header를 보여준다. */}
            {isPc ? <Sidebar /> : <Header />}

            <main className="main-contents-area">
                {/* 5. PC 전용 유틸리티 영역 처리 */}
                {isPc && (
                    <div className="pc-top-utility">
                        <RecordTimeComponent />
                    </div>
                )}

                <section className="page-view-container">
                    {children}
                </section>
            </main>
        </div>
    );
}