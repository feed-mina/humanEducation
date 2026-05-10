import { useRenderCount } from "@/components/DynamicEngine/hook/useRenderCount";
import React from 'react';
import {useMetadata} from "@/components/providers/MetadataProvider";

export function withRenderTrack<P extends object>(
    Component: React.ComponentType<P>,
    componentName: string
) {
    // 1. 익명 함수 대신 이름을 가진 함수로 정의하여 디버깅 효율을 높입니다.
    const WrappedComponent = (props: P) => {

        // 1. 컨텍스트에서 현재 화면 ID를 가져온다
        const { screenId } = useMetadata();
        useRenderCount(`Field: ${componentName} (Screen: ${screenId})`);
        return <Component {...props} />;
    };
// 2. 핵심: 리액트 개발자 도구(DevTools)에서 컴포넌트를 식별할 수 있게 이름을 부여합니다.
    WrappedComponent.displayName = `WithRenderTrack(${componentName})`;
    return WrappedComponent;
}