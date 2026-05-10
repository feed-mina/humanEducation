import React from 'react';

interface SkeletonProps {
    width?: string | number;
    height?: string | number;
    className?: string;
}

const Skeleton = ({ width = "100%", height = "20px", className = "" }: SkeletonProps) => {
    const style = {
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
    };

    // 'skeleton-box' 클래스가 Shimmer 애니메이션을 담당함
    return <div className={`skeleton-box ${className}`} style={style} />;
};

export default Skeleton;