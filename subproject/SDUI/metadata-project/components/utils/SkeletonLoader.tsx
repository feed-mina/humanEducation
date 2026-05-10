// components/SkeletonLoader.tsx
import Skeleton from './Skeleton';
import '../../app/styles/SkeletonLoader.css';

const SkeletonLoader = () => {
    return (
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {/* 타이틀 영역 */}
            <Skeleton width="40%" height="32px" />

            {/* 입력 폼 영역 흉내 */}
            <div style={{ display: 'flex', gap: '10px' }}>
                <Skeleton width="20%" height="40px" />
                <Skeleton width="75%" height="40px" />
            </div>

            {/* 본문 영역 흉내 */}
            <Skeleton height="150px" />

            {/* 버튼 영역 */}
            <Skeleton width="100px" height="40px" className="align-self-end" />
        </div>
    );
};

export default SkeletonLoader;