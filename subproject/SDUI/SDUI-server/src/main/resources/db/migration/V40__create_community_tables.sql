-- =============================================
-- V40: 커뮤니티 테이블 생성
-- =============================================

-- 커뮤니티 게시글
CREATE TABLE IF NOT EXISTS community_post (
    post_id BIGSERIAL PRIMARY KEY,
    author_sqno BIGINT NOT NULL REFERENCES users(user_sqno),
    title VARCHAR(200) NOT NULL,
    content TEXT,
    like_count BIGINT DEFAULT 0,
    report_count BIGINT DEFAULT 0,
    del_yn VARCHAR(1) DEFAULT 'N',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 게시글 이미지
CREATE TABLE IF NOT EXISTS post_image (
    post_image_id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL REFERENCES community_post(post_id) ON DELETE CASCADE,
    storage_url VARCHAR(500) NOT NULL,
    original_name VARCHAR(255),
    stored_name VARCHAR(255),
    mime_type VARCHAR(100),
    file_size BIGINT,
    sort_order INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 좋아요
CREATE TABLE IF NOT EXISTS post_like (
    post_like_id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL REFERENCES community_post(post_id) ON DELETE CASCADE,
    user_sqno BIGINT NOT NULL REFERENCES users(user_sqno),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(post_id, user_sqno)
);

-- 신고
CREATE TABLE IF NOT EXISTS post_report (
    post_report_id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL REFERENCES community_post(post_id) ON DELETE CASCADE,
    reporter_sqno BIGINT NOT NULL REFERENCES users(user_sqno),
    reason_code VARCHAR(30),
    detail_text VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(post_id, reporter_sqno)
);

-- 팔로우
CREATE TABLE IF NOT EXISTS user_follow (
    follow_id BIGSERIAL PRIMARY KEY,
    follower_sqno BIGINT NOT NULL REFERENCES users(user_sqno),
    followee_sqno BIGINT NOT NULL REFERENCES users(user_sqno),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(follower_sqno, followee_sqno)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_community_post_author ON community_post(author_sqno);
CREATE INDEX IF NOT EXISTS idx_post_like_post ON post_like(post_id);
CREATE INDEX IF NOT EXISTS idx_post_report_post ON post_report(post_id);
CREATE INDEX IF NOT EXISTS idx_user_follow_follower ON user_follow(follower_sqno);
CREATE INDEX IF NOT EXISTS idx_user_follow_followee ON user_follow(followee_sqno);
