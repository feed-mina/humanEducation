-- DESTINATION: SDUI-server/src/main/resources/db/migration/V23__create_user_memberships.sql

CREATE TABLE user_memberships (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    membership_id BIGINT NOT NULL REFERENCES memberships(id),
    started_at    TIMESTAMP NOT NULL,
    expires_at    TIMESTAMP NOT NULL,
    status        VARCHAR(20) NOT NULL DEFAULT 'active',
    granted_by    VARCHAR(20) NOT NULL DEFAULT 'purchase',
    created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_memberships_user_id     ON user_memberships(user_id);
CREATE INDEX idx_user_memberships_user_status ON user_memberships(user_id, status);
CREATE INDEX idx_user_memberships_expires_at  ON user_memberships(expires_at);
