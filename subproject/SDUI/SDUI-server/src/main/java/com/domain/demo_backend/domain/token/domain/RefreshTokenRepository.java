package com.domain.demo_backend.domain.token.domain;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface RefreshTokenRepository extends CrudRepository<RefreshToken, Long> {
    // userSqno가 @Id이므로 기본 제공되는 findById를 사용하면 돼.
    // 만약 이메일로도 찾고 싶다면 아래처럼 추가할 수 있지만, Redis는 Key 기반 조회가 가장 빨라.
    Optional<RefreshToken> findByEmail(String email);
}
