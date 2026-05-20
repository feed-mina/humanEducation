package com.domain.demo_backend.domain.community.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PostImageRepository extends JpaRepository<PostImage, Long> {

    List<PostImage> findByPost_PostIdOrderBySortOrder(Long postId);

    void deleteByPost_PostId(Long postId);
}
