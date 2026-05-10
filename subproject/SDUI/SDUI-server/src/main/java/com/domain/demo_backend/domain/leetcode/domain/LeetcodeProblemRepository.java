package com.domain.demo_backend.domain.leetcode.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface LeetcodeProblemRepository extends JpaRepository<LeetcodeProblem, Long> {

    Optional<LeetcodeProblem> findFirstBySentDateIsNullOrderByDisplayOrderAsc();
}
