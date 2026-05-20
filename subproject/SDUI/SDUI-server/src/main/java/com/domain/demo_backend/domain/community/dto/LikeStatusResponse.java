package com.domain.demo_backend.domain.community.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class LikeStatusResponse {
    private boolean liked;
    private long likeCount;
}
