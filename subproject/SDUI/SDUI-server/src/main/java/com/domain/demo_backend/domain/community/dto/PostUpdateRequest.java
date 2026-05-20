package com.domain.demo_backend.domain.community.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
public class PostUpdateRequest {
    private String title;
    private String content;
    private List<String> retainedImages;
}
