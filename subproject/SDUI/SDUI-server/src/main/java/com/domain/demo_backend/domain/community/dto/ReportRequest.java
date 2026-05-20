package com.domain.demo_backend.domain.community.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class ReportRequest {
    private String reasonCode;
    private String detailText;
}
