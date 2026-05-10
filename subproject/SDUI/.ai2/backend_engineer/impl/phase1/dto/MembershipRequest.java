// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/membership/dto/MembershipRequest.java
package com.domain.demo_backend.domain.membership.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class MembershipRequest {
    private String name;
    private boolean canLearn;
    private boolean canConverse;
    private boolean canAnalyze;
    private int durationDays;
    private int priceCents;
    private String description;
}
