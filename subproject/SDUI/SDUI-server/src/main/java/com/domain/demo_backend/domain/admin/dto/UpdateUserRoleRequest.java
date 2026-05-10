package com.domain.demo_backend.domain.admin.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class UpdateUserRoleRequest {
    private List<Long> userIds;
    private String newRole;
}
