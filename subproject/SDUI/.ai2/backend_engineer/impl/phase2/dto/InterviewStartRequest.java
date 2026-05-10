// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/dto/InterviewStartRequest.java
package com.domain.demo_backend.domain.ai.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class InterviewStartRequest {
    private String resumeText;
    private String language;  // "en" | "ko"
}
