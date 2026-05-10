// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/dto/ChatMessage.java
package com.domain.demo_backend.domain.ai.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ChatMessage {
    private String role;     // "system" | "user" | "assistant"
    private String content;
}
