package springChatbot.dto;

import lombok.Data;

import java.sql.Timestamp;

@Data
public class ChatHistory {
    private Long id;
    private String userMessage;
    private String botResponse;
    private Timestamp createdAt;
    private String modelName; 
}