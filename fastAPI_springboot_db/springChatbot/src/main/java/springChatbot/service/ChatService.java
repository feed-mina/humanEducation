package springChatbot.service;

import springChatbot.dto.ChatRequest;
import springChatbot.dto.ChatHistory;
import springChatbot.mapper.ChatHistoryMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Service
public class ChatService {

    @Value("${fastapi.url:http://localhost:8000}")
    private String fastApiUrl;

    private final RestTemplate restTemplate;
    private final ChatHistoryMapper chatHistoryMapper;

    @Autowired
    public ChatService(ChatHistoryMapper chatHistoryMapper) {
        this.restTemplate = new RestTemplate();
        this.chatHistoryMapper = chatHistoryMapper;
    }

    public String getChatResponse(ChatRequest chatRequest) {
        String url = fastApiUrl + "/chat/message";
        return restTemplate.postForObject(url, chatRequest, ChatResponse.class).getResponse();
    }

    public void saveChatHistory(String userMessage, String botResponse) {
        ChatHistory chatHistory = new ChatHistory();
        chatHistory.setUserMessage(userMessage);
        chatHistory.setBotResponse(botResponse);
        chatHistory.setModelName("gemma2"); 
        chatHistoryMapper.insertChatHistory(chatHistory);
    }

    public List<ChatHistory> getChatHistory() {
        return chatHistoryMapper.selectChatHistory();
    }
}

class ChatResponse {
    private String response;

    public String getResponse() {
        return response;
    }

    public void setResponse(String response) {
        this.response = response;
    }
}