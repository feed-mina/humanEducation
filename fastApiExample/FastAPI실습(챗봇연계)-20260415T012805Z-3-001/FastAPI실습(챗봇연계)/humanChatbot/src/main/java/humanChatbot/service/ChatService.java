package humanChatbot.service;

import humanChatbot.dto.ChatRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class ChatService {

    @Value("${fastapi.url:http://localhost:8000}")
    private String fastApiUrl;

    private final RestTemplate restTemplate;

    public ChatService() {
        this.restTemplate = new RestTemplate();
    }

    public String getChatResponse(ChatRequest chatRequest) {
        String url = fastApiUrl + "/chat/message";
        return restTemplate.postForObject(url, chatRequest, ChatResponse.class).getResponse();
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