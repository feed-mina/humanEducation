package springChatbot.controller;

import springChatbot.dto.ChatRequest;
import springChatbot.dto.ChatHistory;
import springChatbot.service.ChatService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class ChatController {

    @Autowired
    private ChatService chatService;

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping("/chat")
    public String sendMessage(@RequestParam("message") String message, Model model) {
        ChatRequest chatRequest = new ChatRequest(message);
        String response = chatService.getChatResponse(chatRequest);
        // 대화 기록을 저장
        chatService.saveChatHistory(message, response);
        model.addAttribute("message", message);
        model.addAttribute("response", response);
        return "index";
    }

    // 대화 기록을 조회하여 history.jsp로 전달
    @GetMapping("/history")
    public String showHistory(Model model) {
        model.addAttribute("historyList", chatService.getChatHistory());
        return "history";
    }
}