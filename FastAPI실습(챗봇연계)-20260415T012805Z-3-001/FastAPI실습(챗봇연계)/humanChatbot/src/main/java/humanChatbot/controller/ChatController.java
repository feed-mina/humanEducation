package humanChatbot.controller;

import humanChatbot.dto.ChatRequest;
import humanChatbot.service.ChatService;
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
        model.addAttribute("message", message);
        model.addAttribute("response", response);
        return "index";
    }
}