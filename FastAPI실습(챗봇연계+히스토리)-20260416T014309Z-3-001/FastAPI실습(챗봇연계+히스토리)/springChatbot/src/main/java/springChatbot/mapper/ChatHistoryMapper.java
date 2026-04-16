package springChatbot.mapper;

import springChatbot.dto.ChatHistory;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface ChatHistoryMapper {
    void insertChatHistory(ChatHistory chatHistory);
    List<ChatHistory> selectChatHistory();
}