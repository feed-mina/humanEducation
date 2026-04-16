<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chat History</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .history-container {
            width: 100%;
            max-width: 800px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .history-header {
            background-color: #4a90e2;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 1.2em;
        }
        .history-body {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: #fafafa;
        }
        .history-entry {
            margin-bottom: 20px;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .user-message {
            font-weight: bold;
            color: #4a90e2;
        }
        .bot-message {
            color: #333;
        }
        .timestamp {
            font-size: 0.9em;
            color: #777;
        }
        .back-link {
            text-align: center;
            padding: 10px;
        }
        .back-link a {
            color: #4a90e2;
            text-decoration: none;
            font-size: 1em;
        }
        .back-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="history-container">
        <div class="history-header">
            <h1>Chat History</h1>
        </div>
        <div class="history-body">
            <c:forEach var="history" items="${historyList}">
                <div class="history-entry">
                    <div class="user-message">User: ${history.userMessage}</div>
                    <div class="bot-message">Bot: ${history.botResponse}</div>
                    <div class="timestamp">
                        <!-- 수정: 타임존과 포맷 명확히 지정 -->
                        <fmt:formatDate value="${history.createdAt}" pattern="yyyy-MM-dd HH:mm:ss" timeZone="Asia/Seoul"/>
                    </div>
                </div>
            </c:forEach>
            <c:if test="${empty historyList}">
                <div>No chat history available.</div>
            </c:if>
        </div>
        <div class="back-link">
            <a href="/">Back to Chat</a>
        </div>
    </div>
</body>
</html>