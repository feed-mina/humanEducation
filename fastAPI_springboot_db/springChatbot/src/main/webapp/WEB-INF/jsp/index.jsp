<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chatbot</title>
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
        .chat-container {
            width: 100%;
            max-width: 600px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .chat-header {
            background-color: #4a90e2;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 1.2em;
            /* 수정: 헤더에 대화 기록 링크 추가를 위한 공간 확보 */
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-body {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: #fafafa;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #4a90e2;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: #e9ecef;
            color: #333;
            margin-right: auto;
        }
        .chat-footer {
            padding: 10px;
            border-top: 1px solid #ddd;
            background-color: #fff;
        }
        .chat-form {
            display: flex;
            gap: 10px;
        }
        textarea {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            resize: none;
            font-size: 1em;
            outline: none;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #4a90e2;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        input[type="submit"]:hover {
            background-color: #357abd;
        }
        /* 수정: 헤더 링크 스타일 개선 */
        .header-link a {
            color: white;
            text-decoration: none;
            font-size: 1em;
            padding: 5px 10px;
            border-radius: 5px;
            background-color: #357abd;
        }
        .header-link a:hover {
            background-color: #2a6395;
        }
        /* 기존: 푸터 링크 스타일 유지 */
        .history-link {
            text-align: center;
            margin-top: 10px;
        }
        .history-link a {
            color: #4a90e2;
            text-decoration: none;
            font-size: 1em;
        }
        .history-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>Spring Chatbot</h1>
            <!-- 수정: 헤더에 대화 기록 페이지로 이동하는 링크 추가 -->
            <div class="header-link">
                <a href="/history">View History</a>
            </div>
        </div>
        <div class="chat-body">
            <% if (request.getAttribute("message") != null) { %>
                <div class="message user-message">
                    ${message}
                </div>
            <% } %>
            <% if (request.getAttribute("response") != null) { %>
                <div class="message bot-message">
                    ${response}
                </div>
            <% } %>
        </div>
        <div class="chat-footer">
            <form method="post" action="/chat" class="chat-form">
                <textarea id="message" name="message" rows="2" placeholder="Type your message...">${message}</textarea>
                <input type="submit" value="Send">
            </form>
            <!-- 기존: 푸터에 대화 기록 링크 유지 -->
            <div class="history-link">
                <a href="/history">View Chat History</a>
            </div>
        </div>
    </div>
</body>
</html>