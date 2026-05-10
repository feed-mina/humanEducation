// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/client/OpenAiClient.java
// 주의: Spring Boot 3.1.4 (Spring 6.0.x) → RestClient 없음
//   STT: RestTemplate (Spring 내장, multipart 지원)
//   Streaming: java.net.http.HttpClient (Java 17 내장, 추가 의존성 없음)
package com.domain.demo_backend.domain.ai.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

@Slf4j
@Component
public class OpenAiClient {

    private static final String OPENAI_BASE_URL = "https://api.openai.com/v1";

    @Value("${openai.api-key}")
    private String apiKey;

    @Value("${openai.model:gpt-4o}")
    private String model;

    @Value("${openai.whisper-model:whisper-1}")
    private String whisperModel;

    private final RestTemplate restTemplate = new RestTemplate();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * STT: Whisper API (multipart/form-data)
     * Spring RestTemplate의 기본 multipart 지원 활용
     */
    public String transcribe(MultipartFile audio, String language) throws IOException {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.set("Authorization", "Bearer " + apiKey);

        byte[] audioBytes = audio.getBytes();
        String originalFilename = audio.getOriginalFilename() != null
                ? audio.getOriginalFilename() : "audio.webm";

        ByteArrayResource audioResource = new ByteArrayResource(audioBytes) {
            @Override
            public String getFilename() { return originalFilename; }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", audioResource);
        body.add("model", whisperModel);
        body.add("language", language != null ? language : "en");

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(
                OPENAI_BASE_URL + "/audio/transcriptions", request, Map.class
        );

        if (response.getBody() == null || response.getBody().get("text") == null) {
            throw new IllegalStateException("Whisper API 응답이 비어 있습니다.");
        }
        return response.getBody().get("text").toString();
    }

    /**
     * Chat Completions SSE 스트리밍
     * java.net.http.HttpClient (Java 17 내장) 사용 — InputStream 라인 단위 파싱
     */
    public void streamChat(
            List<Map<String, String>> messages,
            Consumer<String> onChunk,
            Runnable onComplete) throws Exception {

        String jsonBody = objectMapper.writeValueAsString(Map.of(
                "model", model,
                "messages", messages,
                "stream", true
        ));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OPENAI_BASE_URL + "/chat/completions"))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        HttpResponse<InputStream> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofInputStream()
        );

        if (response.statusCode() != 200) {
            String errorBody = new String(response.body().readAllBytes());
            throw new IllegalStateException("OpenAI API 오류: HTTP " + response.statusCode() + " - " + errorBody);
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("data: ") && !line.equals("data: [DONE]")) {
                    String json = line.substring(6).trim();
                    try {
                        String chunk = extractChunkContent(json);
                        if (chunk != null && !chunk.isEmpty()) {
                            onChunk.accept(chunk);
                        }
                    } catch (Exception e) {
                        log.warn("SSE 청크 파싱 실패 (무시): {}", json);
                    }
                }
            }
        }
        onComplete.run();
    }

    /**
     * OpenAI SSE delta 청크에서 텍스트 추출
     * 형식: {"id":"...","choices":[{"delta":{"content":"Hello"},...}]}
     */
    @SuppressWarnings("unchecked")
    private String extractChunkContent(String json) throws Exception {
        Map<String, Object> data = objectMapper.readValue(json, Map.class);
        List<Map<String, Object>> choices = (List<Map<String, Object>>) data.get("choices");
        if (choices == null || choices.isEmpty()) return null;

        Map<String, Object> delta = (Map<String, Object>) choices.get(0).get("delta");
        if (delta == null) return null;

        Object content = delta.get("content");
        return content != null ? content.toString() : null;
    }
}
