// 주의: Spring Boot 3.1.4 (Spring 6.0.x) → RestClient 없음
//   STT: RestTemplate (Spring 내장, multipart 지원)
//   Streaming: java.net.http.HttpClient (Java 17 내장, 추가 의존성 없음)
package com.domain.demo_backend.domain.ai.client;

import com.domain.demo_backend.domain.kakao.service.OperationAlertService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
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

    @Value("${openai.cost.threshold:5.0}")
    private double costThreshold;

    @Autowired
    private OperationAlertService operationAlertService;

    // 일일 누적 비용 추적 (단위: 마이크로달러, $0.000001)
    private final AtomicLong dailyMicroDollars = new AtomicLong(0);

    private final WebClient webClient = WebClient.create();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Scheduled(cron = "0 0 0 * * *", zone = "Asia/Seoul")
    public void resetDailyCost() {
        dailyMicroDollars.set(0);
        log.debug("OpenAI 일일 비용 카운터 초기화");
    }

    /**
     * 추정 비용을 누적하고 임계 초과 시 Slack 알림을 발송한다.
     * GPT-4o 기준: 입력 $2.5/1M tokens, 출력 $10/1M tokens 으로 추정.
     * (4자 ≈ 1 token)
     */
    private void trackCost(int inputChars, int outputChars) {
        long inputMicro  = (long)((inputChars  / 4.0) * 2.5);   // $2.5/1M tokens → μ$
        long outputMicro = (long)((outputChars / 4.0) * 10.0);  // $10/1M tokens  → μ$
        long total = dailyMicroDollars.addAndGet(inputMicro + outputMicro);
        double totalDollars = total / 1_000_000.0;
        if (totalDollars >= costThreshold) {
            operationAlertService.sendCostAlert(totalDollars, costThreshold);
            dailyMicroDollars.set(0);  // 알림 후 초기화 (반복 알림 방지)
        }
    }

    /**
     * STT: Whisper API (multipart/form-data)
     */
    @SuppressWarnings("unchecked")
    public String transcribe(MultipartFile audio, String language) throws IOException {
        byte[] audioBytes = audio.getBytes();
        String originalFilename = audio.getOriginalFilename() != null
                ? audio.getOriginalFilename() : "audio.webm";

        ByteArrayResource audioResource = new ByteArrayResource(audioBytes) {
            @Override
            public String getFilename() { return originalFilename; }
        };

        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", audioResource).filename(originalFilename);
        builder.part("model", whisperModel);
        if (language != null && !language.isBlank()) {
            builder.part("language", language);
        }

        Map<String, Object> response = webClient.post()
                .uri(OPENAI_BASE_URL + "/audio/transcriptions")
                .header("Authorization", "Bearer " + apiKey)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(builder.build()))
                .retrieve()
                .bodyToMono(Map.class)
                .block();

        if (response == null || response.get("text") == null) {
            throw new IllegalStateException("Whisper API 응답이 비어 있습니다.");
        }
        return response.get("text").toString();
    }

    /**
     * Chat Completions SSE 스트리밍 (비전 지원 — content가 String 또는 List인 경우)
     * 이미지 면접 등 멀티모달 메시지에 사용
     */
    public void streamChatObjects(
            List<Map<String, Object>> messages,
            Consumer<String> onChunk,
            Runnable onComplete) throws Exception {

        int inputChars = messages.stream()
                .mapToInt(m -> {
                    Object c = m.get("content");
                    return c instanceof String s ? s.length() : c != null ? c.toString().length() : 0;
                }).sum();
        AtomicLong outputChars = new AtomicLong(0);

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
                            outputChars.addAndGet(chunk.length());
                            onChunk.accept(chunk);
                        }
                    } catch (Exception e) {
                        log.warn("SSE 청크 파싱 실패 (무시): {}", json);
                    }
                }
            }
        }
        onComplete.run();
        trackCost(inputChars, (int) outputChars.get());
    }

    /**
     * Chat Completions SSE 스트리밍
     * java.net.http.HttpClient (Java 17 내장) 사용 — InputStream 라인 단위 파싱
     */
    public void streamChat(
            List<Map<String, String>> messages,
            Consumer<String> onChunk,
            Runnable onComplete) throws Exception {

        int inputChars = messages.stream()
                .mapToInt(m -> m.getOrDefault("content", "").length())
                .sum();
        AtomicLong outputChars = new AtomicLong(0);

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
                            outputChars.addAndGet(chunk.length());
                            onChunk.accept(chunk);
                        }
                    } catch (Exception e) {
                        log.warn("SSE 청크 파싱 실패 (무시): {}", json);
                    }
                }
            }
        }
        onComplete.run();
        trackCost(inputChars, (int) outputChars.get());
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

    @SuppressWarnings("unchecked")
    public String chat(List<Map<String, String>> messages) throws Exception {
        int inputChars = messages.stream()
                .mapToInt(m -> m.getOrDefault("content", "").length())
                .sum();

        String jsonBody = objectMapper.writeValueAsString(Map.of(
                "model", model,
                "messages", messages,
                "stream", false
        ));

        Map response = webClient.post()
                .uri(OPENAI_BASE_URL + "/chat/completions")
                .header("Authorization", "Bearer " + apiKey)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(jsonBody)
                .retrieve()
                .bodyToMono(Map.class)
                .block();

        if (response == null || !response.containsKey("choices")) {
            throw new IllegalStateException("OpenAI API 응답이 비어 있습니다.");
        }

        List<Map<String, Object>> choices = (List<Map<String, Object>>) response.get("choices");
        Map<String, Object> choice = choices.get(0);
        Map<String, Object> assistantMessage = (Map<String, Object>) choice.get("message");
        String reply = (String) assistantMessage.get("content");

        trackCost(inputChars, reply != null ? reply.length() : 0);
        return reply;
    }
}
