package com.domain.demo_backend.domain.kridechat.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

@Slf4j
@Component
public class FastApiChatClient {

    private final WebClient webClient;

    public FastApiChatClient(
            @Value("${kride.fastapi.url:http://localhost:8000}") String fastApiUrl) {
        this.webClient = WebClient.builder()
                .baseUrl(fastApiUrl)
                .build();
    }

    public Mono<Map<String, Object>> recommendAi(List<String> artists, List<String> regions, List<String> purposes) {
        Map<String, Object> body = Map.of(
                "artists", artists != null ? artists : List.of(),
                "regions", regions != null ? regions : List.of(),
                "purposes", purposes != null ? purposes : List.of()
        );

        return webClient.post()
                .uri("/api/recommend/ai")
                .bodyValue(body)
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {});
    }

    public Mono<Map<String, Object>> generateItinerary(
            List<String> artists, List<String> regions, List<String> purposes, int duration) {
        Map<String, Object> body = Map.of(
                "artists", artists != null ? artists : List.of(),
                "regions", regions != null ? regions : List.of(),
                "purposes", purposes != null ? purposes : List.of(),
                "duration", duration
        );

        return webClient.post()
                .uri("/api/recommend/itinerary")
                .bodyValue(body)
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {});
    }

    public Flux<String> streamChat(String message) {
        return webClient.post()
                .uri("/api/chat/stream")
                .bodyValue(Map.of("message", message))
                .retrieve()
                .bodyToFlux(String.class);
    }
}
