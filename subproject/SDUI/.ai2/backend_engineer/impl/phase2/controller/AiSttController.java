// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/controller/AiSttController.java
package com.domain.demo_backend.domain.ai.controller;

import com.domain.demo_backend.domain.ai.dto.SttResponse;
import com.domain.demo_backend.domain.ai.service.SttService;
import com.domain.demo_backend.global.common.response.ApiResponse;
import com.domain.demo_backend.global.security.CustomUserDetails;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/api/ai")
@RequiredArgsConstructor
public class AiSttController {

    private final SttService sttService;

    /**
     * POST /api/ai/stt
     * multipart/form-data: audio(file), language(string, default="en")
     */
    @PostMapping("/stt")
    public ResponseEntity<ApiResponse<SttResponse>> transcribe(
            @RequestParam("audio") MultipartFile audio,
            @RequestParam(value = "language", defaultValue = "en") String language,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        log.info("STT 요청 - userId={}, language={}, size={}bytes",
                userDetails.getUserSqno(), language, audio.getSize());

        SttResponse result = sttService.transcribe(audio, language);
        return ResponseEntity.ok(ApiResponse.success(result));
    }
}
