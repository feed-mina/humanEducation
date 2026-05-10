package com.domain.demo_backend.domain.tutorial.controller;

import com.domain.demo_backend.domain.tutorial.dto.TutorialSaveRequest;
import com.domain.demo_backend.domain.tutorial.service.TutorialService;
import com.domain.demo_backend.global.common.dto.ApiResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/ui/tutorial")
@RequiredArgsConstructor
public class TutorialController {

    private final TutorialService tutorialService;

    @PostMapping("/save")
    public ApiResponse<String> saveTutorial(@RequestBody List<TutorialSaveRequest> requestList) {
        tutorialService.saveTutorialMetadata(requestList);
        return ApiResponse.success("성공적으로 저장되었습니다.");
    }

    @GetMapping("/history")
    public ApiResponse<List<String>> getHistoryList() {
        return ApiResponse.success(tutorialService.getHistoryList());
    }

    @GetMapping("/history/{screenId}")
    public ApiResponse<List<TutorialSaveRequest>> loadHistory(@PathVariable String screenId) {
        return ApiResponse.success(tutorialService.loadHistory(screenId));
    }
}