package com.domain.demo_backend.domain.tutorial.service;

import com.domain.demo_backend.domain.tutorial.dto.TutorialSaveRequest;
import com.domain.demo_backend.domain.ui.domain.UiMetadata;
import com.domain.demo_backend.domain.ui.repository.UiMetadataRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TutorialService {

    private final UiMetadataRepository uiMetadataRepository;

    /**
     * 튜토리얼 Playground에서 생성한 데이터를 'TUTORIAL_DEMO' 화면으로 저장합니다.
     * 기존 데이터는 히스토리(TUTORIAL_DEMO_v{timestamp})로 백업 후 덮어씁니다.
     */
    @Transactional
    public void saveTutorialMetadata(List<TutorialSaveRequest> requestList) {
        // 1. 기존 데이터 백업 (Versioning)
        // 주의: UiMetadataRepository에 List<UiMetadata> findByScreenId(String screenId);
        // 메서드가 필요합니다.
        List<UiMetadata> currentData = uiMetadataRepository.findByScreenId("TUTORIAL_DEMO");

        if (currentData != null && !currentData.isEmpty()) {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
            String historyScreenId = "TUTORIAL_DEMO_v" + timestamp;

            List<UiMetadata> historyData = currentData.stream()
                    .map(origin -> copyForHistory(origin, historyScreenId))
                    .collect(Collectors.toList());

            uiMetadataRepository.saveAll(historyData);
        }

        // 2. 기존 데모 데이터 삭제 (screen_id = 'TUTORIAL_DEMO')
        // 주의: UiMetadataRepository에 void deleteByScreenId(String screenId); 메서드가 필요합니다.
        uiMetadataRepository.deleteByScreenId("TUTORIAL_DEMO");

        // 3. DTO -> Entity 변환
        List<UiMetadata> entities = requestList.stream()
                .map(this::convertToEntity)
                .collect(Collectors.toList());

        // 4. 저장
        uiMetadataRepository.saveAll(entities);
    }

    /**
     * 저장된 튜토리얼 히스토리 목록(screen_id)을 조회합니다.
     */
    public List<String> getHistoryList() {
        // 주의: UiMetadataRepository에 List<UiMetadata> findByScreenIdStartingWith(String
        // prefix); 메서드가 필요합니다.
        List<UiMetadata> allHistory = uiMetadataRepository.findByScreenIdStartingWith("TUTORIAL_DEMO_v");
        return allHistory.stream()
                .map(UiMetadata::getScreenId)
                .distinct()
                .sorted((s1, s2) -> s2.compareTo(s1)) // 최신순 정렬
                .collect(Collectors.toList());
    }

    /**
     * 특정 히스토리 버전의 메타데이터를 불러옵니다.
     */
    public List<TutorialSaveRequest> loadHistory(String screenId) {
        List<UiMetadata> metadataList = uiMetadataRepository.findByScreenId(screenId);
        return metadataList.stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
    }

    private UiMetadata copyForHistory(UiMetadata origin, String historyScreenId) {
        UiMetadata copy = new UiMetadata();
        copy.setScreenId(historyScreenId);
        copy.setComponentId(origin.getComponentId());
        copy.setComponentType(origin.getComponentType());
        copy.setLabelText(origin.getLabelText());
        copy.setSortOrder(origin.getSortOrder());
        copy.setCssClass(origin.getCssClass());
        // 필요한 경우 추가 필드 복사 (예: group_direction)
        // copy.setGroupDirection(origin.getGroupDirection());
        return copy;
    }

    private UiMetadata convertToEntity(TutorialSaveRequest dto) {
        // UiMetadata 엔티티의 빌더나 Setter를 사용 (여기서는 Setter 가정)
        UiMetadata entity = new UiMetadata();
        entity.setScreenId("TUTORIAL_DEMO"); // 데모용 스크린 ID 고정
        entity.setComponentId(dto.getComponentId());
        entity.setComponentType(dto.getComponentType());
        entity.setLabelText(dto.getLabelText());
        entity.setSortOrder(dto.getSortOrder());
        entity.setCssClass(dto.getCssClass());
        // 필요한 경우 기본값 설정 (예: group_direction 등)
        return entity;
    }

    private TutorialSaveRequest convertToDto(UiMetadata entity) {
        TutorialSaveRequest dto = new TutorialSaveRequest();
        dto.setComponentId(entity.getComponentId());
        dto.setComponentType(entity.getComponentType());
        dto.setLabelText(entity.getLabelText());
        dto.setSortOrder(entity.getSortOrder());
        dto.setCssClass(entity.getCssClass());
        return dto;
    }
}