package com.domain.demo_backend.domain.study.scheduler;

import com.domain.demo_backend.domain.study.domain.StudyMaterialRepository;
import com.domain.demo_backend.domain.study.service.SlackFileService;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.ZoneId;

/**
 * 매일 19:00 KST에 미발송 정보처리기사 학습 자료 1개를 Slack으로 발송한다.
 * 전체 33개 발송 완료 후에는 로그만 출력하고 종료.
 */
@Component
@RequiredArgsConstructor
public class DailyStudyScheduler {

    private final StudyMaterialRepository materialRepo;
    private final SlackFileService slackFileService;
    private final Logger log = LoggerFactory.getLogger(DailyStudyScheduler.class);

    @Scheduled(cron = "0 0 19 * * *", zone = "Asia/Seoul")
    public void sendDailyStudyMaterial() {
        sendNextMaterial();
    }

    public void sendNextMaterial() {
        try {
            materialRepo.findFirstBySentDateIsNullOrderByDisplayOrderAsc()
                    .ifPresentOrElse(material -> {
                        slackFileService.uploadAndShare(material);
                        material.setSentDate(LocalDate.now(ZoneId.of("Asia/Seoul")));
                        materialRepo.save(material);
                        log.info("학습자료 발송 완료. id={}, name={}", material.getId(), material.getDisplayName());
                    }, () -> log.info("정보처리기사 학습자료 전체 발송 완료 (33/33)"));
        } catch (Exception e) {
            log.error("학습자료 발송 실패", e);
        }
    }
}
