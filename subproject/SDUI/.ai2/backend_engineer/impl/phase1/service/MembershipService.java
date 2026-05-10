// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/membership/service/MembershipService.java
package com.domain.demo_backend.domain.membership.service;

import com.domain.demo_backend.domain.membership.domain.Membership;
import com.domain.demo_backend.domain.membership.domain.MembershipRepository;
import com.domain.demo_backend.domain.membership.dto.MembershipRequest;
import com.domain.demo_backend.domain.membership.dto.MembershipResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class MembershipService {

    private final MembershipRepository membershipRepository;

    @Transactional(readOnly = true)
    public List<MembershipResponse> findAll() {
        return membershipRepository.findAll().stream()
                .map(MembershipResponse::new)
                .collect(Collectors.toList());
    }

    @Transactional
    public MembershipResponse create(MembershipRequest req) {
        if (membershipRepository.existsByName(req.getName())) {
            throw new IllegalArgumentException("이미 존재하는 멤버십 이름입니다: " + req.getName());
        }
        Membership membership = Membership.builder()
                .name(req.getName())
                .canLearn(req.isCanLearn())
                .canConverse(req.isCanConverse())
                .canAnalyze(req.isCanAnalyze())
                .durationDays(req.getDurationDays())
                .priceCents(req.getPriceCents())
                .description(req.getDescription())
                .build();
        return new MembershipResponse(membershipRepository.save(membership));
    }

    @Transactional
    public void delete(Long id) {
        if (!membershipRepository.existsById(id)) {
            throw new IllegalArgumentException("존재하지 않는 멤버십입니다: " + id);
        }
        membershipRepository.deleteById(id);
    }
}
