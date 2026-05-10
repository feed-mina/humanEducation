
//[메모]
http://localhost:3000/view/AI_ENGLISH_CHAT_PAGE 페이지에 
C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\assets\log에 

ENGLISH_대화종료.png , AI_ENGLISH_CHAT_PAGE.png, AI영어대화_정지버튼_답변완료.png



========================================
마이크 누르고 정지버튼(ㅁ) 누를때 
http://localhost:3000/api/ai/stt
http://localhost:3000/api/ai/chat/stream
페이로드
{
    "messages": [
        {
            "role": "user",
            "content": "you"
        }
    ],
    "language": "en"
}

===============================================
마이크 누르고 답변완료 누를때 
http://localhost:3000/api/ai/stt


{
    "status": "success",
    "data": {
        "text": "you"
    },
    "timestamp": "2026-03-14T00:16:49.1238956"
}

http://localhost:3000/api/ai/chat/stream
페이로드 {
    "messages": [
        {
            "role": "user",
            "content": "you"
        }
    ],
    "language": "en"
}


============================
http://localhost:3000/api/execute/AI_ENGLISH_CHAT_CONFIG
{
    "status": "success",
    "sqlKey": "AI_ENGLISH_CHAT_CONFIG",
    "data": {
        "mic_btn_label": "\uD83C\uDFA4 녹음 시작",
        "submit_btn_label": "답변완료",
        "end_btn_label": "대화 종료",
        "welcome_message": "Hello! I'm your English conversation partner. What would you like to practice today?",
        "language": "en",
        "required_tier": "PREMIUM",
        "upgrade_message": "음성 대화 기능은 프리미엄 멤버십이 필요합니다."
    }
}
==================
http://localhost:3000/api/v1/user-memberships/current
{"status":"success","timestamp":"2026-03-13T23:59:30.4689387"}

==================
http://localhost:3000/api/ui/AI_ENGLISH_CHAT_PAGE
{
    "status": "success",
    "data": [
        {
            "componentId": "ai_en_root",
            "parentGroupId": null,
            "labelText": "",
            "componentType": "GROUP",
            "sortOrder": 1,
            "isRequired": false,
            "isReadonly": true,
            "isVisible": null,
            "cssClass": "ai-page-root",
            "groupDirection": "COLUMN",
            "actionType": null,
            "actionUrl": null,
            "dataApiUrl": null,
            "dataSqlKey": null,
            "refDataId": null,
            "children": [
                {
                    "componentId": "ai_en_config",
                    "parentGroupId": "ai_en_root",
                    "labelText": "",
                    "componentType": "DATA_SOURCE",
                    "sortOrder": 2,
                    "isRequired": false,
                    "isReadonly": true,
                    "isVisible": null,
                    "cssClass": null,
                    "groupDirection": null,
                    "actionType": "AUTO_FETCH",
                    "actionUrl": null,
                    "dataApiUrl": null,
                    "dataSqlKey": "AI_ENGLISH_CHAT_CONFIG",
                    "refDataId": null
                },
                {
                    "componentId": "ai_en_chat",
                    "parentGroupId": "ai_en_root",
                    "labelText": "AI 영어 대화",
                    "componentType": "AI_CHAT",
                    "sortOrder": 3,
                    "isRequired": false,
                    "isReadonly": false,
                    "isVisible": null,
                    "cssClass": "ai-chat-en",
                    "groupDirection": null,
                    "actionType": "AI_CHAT_EN",
                    "actionUrl": null,
                    "dataApiUrl": null,
                    "dataSqlKey": null,
                    "refDataId": "ai_en_config"
                }
            ]
        }
    ],
    "message": null
}

===============================================================
프론트 관련 로그 

 GET /view/MAIN_PAGE 200 in 14.5s (compile: 11.4s, render: 3.1s)
 GET /view/AI_ENGLISH_CHAT_PAGE 200 in 1697ms (compile: 153ms, render: 1543ms)
 GET /view/AI_ENGLISH_CHAT_PAGE 200 in 180ms (compile: 48ms, render: 132ms)
 GET /view/MEMBERSHIP_SHOP_PAGE 200 in 993ms (compile: 418ms, render: 576ms)

=================================================================
서버 관련 로그 

dlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.web.servlet.DispatcherServlet - GET "/api/v1/user-memberships/current", parameters={}
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:05 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.jdbc.datasource.DataSourceUtils - Setting JDBC Connection [HikariProxyConnection@448292496 wrapping org.postgresql.jdbc.PgConnection@5ed52666] read-only
Hibernate:
    select
        u1_0.id,
        u1_0.created_at,
        u1_0.expires_at,
        u1_0.granted_by,
        u1_0.membership_id,
        m1_0.id,
        m1_0.can_analyze,
        m1_0.can_converse,
        m1_0.can_learn,
        m1_0.created_at,
        m1_0.description,
        m1_0.duration_days,
        m1_0.name,
        m1_0.price_cents,
        m1_0.updated_at,
        u1_0.started_at,
        u1_0.status,
        u1_0.updated_at,
        u1_0.user_id
    from
        user_memberships u1_0
    join
        memberships m1_0
            on m1_0.id=u1_0.membership_id
    where
        u1_0.user_id=?
        and u1_0.status='active'
        and u1_0.expires_at>?
    order by
        u1_0.created_at desc
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.jdbc.datasource.DataSourceUtils - Resetting read-only flag of JDBC Connection [HikariProxyConnection@448292496 wrapping org.postgresql.jdbc.PgConnection@5ed52666]
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.HttpEntityMethodProcessor - Using 'application/json', given [application/json, text/plain, */*] and supported [application/json, application/*+json]
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.w.s.m.m.a.HttpEntityMethodProcessor - Writing [com.domain.demo_backend.global.common.response.ApiResponse@505f8d7f]
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-1] DEBUG o.s.web.servlet.DispatcherServlet - Completed 200 OK
Hibernate:
    select
        u1_0.user_sqno,
        u1_0.created_at,
        u1_0.del_yn,
        u1_0.detail_address,
        u1_0.drug_using_type,
        u1_0.email,
        u1_0.hashed_password,
        u1_0.password,
        u1_0.phone,
        u1_0.road_address,
        u1_0.role,
        u1_0.social_type,
        u1_0.time_using_type,
        u1_0.updated_at,
        u1_0.user_id,
        u1_0.verification_code,
        u1_0.verification_expired_at,
        u1_0.verify_yn,
        u1_0.withdraw_at,
        u1_0.zip_code
    from
        users u1_0
    where
        u1_0.email=?
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.web.servlet.DispatcherServlet - GET "/api/v1/user-memberships/current", parameters={}
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.membership.controller.UserMembershipController#getCurrent(CustomUserDetails)
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.jdbc.datasource.DataSourceUtils - Setting JDBC Connection [HikariProxyConnection@888027851 wrapping org.postgresql.jdbc.PgConnection@35814823] read-only
Hibernate:
    select
        u1_0.id,
        u1_0.created_at,
        u1_0.expires_at,
        u1_0.granted_by,
        u1_0.membership_id,
        m1_0.id,
        m1_0.can_analyze,
        m1_0.can_converse,
        m1_0.can_learn,
        m1_0.created_at,
        m1_0.description,
        m1_0.duration_days,
        m1_0.name,
        m1_0.price_cents,
        m1_0.updated_at,
        u1_0.started_at,
        u1_0.status,
        u1_0.updated_at,
        u1_0.user_id
    from
        user_memberships u1_0
    join
        memberships m1_0
            on m1_0.id=u1_0.membership_id
    where
        u1_0.user_id=?
        and u1_0.status='active'
        and u1_0.expires_at>?
    order by
        u1_0.created_at desc
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.jdbc.datasource.DataSourceUtils - Resetting read-only flag of JDBC Connection [HikariProxyConnection@888027851 wrapping org.postgresql.jdbc.PgConnection@35814823]
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.HttpEntityMethodProcessor - Using 'application/json', given [application/json, text/plain, */*] and supported [application/json, application/*+json]
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.w.s.m.m.a.HttpEntityMethodProcessor - Writing [com.domain.demo_backend.global.common.response.ApiResponse@25685d10]
2026-03-14 00:36:06 [http-nio-0.0.0.0-8080-exec-8] DEBUG o.s.web.servlet.DispatcherServlet - Completed 200 OK
Hibernate: 
    select
        u1_0.user_sqno,
        u1_0.created_at,
        u1_0.del_yn,
        u1_0.detail_address,
        u1_0.drug_using_type,
        u1_0.email,
        u1_0.hashed_password,
        u1_0.password,
        u1_0.phone,
        u1_0.road_address,
        u1_0.role,
        u1_0.social_type,
        u1_0.time_using_type,
        u1_0.updated_at,
        u1_0.user_id,
        u1_0.verification_code,
        u1_0.verification_expired_at,
        u1_0.verify_yn,
        u1_0.withdraw_at,
        u1_0.zip_code
    from
        users u1_0 
    where
        u1_0.email=?
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.servlet.DispatcherServlet - POST "/api/ai/stt", parameters={multipart}
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiSttController#transcribe(MultipartFile, String, CustomUserDetails)
2026-03-14 00:38:33 [http-nio-0.0.0.0-8080-exec-4] INFO  c.d.d.d.a.controller.AiSttController - STT 요청 - userId=2, language=en, size=104624bytes
2026-03-14 00:38:34 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.client.RestTemplate - HTTP POST https://api.openai.com/v1/audio/transcriptions
2026-03-14 00:38:34 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.client.RestTemplate - Accept=[application/json, application/*+json]
2026-03-14 00:38:34 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.client.RestTemplate - Writing [{file=[Byte array resource [resource loaded from byte array]], model=[whisper-1], language=[en]}] as "multipart/form-data"
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.client.RestTemplate - Response 200 OK
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.client.RestTemplate - Reading to [java.util.Map<?, ?>]
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-4] INFO  c.d.d.domain.ai.service.SttService - STT 완료: 13자
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.HttpEntityMethodProcessor - Using 'application/json', given [application/json, text/plain, */*] and supported [application/json, application/*+json]
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.w.s.m.m.a.HttpEntityMethodProcessor - Writing [com.domain.demo_backend.global.common.response.ApiResponse@d6cb60f] 
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-4] DEBUG o.s.web.servlet.DispatcherServlet - Completed 200 OK
Hibernate:
    select
        u1_0.user_sqno,
        u1_0.created_at,
        u1_0.del_yn,
        u1_0.detail_address,
        u1_0.drug_using_type,
        u1_0.email,
        u1_0.hashed_password,
        u1_0.password,
        u1_0.phone,
        u1_0.road_address,
        u1_0.role,
        u1_0.social_type,
        u1_0.time_using_type,
        u1_0.updated_at,
        u1_0.user_id,
        u1_0.verification_code,
        u1_0.verification_expired_at,
        u1_0.verify_yn,
        u1_0.withdraw_at,
        u1_0.zip_code
    from
        users u1_0
    where
        u1_0.email=?
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.web.servlet.DispatcherServlet - POST "/api/ai/chat/stream", parameters={}
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to com.domain.demo_backend.domain.ai.controller.AiChatController#streamChat(ChatRequest, CustomUserDetails)
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.s.m.m.a.RequestResponseBodyMethodProcessor - Read "application/json;charset=UTF-8" to [com.domain.demo_backend.domain.ai.dto.ChatRequest@2a0c2380]
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] INFO  c.d.d.d.a.c.AiChatController - 채팅 스트리밍 요청 - userId=2
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.w.c.r.async.WebAsyncManager - Started async request
2026-03-14 00:38:37 [http-nio-0.0.0.0-8080-exec-6] DEBUG o.s.web.servlet.DispatcherServlet - Exiting but response remains open for further handling
2026-03-14 00:38:38 [sse-6] DEBUG o.s.w.c.r.async.WebAsyncManager - Async result set, dispatch to /api/ai/chat/stream
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] ERROR o.a.c.c.C.[.[.[.[dispatcherServlet] - Servlet.service() for servlet [dispatcherServlet] threw exception
org.springframework.security.access.AccessDeniedException: Access Denied
        at org.springframework.security.web.access.intercept.AuthorizationFilter.doFilter(AuthorizationFilter.java:98)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:126)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:120)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.authentication.AnonymousAuthenticationFilter.doFilter(AnonymousAuthenticationFilter.java:100)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.servletapi.SecurityContextHolderAwareRequestFilter.doFilter(SecurityContextHolderAwareRequestFilter.java:179)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.savedrequest.RequestCacheAwareFilter.doFilter(RequestCacheAwareFilter.java:63)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:82)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:69)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.FilterChainProxy.doFilterInternal(FilterChainProxy.java:233)
        at org.springframework.security.web.FilterChainProxy.doFilter(FilterChainProxy.java:191)
        at org.springframework.web.filter.DelegatingFilterProxy.invokeDelegate(DelegatingFilterProxy.java:352)
        at org.springframework.web.filter.DelegatingFilterProxy.doFilter(DelegatingFilterProxy.java:268)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.RequestContextFilter.doFilterInternal(RequestContextFilter.java:100)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:116)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.apache.catalina.core.ApplicationDispatcher.invoke(ApplicationDispatcher.java:642)
        at org.apache.catalina.core.ApplicationDispatcher.doDispatch(ApplicationDispatcher.java:570)
        at org.apache.catalina.core.ApplicationDispatcher.dispatch(ApplicationDispatcher.java:541)
        at org.apache.catalina.core.AsyncContextImpl$AsyncRunnable.run(AsyncContextImpl.java:564)
        at org.apache.catalina.core.AsyncContextImpl.doInternalDispatch(AsyncContextImpl.java:338)
        at org.apache.catalina.core.StandardWrapperValve.invoke(StandardWrapperValve.java:165)
        at org.apache.catalina.core.StandardContextValve.invoke(StandardContextValve.java:90)
        at org.apache.catalina.authenticator.AuthenticatorBase.invoke(AuthenticatorBase.java:482)
        at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:115)
        at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:93) 
        at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:74)
        at org.apache.catalina.valves.RemoteIpValve.invoke(RemoteIpValve.java:735)      
        at org.apache.catalina.connector.CoyoteAdapter.asyncDispatch(CoyoteAdapter.java:235)
        at org.apache.coyote.AbstractProcessor.dispatch(AbstractProcessor.java:243)     
        at org.apache.coyote.AbstractProcessorLight.process(AbstractProcessorLight.java:57)
        at org.apache.coyote.AbstractProtocol$ConnectionHandler.process(AbstractProtocol.java:894)
        at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.doRun(NioEndpoint.java:1740)
        at org.apache.tomcat.util.net.SocketProcessorBase.run(SocketProcessorBase.java:52)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1191)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:659)
        at org.apache.tomcat.util.threads.TaskThread$WrappingRunnable.run(TaskThread.java:61)
        at java.base/java.lang.Thread.run(Thread.java:840)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] ERROR o.a.c.c.C.[.[.[.[dispatcherServlet] - Servlet.service() for servlet [dispatcherServlet] in context with path [] threw exception [Unable to handle the Spring Security Exception because the response is already committed.] with root cause
org.springframework.security.access.AccessDeniedException: Access Denied
        at org.springframework.security.web.access.intercept.AuthorizationFilter.doFilter(AuthorizationFilter.java:98)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:126)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:120)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.authentication.AnonymousAuthenticationFilter.doFilter(AnonymousAuthenticationFilter.java:100)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.servletapi.SecurityContextHolderAwareRequestFilter.doFilter(SecurityContextHolderAwareRequestFilter.java:179)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.savedrequest.RequestCacheAwareFilter.doFilter(RequestCacheAwareFilter.java:63)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:82)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:69)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.FilterChainProxy.doFilterInternal(FilterChainProxy.java:233)
        at org.springframework.security.web.FilterChainProxy.doFilter(FilterChainProxy.java:191)
        at org.springframework.web.filter.DelegatingFilterProxy.invokeDelegate(DelegatingFilterProxy.java:352)
        at org.springframework.web.filter.DelegatingFilterProxy.doFilter(DelegatingFilterProxy.java:268)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.RequestContextFilter.doFilterInternal(RequestContextFilter.java:100)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:116)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.apache.catalina.core.ApplicationDispatcher.invoke(ApplicationDispatcher.java:642)
        at org.apache.catalina.core.ApplicationDispatcher.doDispatch(ApplicationDispatcher.java:570)
        at org.apache.catalina.core.ApplicationDispatcher.dispatch(ApplicationDispatcher.java:541)
        at org.apache.catalina.core.AsyncContextImpl$AsyncRunnable.run(AsyncContextImpl.java:564)
        at org.apache.catalina.core.AsyncContextImpl.doInternalDispatch(AsyncContextImpl.java:338)
        at org.apache.catalina.core.StandardWrapperValve.invoke(StandardWrapperValve.java:165)
        at org.apache.catalina.core.StandardContextValve.invoke(StandardContextValve.java:90)
        at org.apache.catalina.authenticator.AuthenticatorBase.invoke(AuthenticatorBase.java:482)
        at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:115)
        at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:93) 
        at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:74)
        at org.apache.catalina.valves.RemoteIpValve.invoke(RemoteIpValve.java:735)      
        at org.apache.catalina.connector.CoyoteAdapter.asyncDispatch(CoyoteAdapter.java:235)
        at org.apache.coyote.AbstractProcessor.dispatch(AbstractProcessor.java:243)     
        at org.apache.coyote.AbstractProcessorLight.process(AbstractProcessorLight.java:57)
        at org.apache.coyote.AbstractProtocol$ConnectionHandler.process(AbstractProtocol.java:894)
        at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.doRun(NioEndpoint.java:1740)
        at org.apache.tomcat.util.net.SocketProcessorBase.run(SocketProcessorBase.java:52)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1191)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:659)
        at org.apache.tomcat.util.threads.TaskThread$WrappingRunnable.run(TaskThread.java:61)
        at java.base/java.lang.Thread.run(Thread.java:840)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] ERROR o.a.c.c.C.[.[.[.[dispatcherServlet] - Servlet.service() for servlet [dispatcherServlet] threw exception
org.springframework.security.access.AccessDeniedException: Access Denied
        at org.springframework.security.web.access.intercept.AuthorizationFilter.doFilter(AuthorizationFilter.java:98)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:126)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:120)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.authentication.AnonymousAuthenticationFilter.doFilter(AnonymousAuthenticationFilter.java:100)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.servletapi.SecurityContextHolderAwareRequestFilter.doFilter(SecurityContextHolderAwareRequestFilter.java:179)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.savedrequest.RequestCacheAwareFilter.doFilter(RequestCacheAwareFilter.java:63)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:82)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:69)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.FilterChainProxy.doFilterInternal(FilterChainProxy.java:233)
        at org.springframework.security.web.FilterChainProxy.doFilter(FilterChainProxy.java:191)
        at org.springframework.web.filter.DelegatingFilterProxy.invokeDelegate(DelegatingFilterProxy.java:352)
        at org.springframework.web.filter.DelegatingFilterProxy.doFilter(DelegatingFilterProxy.java:268)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.RequestContextFilter.doFilterInternal(RequestContextFilter.java:100)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:116)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.apache.catalina.core.ApplicationDispatcher.invoke(ApplicationDispatcher.java:642)
        at org.apache.catalina.core.ApplicationDispatcher.doInclude(ApplicationDispatcher.java:520)
        at org.apache.catalina.core.ApplicationDispatcher.include(ApplicationDispatcher.java:463)
        at org.apache.catalina.core.StandardHostValve.custom(StandardHostValve.java:343)
        at org.apache.catalina.core.StandardHostValve.status(StandardHostValve.java:222)
        at org.apache.catalina.core.StandardHostValve.throwable(StandardHostValve.java:308)
        at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:149)
        at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:93) 
        at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:74)
        at org.apache.catalina.valves.RemoteIpValve.invoke(RemoteIpValve.java:735)      
        at org.apache.catalina.connector.CoyoteAdapter.asyncDispatch(CoyoteAdapter.java:235)
        at org.apache.coyote.AbstractProcessor.dispatch(AbstractProcessor.java:243)     
        at org.apache.coyote.AbstractProcessorLight.process(AbstractProcessorLight.java:57)
        at org.apache.coyote.AbstractProtocol$ConnectionHandler.process(AbstractProtocol.java:894)
        at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.doRun(NioEndpoint.java:1740)
        at org.apache.tomcat.util.net.SocketProcessorBase.run(SocketProcessorBase.java:52)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1191)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:659)
        at org.apache.tomcat.util.threads.TaskThread$WrappingRunnable.run(TaskThread.java:61)
        at java.base/java.lang.Thread.run(Thread.java:840)
2026-03-14 00:38:39 [http-nio-0.0.0.0-8080-exec-10] ERROR o.a.c.c.C.[Tomcat].[localhost] - Exception Processing ErrorPage[errorCode=0, location=/error]
jakarta.servlet.ServletException: Unable to handle the Spring Security Exception because the response is already committed.
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:144)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:120)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.authentication.AnonymousAuthenticationFilter.doFilter(AnonymousAuthenticationFilter.java:100)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.servletapi.SecurityContextHolderAwareRequestFilter.doFilter(SecurityContextHolderAwareRequestFilter.java:179)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.savedrequest.RequestCacheAwareFilter.doFilter(RequestCacheAwareFilter.java:63)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:82)
        at org.springframework.security.web.context.SecurityContextHolderFilter.doFilter(SecurityContextHolderFilter.java:69)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.FilterChainProxy.doFilterInternal(FilterChainProxy.java:233)
        at org.springframework.security.web.FilterChainProxy.doFilter(FilterChainProxy.java:191)
        at org.springframework.web.filter.DelegatingFilterProxy.invokeDelegate(DelegatingFilterProxy.java:352)
        at org.springframework.web.filter.DelegatingFilterProxy.doFilter(DelegatingFilterProxy.java:268)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.RequestContextFilter.doFilterInternal(RequestContextFilter.java:100)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:116)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:101)
        at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:174)
        at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)
        at org.apache.catalina.core.ApplicationDispatcher.invoke(ApplicationDispatcher.java:642)
        at org.apache.catalina.core.ApplicationDispatcher.doInclude(ApplicationDispatcher.java:520)
        at org.apache.catalina.core.ApplicationDispatcher.include(ApplicationDispatcher.java:463)
        at org.apache.catalina.core.StandardHostValve.custom(StandardHostValve.java:343)
        at org.apache.catalina.core.StandardHostValve.status(StandardHostValve.java:222)
        at org.apache.catalina.core.StandardHostValve.throwable(StandardHostValve.java:308)
        at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:149)
        at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:93) 
        at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:74)
        at org.apache.catalina.valves.RemoteIpValve.invoke(RemoteIpValve.java:735)      
        at org.apache.catalina.connector.CoyoteAdapter.asyncDispatch(CoyoteAdapter.java:235)
        at org.apache.coyote.AbstractProcessor.dispatch(AbstractProcessor.java:243)     
        at org.apache.coyote.AbstractProcessorLight.process(AbstractProcessorLight.java:57)
        at org.apache.coyote.AbstractProtocol$ConnectionHandler.process(AbstractProtocol.java:894)
        at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.doRun(NioEndpoint.java:1740)
        at org.apache.tomcat.util.net.SocketProcessorBase.run(SocketProcessorBase.java:52)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1191)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:659)
        at org.apache.tomcat.util.threads.TaskThread$WrappingRunnable.run(TaskThread.java:61)
        at java.base/java.lang.Thread.run(Thread.java:840)
Caused by: org.springframework.security.access.AccessDeniedException: Access Denied     
        at org.springframework.security.web.access.intercept.AuthorizationFilter.doFilter(AuthorizationFilter.java:98)
        at org.springframework.security.web.FilterChainProxy$VirtualFilterChain.doFilter(FilterChainProxy.java:374)
        at org.springframework.security.web.access.ExceptionTranslationFilter.doFilter(ExceptionTranslationFilter.java:126)
        ... 57 common frames omitted

> IDLE
> :bootRun

$