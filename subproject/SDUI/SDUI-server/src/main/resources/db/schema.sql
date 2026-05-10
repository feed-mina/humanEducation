create table email_verification
(
    email_id          bigint auto_increment
        primary key,
    user_sqno         bigint null,
    user_id           varchar(50) null,
    email             varchar(255) null,
    verification_code varchar(10) not null,
    expires_at        timestamp null
);

create table user_info
(
    id                               bigint not null
        primary key,
    connected_at                     timestamp null,
    profile_nickname_needs_agreement tinyint(1)   null,
    profile_image_needs_agreement    tinyint(1)   null,
    has_email                        tinyint(1)   null,
    email_needs_agreement            tinyint(1)   null,
    is_email_valid                   tinyint(1)   null,
    is_email_verified                tinyint(1)   null,
    email                            varchar(255) null,
    has_age_range                    tinyint(1)   null,
    age_range_needs_agreement        tinyint(1)   null,
    has_birthday                     tinyint(1)   null,
    birthday_needs_agreement         tinyint(1)   null,
    has_gender                       tinyint(1)   null
);

create table users
(
    user_sqno               bigint auto_increment
        primary key,
    user_id                 varchar(50)           not null,
    username                varchar(100)          not null,
    password                varchar(255)          not null,
    hashed_password         varchar(255) null,
    role                    varchar(50) null,
    phone                   varchar(20) null,
    email                   varchar(100) null,
    created_at              timestamp default CURRENT_TIMESTAMP null,
    updated_at              timestamp default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    nickname                varchar(255) null,
    delYn                   char      default 'N' null,
    verifyYn                char      default 'N' not null comment '이메일 인증 여부 (N:미인증, Y:인증)',
    socialType              varchar(20) null comment '소셜 로그인 타입 (K:카카오)',
    verification_code       varchar(10) null comment '이메일 인증 코드',
    verification_expired_at datetime null comment '인증코드 만료시간',
    constraint unique_email
        unique (email)
);

create table chats
(
    chat_id       bigint auto_increment
        primary key,
    sender_sqno   bigint null,
    receiver_sqno bigint null,
    message       text not null,
    sent_at       timestamp default CURRENT_TIMESTAMP null,
    constraint chats_ibfk_1
        foreign key (sender_sqno) references users (user_sqno)
            on delete cascade,
    constraint chats_ibfk_2
        foreign key (receiver_sqno) references users (user_sqno)
            on delete cascade
);

create index receiver_sqno
    on chats (receiver_sqno);

create index sender_sqno
    on chats (sender_sqno);

create table diary
(
    diary_id            bigint auto_increment
        primary key,
    user_id             varchar(255) null,
    user_sqno           bigint null,
    author              varchar(255) null,
    title               varchar(255) not null,
    content             text         not null,
    tag1                varchar(255) null,
    tag2                varchar(255) null,
    tag3                varchar(255) null,
    emotion             int null,
    diary_status        enum ('true', 'false') default 'false'           null,
    diary_type          varchar(20) null,
    del_yn              char      default 'N' null,
    del_dt              timestamp null,
    reg_dt              timestamp default CURRENT_TIMESTAMP null,
    updt_dt             timestamp default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    frst_reg_ip         varchar(45) null,
    last_updt_ip        varchar(45) null,
    frst_rgst_usps_sqno bigint null,
    last_updt_usps_sqno bigint null,
    constraint diary_ibfk_1
        foreign key (user_sqno) references users (user_sqno)
            on delete cascade
);

create index user_sqno
    on diary (user_sqno);

create table diary_views
(
    view_id     bigint auto_increment
        primary key,
    diary_id    bigint null,
    viewer_sqno bigint null,
    viewed_at   timestamp default CURRENT_TIMESTAMP null,
    constraint diary_views_ibfk_1
        foreign key (diary_id) references diary (diary_id)
            on delete cascade,
    constraint diary_views_ibfk_2
        foreign key (viewer_sqno) references users (user_sqno)
            on delete cascade
);

create index diary_id
    on diary_views (diary_id);

create index viewer_sqno
    on diary_views (viewer_sqno);

create table webrtc_connections
(
    connection_id bigint auto_increment
        primary key,
    caller_sqno   bigint null,
    receiver_sqno bigint null,
    status        varchar(20) default 'active' null,
    created_at    timestamp   default CURRENT_TIMESTAMP null,
    ended_at      timestamp null,
    constraint webrtc_connections_ibfk_1
        foreign key (caller_sqno) references users (user_sqno)
            on delete cascade,
    constraint webrtc_connections_ibfk_2
        foreign key (receiver_sqno) references users (user_sqno)
            on delete cascade
);

create index caller_sqno
    on webrtc_connections (caller_sqno);

create index receiver_sqno
    on webrtc_connections (receiver_sqno);


