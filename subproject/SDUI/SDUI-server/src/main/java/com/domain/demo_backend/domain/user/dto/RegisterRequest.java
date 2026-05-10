package com.domain.demo_backend.domain.user.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;

@Getter
@Setter
public class RegisterRequest {
    private String userId;
    @NotBlank(message = "비밀번호는 필수입니다.")
    @JsonAlias("pw")
    private String password;
    private String role;

    @NotBlank(message = "핸드폰 번호는 필수 입력 값입니다.")
    @Pattern(regexp = "^01(?:0|1|[6-9])-?(?:\\d{3}|\\d{4})-?\\d{4}$", message = "올바른 핸드폰 번호 형식이 아닙니다.")
    private String phone;

    @Email(message = "유효한 이메일을 입력하세요.")
    private String email;
    @NotBlank(message = "우편번호는 필수입니다.")
    private String zipCode;
    private String roadAddress;
    private String detailAddress;

    private LocalDate createdAt;
    private String updatedAt;

    private String accessToken;
    //이메일 인증 코드
    private String code;
}
