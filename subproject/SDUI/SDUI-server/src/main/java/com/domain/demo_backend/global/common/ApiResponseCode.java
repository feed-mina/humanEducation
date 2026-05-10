package com.domain.demo_backend.global.common;

public class ApiResponseCode {

    public static final ApiResponseDto<String> DEFAULT_OK = new ApiResponseDto<>("Success", "200");
    public static final ApiResponseDto<String> SERVER_ERROR = new ApiResponseDto<>("Internal Server", "500");
    ApiResponseDto<String> response = ApiResponseCode.DEFAULT_OK;

}
