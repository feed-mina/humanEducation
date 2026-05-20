package com.domain.demo_backend.domain.community.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.UUID;

@Slf4j
@Service
public class SupabaseStorageService {

    @Value("${kride.supabase.url:}")
    private String supabaseUrl;

    @Value("${kride.supabase.key:}")
    private String supabaseKey;

    @Value("${kride.supabase.bucket:kride-community}")
    private String bucket;

    private final RestTemplate restTemplate = new RestTemplate();

    public String upload(MultipartFile file, Long postId) throws IOException {
        String originalName = file.getOriginalFilename();
        String ext = "";
        if (originalName != null && originalName.contains(".")) {
            ext = originalName.substring(originalName.lastIndexOf("."));
        }
        String storedName = UUID.randomUUID() + ext;
        String objectPath = "community/" + postId + "/" + storedName;

        String uploadUrl = supabaseUrl + "/storage/v1/object/" + bucket + "/" + objectPath;

        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + supabaseKey);
        headers.set("apikey", supabaseKey);
        headers.setContentType(MediaType.parseMediaType(
                file.getContentType() != null ? file.getContentType() : "application/octet-stream"));

        HttpEntity<byte[]> entity = new HttpEntity<>(file.getBytes(), headers);

        try {
            restTemplate.exchange(uploadUrl, HttpMethod.POST, entity, String.class);
        } catch (Exception e) {
            log.error("Supabase Storage 업로드 실패: {}", e.getMessage());
            throw new IOException("이미지 업로드에 실패했습니다.", e);
        }

        return supabaseUrl + "/storage/v1/object/public/" + bucket + "/" + objectPath;
    }

    public String getStoredName(String publicUrl) {
        if (publicUrl == null) return null;
        int lastSlash = publicUrl.lastIndexOf("/");
        return lastSlash >= 0 ? publicUrl.substring(lastSlash + 1) : publicUrl;
    }
}
