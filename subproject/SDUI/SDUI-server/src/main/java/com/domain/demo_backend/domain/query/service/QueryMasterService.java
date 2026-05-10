package com.domain.demo_backend.domain.query.service;

import com.domain.demo_backend.domain.query.domain.QueryMaster;
import com.domain.demo_backend.domain.query.repository.QueryMasterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class QueryMasterService {
    private final QueryMasterRepository queryMasterRepository;
    private final StringRedisTemplate stringRedisTemplate;

    @Autowired
    public QueryMasterService(QueryMasterRepository queryMasterRepository, StringRedisTemplate stringRedisTemplate) {
        this.queryMasterRepository = queryMasterRepository;
        this.stringRedisTemplate = stringRedisTemplate;
    }

    public QueryMaster getQueryInfo(String sqlKey) {
        // DB에서 해당 키의 전체 정보를 찾아서 반환한다.
        // 필요하다면 여기에서 Redis에 객체 자체를 저장하는 로직 추가 하기
        return queryMasterRepository.findBySqlKey(sqlKey).orElse(null);
    }

    public String getQuery(String sqlKey) {
        // 먼저 Redis에서 해당 키의 SQL이 있는지 확인
        String cachedQuery = stringRedisTemplate.opsForValue().get("SQL:" + sqlKey);
        if (cachedQuery != null) {
            return cachedQuery;
        }

        // Redis에 없다면 DB에서 찾는다
        QueryMaster queryMaster = queryMasterRepository.findBySqlKey(sqlKey)
                .orElseThrow(() -> new RuntimeException("등록되지 않은  sql_key입니다: " + sqlKey));


        // 찾은 쿼리를 다음에 빨리 쓰기 위해 Redis에 저장(캐싱)한다.
        stringRedisTemplate.opsForValue().set("SQL:" + sqlKey, queryMaster.getQueryText());
        return queryMaster.getQueryText();


    }
}
