package com.domain.demo_backend.domain.query.repository;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.SelectProvider;

import java.util.List;
import java.util.Map;

@Mapper
public interface CommonMapper {
    // XML 없이 자바 코드로 들어오는 SQL을 그대로 실행하는 설정
    @SelectProvider(type = SqlProvider.class, method = "provideSql")
    List<Map<String, Object>> executeDynamicQuery(String sql, Map<String, Object> params);

    // SQL을 그대로 반환해주는 내부 도우미 클래스
    class SqlProvider {
        public String provideSql(String sql, Map<String, Object> params) {
            return sql; // 전달받은 SQL 문장을 가공 없이 그대로 실행기에 던집니다.
        }
    }
}