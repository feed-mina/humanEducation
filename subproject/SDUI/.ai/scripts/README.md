# DB 백업 / 복구 스크립트

> ⚠️ 이 폴더는 Flyway 마이그레이션 경로 밖에 있습니다. 자동 실행되지 않습니다.

## V8 실행 전 백업 절차

### 1단계 — 백업 생성

```bash
docker exec -i sdui-db psql -U mina -d SDUI_TD < scripts/backup_ui_metadata.sql
```

`ui_metadata_backup_v7` 테이블이 DB 안에 생성됩니다.

### 2단계 — V8 마이그레이션 실행

Spring Boot를 재시작하면 Flyway가 자동으로 V8을 실행합니다.

```bash
./gradlew bootRun
```

### 3단계 — (롤백 필요 시) 복구

```bash
docker exec -i sdui-db psql -U mina -d SDUI_TD < scripts/restore_ui_metadata.sql
```

그 후 Flyway 이력에서 V8을 제거합니다:

```bash
docker exec -i sdui-db psql -U mina -d SDUI_TD -c \
  "DELETE FROM flyway_schema_history WHERE version = '8';"
```

---

## 백업 확인 쿼리

```sql
-- 백업 내용 확인
SELECT component_id, component_type, css_class, parent_group_id
FROM ui_metadata_backup_v7
WHERE screen_id = 'MAIN_PAGE'
ORDER BY sort_order;
```