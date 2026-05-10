#!/bin/bash

# ==========================================
# SDUI 프로젝트 DB 자동 백업 스크립트 (Local Only)
# ==========================================

# 1. 설정 변수
DATE=$(date +%F_%H-%M-%S)
BACKUP_PATH="/home/ubuntu/backups"
DB_CONTAINER="sdui-db"
DB_USER="mina"
DB_NAME="SDUI_TD"  # 운영: SDUI_TD, 테스트: SDUI_LAB
RETENTION_DAYS=7   # 7일 지난 백업 파일은 자동 삭제

# 2. 백업 디렉토리 생성
mkdir -p $BACKUP_PATH

# 3. 파일명 설정
SQL_FILE="$BACKUP_PATH/${DB_NAME}_$DATE.sql"
TAR_FILE="$BACKUP_PATH/${DB_NAME}_$DATE.tar.gz"

echo "[$(date)] 백업 시작: $DB_NAME"

# 4. 백업 실행 (Docker 내부 pg_dump)
# 주의: DB 비밀번호가 필요한 경우 docker exec 명령어 앞에 PGPASSWORD 환경변수를 설정하거나 .pgpass 파일을 사용해야 합니다.
# 예: docker exec -e PGPASSWORD='비밀번호' ...
docker exec $DB_CONTAINER pg_dump -U $DB_USER -d $DB_NAME > $SQL_FILE

# 5. 압축하기
if [ -f "$SQL_FILE" ]; then
    tar -czf $TAR_FILE -C $BACKUP_PATH $(basename $SQL_FILE)
    echo "[$(date)] 압축 완료: $TAR_FILE"

    # 6. 오래된 백업 삭제 (공간 절약)
    find $BACKUP_PATH -name "${DB_NAME}_*.tar.gz" -mtime +$RETENTION_DAYS -delete
    echo "[$(date)] ${RETENTION_DAYS}일 이상 된 오래된 백업 파일 삭제 완료"

    # 7. 임시 SQL 파일 삭제 (압축파일만 보관)
    rm $SQL_FILE
else
    echo "[$(date)] 오류: 백업 파일이 생성되지 않았습니다."
fi