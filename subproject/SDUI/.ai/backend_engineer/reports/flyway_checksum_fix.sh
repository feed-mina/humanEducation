#!/bin/bash
# Flyway checksum 문제 해결 스크립트
# 작성일: 2026-03-03
# 목적: Flyway validation을 비활성화하고 컨테이너 재시작

echo "=== Step 1: 현재 컨테이너 상태 확인 ==="
docker ps -a | grep sdui-backend-lab

echo ""
echo "=== Step 2: 기존 컨테이너 중지 및 제거 ==="
docker stop sdui-backend-lab 2>/dev/null || echo "컨테이너가 실행 중이 아닙니다."
docker rm sdui-backend-lab 2>/dev/null || echo "제거할 컨테이너가 없습니다."

echo ""
echo "=== Step 3: 환경 변수 설정 ==="
export MAIL_PW="xbfkdgfxevqytije"
export JWT_KEY="d6ac9ecc0a3aa3c395313fb236e0ec10d71ab78fb36f54ba626664eba0b842b1"

echo ""
echo "=== Step 4: Flyway validation 비활성화하고 컨테이너 시작 ==="
docker run -d \
  --name sdui-backend-lab \
  -p 8081:8080 \
  --network sdui-network \
  -e SPRING_PROFILES_ACTIVE=prod \
  -e SPRING_DATASOURCE_URL=jdbc:postgresql://sdui-db:5432/SDUI_LAB \
  -e SPRING_DATASOURCE_USERNAME=mina \
  -e SPRING_DATASOURCE_PASSWORD=password \
  -e SPRING_DATA_REDIS_HOST=sdui-redis \
  -e SPRING_MAIL_PASSWORD=$MAIL_PW \
  -e JWT_SECRET_KEY=$JWT_KEY \
  -e SPRING_FLYWAY_VALIDATE_ON_MIGRATE=false \
  yerinmin/sdui-app:lab-claude-dev

echo ""
echo "=== Step 5: 컨테이너 시작 확인 (10초 대기) ==="
sleep 10
docker ps | grep sdui-backend-lab

echo ""
echo "=== Step 6: 로그 확인 ==="
echo "다음 명령으로 로그를 모니터링하세요:"
echo "docker logs -f sdui-backend-lab"
