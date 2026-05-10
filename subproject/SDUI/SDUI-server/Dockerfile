# 1단계: 빌드 스테이지 (도커 안에서 자바 빌드)
FROM eclipse-temurin:17-jdk-jammy AS build
WORKDIR /app
# Gradle 래퍼와 설정 파일들을 먼저 복사 (캐싱 활용)
COPY gradlew .
COPY gradle gradle
COPY build.gradle .
COPY settings.gradle .
# 실행 권한 부여 및 의존성 다운로드
RUN chmod +x gradlew
RUN ./gradlew build -x test --stacktrace || return 0
# 소스 코드 복사 및 빌드
COPY src ./src
RUN ./gradlew clean bootJar -x test
# 소스 코드 복사 및 실제 빌드
COPY src ./src
RUN ./gradlew clean bootJar -x test
# 2단계: 실행 스테이지 (안정적인 Temurin 이미지 사용)
FROM eclipse-temurin:17-jdk-jammy
WORKDIR /app
# 빌드 스테이지에서 생성된 jar 파일만 복사
# 일반 jar 파일이 아닌 실행 가능한 jar만 가져오기 위해 이름 패턴 지정
COPY --from=build /app/build/libs/*-SNAPSHOT.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]