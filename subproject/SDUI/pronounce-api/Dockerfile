FROM python:3.11-slim

# konlpy(JPype)가 Java를 필요로 함
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# TTS 파일 저장 디렉토리 (app.mount("/static", ...) 에서 필요)
RUN mkdir -p static

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
