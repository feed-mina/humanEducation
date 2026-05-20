FROM pytorch/torchserve:latest-gpu

# 핸들러 의존성 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 핸들러 + 설정 복사
COPY handlers/ /home/model-store/handlers/
COPY config.properties /home/model-store/config.properties

# .mar 파일은 빌드 시 COPY 또는 런타임에 볼륨 마운트
COPY *.mar /home/model-store/

EXPOSE 8085 8086 7070

CMD ["torchserve", \
     "--start", \
     "--model-store", "/home/model-store", \
     "--ts-config", "/home/model-store/config.properties", \
     "--ncs"]
