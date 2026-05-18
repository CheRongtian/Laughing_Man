FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -L -o yolov8n-face.pt https://huggingface.co/spaces/mainakhf/passport_photo_maker/resolve/main/yolov8n-face.pt

COPY lm.py .

CMD ["python", "lm.py"]