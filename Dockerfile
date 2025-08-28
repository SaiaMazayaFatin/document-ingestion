FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: ffmpeg (audio), build-essential/libpq-dev (psycopg2), curl (debug)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python","scripts/run_audio.py"]