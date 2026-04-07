# ============================================================
# Stage 1: Dependencies (캐시 최적화)
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# 시스템 의존성 (빌드에 필요한 것만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 의존성을 별도 레이어로 분리 → requirements.txt가 안 바뀌면 캐시 히트
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ============================================================
# Stage 2: 모델 다운로드 (별도 스테이지로 분리)
# ============================================================
FROM python:3.11-slim AS downloader

WORKDIR /download

# 다운로더 스크립트에 필요한 최소 파일만 복사
COPY app/__init__.py app/__init__.py
COPY app/download_models.py app/download_models.py

RUN python -m app.download_models


# ============================================================
# Stage 3: Production (최소 런타임 이미지)
# ============================================================
FROM python:3.11-slim AS production

# 메타데이터
LABEL maintainer="face-age-api"
LABEL description="Face Age Prediction API using OpenCV DNN"

# 보안: 비-root 유저 생성
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# OpenCV 런타임에 필요한 최소 시스템 라이브러리만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Stage 1에서 설치된 Python 패키지 복사
COPY --from=builder /install /usr/local

# Stage 2에서 다운로드된 모델 파일 복사
COPY --from=downloader /download/models ./models

# 애플리케이션 코드 복사
COPY app/ ./app/

# 비-root 유저로 전환
USER appuser

# 포트 노출
EXPOSE 8000

# 헬스체크 (30초 간격으로 /health 엔드포인트 확인)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Uvicorn 실행 (워커 2개, graceful shutdown 지원)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-graceful-shutdown", "10"]
