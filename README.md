# 🎯 Face Age Prediction API

이미지를 업로드하면 얼굴을 자동 검출하고 나이 구간을 예측하는 FastAPI 서버입니다.

> **경량 모델 사용**: OpenCV DNN + Caffe 모델 (총 ~15MB)  
> TensorFlow/PyTorch 없이 동작하여 CI/CD 파이프라인에서도 빠르게 빌드됩니다.

---

## 📁 프로젝트 구조

```
face-age-api/
├── app/
│   ├── __init__.py           # 패키지 초기화
│   ├── main.py               # FastAPI 앱 (엔드포인트 정의)
│   ├── model.py              # 얼굴 검출 + 나이 예측 로직
│   └── download_models.py    # 사전 훈련 모델 다운로드 스크립트
├── models/                   # (자동 생성) 다운로드된 모델 파일
│   ├── face_deploy.prototxt
│   ├── face_net.caffemodel
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
├── Dockerfile                # Docker 컨테이너 빌드
├── .gitignore
├── requirements.txt          # Python 의존성
└── README.md
```

---

## 🚀 실행 방법

### 1. 로컬 실행

```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 의존성 설치
pip install -r requirements.txt

# 서버 실행 (모델 자동 다운로드)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker 실행

```bash
docker build -t face-age-api .
docker run -p 8000:8000 face-age-api
```

---

## 📡 API 엔드포인트

### `GET /health` — 서버 상태 확인

```bash
curl http://localhost:8000/health
```

```json
{ "status": "healthy", "model_loaded": true }
```

### `POST /predict` — 나이 예측

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_face.jpg"
```

**응답 예시:**

```json
{
  "success": true,
  "message": "1개의 얼굴을 검출했습니다.",
  "data": {
    "face_count": 1,
    "faces": [
      {
        "face_id": 1,
        "bbox": { "x1": 120, "y1": 50, "x2": 320, "y2": 300 },
        "detection_confidence": 0.9876,
        "age_range": "(25-32)",
        "age_confidence": 0.7432
      }
    ]
  }
}
```

### Swagger UI (자동 생성)

서버 실행 후 브라우저에서 접속:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔧 나이 예측 구간

| 구간 | 의미 |
|------|------|
| (0-2) | 영아 |
| (4-6) | 유아 |
| (8-12) | 어린이 |
| (15-20) | 청소년 |
| (25-32) | 청년 |
| (38-43) | 중년 초기 |
| (48-53) | 중년 |
| (60-100) | 노년 |

---

## 📌 기술 스택

| 항목 | 기술 |
|------|------|
| API 프레임워크 | FastAPI + Uvicorn |
| 얼굴 검출 | OpenCV DNN (SSD + ResNet-10) |
| 나이 예측 | Caffe age classification model |
| 컨테이너 | Docker |
| Python | 3.11+ |
