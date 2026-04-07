"""
Face Analysis API Server
FastAPI 기반 얼굴 나이·성별 분석 REST API
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.model import predictor
from app.download_models import download_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 다운로드 및 로드."""
    print("=" * 50)
    print("  Face Analysis API 시작 중...")
    print("=" * 50)

    # 1) 모델 파일 다운로드 (없으면)
    try:
        download_models()
    except Exception as e:
        print(f"[ERROR] 모델 다운로드 실패: {e}")
        raise

    # 2) 모델 메모리 로드 (얼굴 검출 + 나이 + 성별)
    predictor.load_models()

    print("=" * 50)
    print("  API 서버 준비 완료!")
    print("  POST /predict  — 이미지 업로드로 나이·성별 분석")
    print("=" * 50)

    yield  # 서버 실행 중

    print("[INFO] 서버 종료")


app = FastAPI(
    title="Face Analysis API",
    description="이미지를 업로드하면 얼굴을 검출하고 나이 구간 및 성별을 예측합니다.",
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트."""
    return {"status": "healthy", "model_loaded": predictor._loaded}


# ──────────────────────────────────────────────
# 나이·성별 분석 API
# ──────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/predict")
async def predict_face(file: UploadFile = File(..., description="얼굴이 포함된 이미지 파일")):
    """
    이미지를 업로드하면 얼굴을 검출하고 나이·성별을 예측합니다.

    - **file**: 이미지 파일 (JPEG, PNG, WebP, BMP 지원)
    - **최대 크기**: 10 MB
    - **응답**: 검출된 얼굴 수, 각 얼굴의 위치/나이 구간/성별/신뢰도

    나이 구간: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
    성별: Male, Female
    """
    # 파일 타입 검증
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식입니다: {file.content_type}. "
                   f"지원: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # 파일 읽기
    image_bytes = await file.read()

    # 크기 제한 확인
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"파일 크기가 너무 큽니다 ({len(image_bytes)} bytes). "
                   f"최대 {MAX_FILE_SIZE // (1024 * 1024)} MB 까지 허용됩니다.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    # 예측 수행
    try:
        result = predictor.predict(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )

    # 얼굴 미검출 시
    if result["face_count"] == 0:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "이미지에서 얼굴을 찾지 못했습니다.",
                "data": result,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": f"{result['face_count']}개의 얼굴에서 나이·성별을 분석했습니다.",
            "data": result,
        },
    )
