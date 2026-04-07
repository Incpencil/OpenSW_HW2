"""
얼굴 검출 + 나이 예측 + 성별 예측 모델 로직.
OpenCV DNN 모듈을 사용하여 가벼운 Caffe 모델로 추론합니다.
"""

import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 나이 예측 범위 (8개 구간)
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]

# 성별 라벨
GENDER_LIST = ["Male", "Female"]

# 모델 전처리에 사용되는 평균값 (BGR)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 얼굴 검출 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.7


class FaceAnalyzer:
    """얼굴 검출 + 나이 예측 + 성별 예측 클래스."""

    def __init__(self):
        self.face_net = None
        self.age_net = None
        self.gender_net = None
        self._loaded = False

    def load_models(self):
        """Caffe 모델(얼굴 검출, 나이, 성별)을 메모리에 로드합니다."""
        face_proto = os.path.join(MODEL_DIR, "face_deploy.prototxt")
        face_model = os.path.join(MODEL_DIR, "face_net.caffemodel")
        age_proto = os.path.join(MODEL_DIR, "age_deploy.prototxt")
        age_model = os.path.join(MODEL_DIR, "age_net.caffemodel")
        gender_proto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
        gender_model = os.path.join(MODEL_DIR, "gender_net.caffemodel")

        # 파일 존재 확인
        for path in [face_proto, face_model, age_proto, age_model,
                     gender_proto, gender_model]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"모델 파일이 없습니다: {path}\n"
                    "먼저 python -m app.download_models 를 실행하세요."
                )

        self.face_net = cv2.dnn.readNet(face_model, face_proto)
        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        self._loaded = True
        print("[INFO] 모델 로드 완료 (얼굴 검출 + 나이 + 성별)")

    def _detect_faces(self, frame: np.ndarray) -> list:
        """
        이미지에서 얼굴 영역(bounding box)을 검출합니다.
        Returns: list of (x1, y1, x2, y2, confidence) tuples
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # 경계를 이미지 범위 내로 클램핑
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            boxes.append((x1, y1, x2, y2, float(confidence)))

        return boxes

    def _predict_age(self, face_img: np.ndarray) -> dict:
        """
        얼굴 이미지에서 나이 구간을 예측합니다.
        Returns: {"age_range": str, "confidence": float}
        """
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227),
            MODEL_MEAN_VALUES, swapRB=False
        )
        self.age_net.setInput(blob)
        preds = self.age_net.forward()

        idx = preds[0].argmax()
        age_range = AGE_BUCKETS[idx]
        confidence = float(preds[0][idx])

        return {"age_range": age_range, "confidence": round(confidence, 4)}

    def _predict_gender(self, face_img: np.ndarray) -> dict:
        """
        얼굴 이미지에서 성별을 예측합니다.
        Returns: {"gender": str, "confidence": float}
        """
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227),
            MODEL_MEAN_VALUES, swapRB=False
        )
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()

        idx = preds[0].argmax()
        gender = GENDER_LIST[idx]
        confidence = float(preds[0][idx])

        return {"gender": gender, "confidence": round(confidence, 4)}

    def predict(self, image_bytes: bytes) -> dict:
        """
        이미지 바이트를 받아 얼굴 검출 + 나이 예측 + 성별 예측 결과를 반환합니다.

        Returns:
            {
                "face_count": int,
                "faces": [
                    {
                        "face_id": int,
                        "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                        "detection_confidence": float,
                        "age_range": str,
                        "age_confidence": float,
                        "gender": str,
                        "gender_confidence": float,
                    },
                    ...
                ]
            }
        """
        if not self._loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        # 이미지 디코딩
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 얼굴 검출
        face_boxes = self._detect_faces(frame)

        results = []
        for i, (x1, y1, x2, y2, det_conf) in enumerate(face_boxes):
            # 얼굴 영역 패딩 (더 넓은 컨텍스트 확보)
            padding = 20
            h, w = frame.shape[:2]
            fx1 = max(0, x1 - padding)
            fy1 = max(0, y1 - padding)
            fx2 = min(w - 1, x2 + padding)
            fy2 = min(h - 1, y2 + padding)

            face_img = frame[fy1:fy2, fx1:fx2].copy()
            if face_img.size == 0:
                continue

            age_result = self._predict_age(face_img)
            gender_result = self._predict_gender(face_img)

            results.append({
                "face_id": i + 1,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "detection_confidence": round(det_conf, 4),
                "age_range": age_result["age_range"],
                "age_confidence": age_result["confidence"],
                "gender": gender_result["gender"],
                "gender_confidence": gender_result["confidence"],
            })

        return {
            "face_count": len(results),
            "faces": results,
        }


# 싱글톤 인스턴스 (하위 호환성 유지)
predictor = FaceAnalyzer()
