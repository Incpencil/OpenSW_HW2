"""
사전 훈련된 모델 파일을 다운로드하는 스크립트.
- 얼굴 검출: OpenCV DNN SSD (Caffe)
- 나이 예측: 8개 구간 분류 모델 (Caffe)
"""

import os
import urllib.request

# 프로젝트 루트의 models 디렉토리
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 다운로드할 모델 파일 목록
MODEL_URLS = {
    # 얼굴 검출 모델 (SSD + ResNet-10)
    "face_deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "face_net.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
    # 나이 예측 모델
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/"
        "master/age_net_definitions/deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/"
        "master/models/age_net.caffemodel"
    ),
}


def download_models():
    """모델 파일이 없으면 다운로드합니다."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"[SKIP] {filename} 이미 존재")
            continue

        print(f"[DOWN] {filename} 다운로드 중...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[DONE] {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[FAIL] {filename} 다운로드 실패: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            raise


if __name__ == "__main__":
    download_models()
