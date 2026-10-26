"""얼굴 검출(YuNet) + 신원 임베딩(SFace) — OpenCV 내장, 모델은 opencv_zoo(Apache-2.0)에서 받아 캐시.

색 분포 비교는 차창 빛 변화에 같은 사람도 다르게 판정해 폐기(LEDGER). 임베딩 코사인 유사도는 조명·각도 변화에 강하다.
"""
import hashlib, os, urllib.request

import cv2
import numpy as np

CACHE = os.path.expanduser("~/.cache/multicam360/models")
MODELS = {
    "yunet": ("face_detection_yunet_2023mar.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "sface": ("face_recognition_sface_2021dec.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
SAME_PERSON = 0.36      # SFace 코사인 유사도 (원 논문 기준 0.363). 실측: 같은 사람 0.93~0.95, 다른 사람 0.26~0.28


def model_path(key):
    name, url, sha = MODELS[key]
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(url, p + ".part")
        os.replace(p + ".part", p)
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != sha:
        raise RuntimeError(f"모델 체크섬 불일치: {p}")
    return p


class FaceKit:
    def __init__(self, score=0.6):
        self.det = cv2.FaceDetectorYN.create(model_path("yunet"), "", (320, 320), score, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(model_path("sface"), "")

    def detect(self, img):
        """→ [(x1, y1, x2, y2, score, face_row)] face_row = YuNet 원본 행(정렬용 랜드마크 포함)."""
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None:
            return []
        return [(f[0], f[1], f[0] + f[2], f[1] + f[3], f[14], f) for f in faces]

    def embed(self, img, face_row):
        return self.rec.feature(self.rec.alignCrop(img, face_row)).flatten()

    @staticmethod
    def sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
