"""M23 실측: 프리미어 미리보기(4K 파노라마 + VR 투영) 프레임 vs swap 원본 렌더 프레임 — 같은 시각에서 얼굴 중심 위치와 선명도.
얼굴 중심 차는 화면 폭·높이 대비 %, 선명도는 얼굴 상자 안 Laplacian 분산 비(렌더 / 미리보기).
usage: python re/swap_check.py <render.mp4> <out.json> <시각:미리보기.png> ...   (시각 = 시퀀스 초)"""
import json, os, subprocess, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbcut.faces import model_path  # noqa: E402

W, H = 1920, 1080


def grab(video, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", f"scale={W}:{H}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()


def face(det, img):
    s = cv2.resize(img, (960, 540))
    _, f = det.detect(s)
    if f is None:
        return None
    x, y, w, h = min(f, key=lambda r: abs(r[0] + r[2] / 2 - 480))[:4] * 2
    return float(x), float(y), float(w), float(h)


def main():
    render, out = sys.argv[1], sys.argv[2]
    det = cv2.FaceDetectorYN.create(model_path("yunet"), "", (960, 540), 0.6)
    rows = []
    for arg in sys.argv[3:]:
        t, png = arg.split(":", 1)
        a = cv2.resize(cv2.imread(png), (W, H)); b = grab(render, float(t))
        fa, fb = face(det, a), face(det, b)
        if not fa or not fb:
            rows.append({"t": float(t), "face": [bool(fa), bool(fb)]}); continue
        ca = (fa[0] + fa[2] / 2, fa[1] + fa[3] / 2); cb = (fb[0] + fb[2] / 2, fb[1] + fb[3] / 2)
        lap = lambda img, f: float(cv2.Laplacian(cv2.cvtColor(img[int(f[1]):int(f[1] + f[3]), int(f[0]):int(f[0] + f[2])],
                                                              cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        rows.append({"t": float(t), "dx_pct": round((cb[0] - ca[0]) / W * 100, 2), "dy_pct": round((cb[1] - ca[1]) / H * 100, 2),
                     "w_ratio": round(fb[2] / fa[2], 3), "sharp_ratio": round(lap(b, fb) / max(lap(a, fa), 1e-6), 2)})
        cv2.imwrite(os.path.splitext(out)[0] + f"_{t}.jpg", np.hstack([cv2.resize(a, (640, 360)), cv2.resize(b, (640, 360))]))
    ok = [r for r in rows if "dx_pct" in r]
    res = {"frames": rows, "center_diff_pct": [v for r in ok for v in (r["dx_pct"], r["dy_pct"])],
           "sharpness_ratio": round(float(np.median([r["sharp_ratio"] for r in ok])), 2) if ok else None,
           "size_ratio": round(float(np.median([r["w_ratio"] for r in ok])), 3) if ok else None}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
