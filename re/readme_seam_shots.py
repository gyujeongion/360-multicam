"""README용 이음새 전후 사진 후보: 원본 프레임을 equirect로 두 방식(페더만 / 플로우+2밴드) 투영해
이음새를 가로지르는 구간을 잘라 차이가 큰 순으로 저장한다. 얼굴이 잡히는 구간은 버린다(YuNet, 두 방식 모두 검사).
usage: python re/readme_seam_shots.py <out_dir> <src[:meta]> ... [--times 3,9,15] [--top 12]"""
import argparse, json, os, subprocess, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbcut import engine  # noqa: E402
from orbcut.engine import DEC_COLOR, Projector, equirect_dirs, load_lenses  # noqa: E402
from orbcut.faces import model_path  # noqa: E402

W, CW, CH = 5760, 720, 480          # equirect 폭, 자를 크기


def pair_at(src, t, S=3840):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", "-filter_complex",
                          f"[0:0][0:1]hstack,{DEC_COLOR},format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(S, 2 * S, 3).copy()


def grad(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("srcs", nargs="+")
    ap.add_argument("--times", default="2,6,10,14,18,22,26"); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    det = cv2.FaceDetectorYN.create(model_path("yunet"), "", (CW, CH), 0.5)
    dirs = equirect_dirs(W)
    cands = []
    for spec in a.srcs:
        src, meta = (spec.split(":", 1) + [None])[:2]
        if not meta:
            meta = src + ".trailer" if os.path.exists(src + ".trailer") else src
        lenses = load_lenses(meta)
        pf, pb = Projector(lenses, dirs, flow=False), Projector(lenses, dirs, flow=True)
        band = (pf.alpha > 0.05) & (pf.alpha < 0.95)
        for t in [float(x) for x in a.times.split(",")]:
            try:
                pair = pair_at(src, t)
            except (subprocess.CalledProcessError, ValueError):     # 길이 밖 시각
                continue
            pb.prev = {}
            engine.SEAM_MODE = "feather"; of = pf(pair)          # 합성 방식은 호출 시점의 전역값을 읽는다
            engine.SEAM_MODE = "band2"; ob = pb(pair)
            gf, gb = grad(of), grad(ob)
            g0 = grad(cv2.remap(pair, pf.mx[0], pf.my[0], cv2.INTER_LINEAR))     # 렌즈 하나만 투영 — 가짜 에지(단차) 기준
            g1 = grad(cv2.remap(pair, pf.mx[1], pf.my[1], cv2.INTER_LINEAR))
            gmax = np.maximum(g0, g1)
            for cx in (W // 4, 3 * W // 4):                      # 왼쪽·오른쪽 이음새
                for cy in range(W // 2 * 30 // 180, W // 2 * 150 // 180 - CH, CH // 2):   # 위도 ±60° 안
                    x0, y0 = cx - CW // 2, cy
                    m = band[y0:y0 + CH, x0:x0 + CW]
                    if m.mean() < 0.05:
                        continue
                    cf, cb = of[y0:y0 + CH, x0:x0 + CW], ob[y0:y0 + CH, x0:x0 + CW]
                    if any(det.detect(c)[1] is not None for c in (cf, cb)):
                        continue
                    # 2밴드가 되살린 고주파(이중상·블러가 줄어든 만큼) × 구간 안 디테일
                    gain = float((gb[y0:y0 + CH, x0:x0 + CW][m].sum() + 1) / (gf[y0:y0 + CH, x0:x0 + CW][m].sum() + 1))
                    detail = float(gb[y0:y0 + CH, x0:x0 + CW][m].mean())
                    sl = (slice(y0, y0 + CH), slice(x0, x0 + CW))
                    fake = float(np.maximum(0, gb[sl] - gmax[sl])[m].mean() / (gmax[sl][m].mean() + 1e-6))   # 오른쪽에 남는 단차
                    cands.append({"src": os.path.basename(os.path.dirname(src)) or src, "t": t, "x0": x0, "y0": y0,
                                  "gain": round(gain, 3), "detail": round(detail, 1), "fake": round(fake, 3),
                                  "score": round((gain - 1) * np.sqrt(detail) * max(0.0, 1 - 4 * fake), 3),
                                  "img": np.hstack([cf, cb])})
            print(spec, t, len(cands), flush=True)
    cands.sort(key=lambda c: -c["score"])
    meta = []
    for i, c in enumerate(cands[:a.top]):
        name = f"cand_{i:02d}.jpg"
        cv2.imwrite(os.path.join(a.out, name), c.pop("img"), [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta.append({"file": name, **c})
    json.dump(meta, open(os.path.join(a.out, "cands.json"), "w"), indent=1)
    print(json.dumps(meta[:5]))


if __name__ == "__main__":
    main()
