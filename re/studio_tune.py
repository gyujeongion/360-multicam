"""M15 실험: 샘플 프레임만 원본에서 바로 투영해 블렌딩 폭·노이즈 제거·샤프닝·톤을 빠르게 비교(렌더 없이).
usage: python re/studio_tune.py <insv> <studio.mp4> <out_dir>"""
import itertools, json, os, subprocess, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multicam360.engine import Projector, equirect_dirs, load_lenses
from studio_compare import H, W, estimate_rotation, grab, noise, remap_rot, seam_mask, sharpness, ssim_map

TIMES = [3, 10, 17, 24, 31, 38, 45]


def pair_at(insv, t, S=3840):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", f"{t:.3f}", "-i", insv, "-frames:v", "1",
                          "-filter_complex", "[0:0][0:1]hstack,format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(S, 2 * S, 3).copy()


def finish(img, dn, sh):
    out = img
    if dn:
        out = cv2.fastNlMeansDenoisingColored(out, None, dn, dn, 5, 15)
    if sh:
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1 + sh, blur, -sh, 0)
    return out


def main(insv, studio, out):
    os.makedirs(out, exist_ok=True)
    lenses = load_lenses(insv)
    dirs = equirect_dirs(W)
    seam0 = seam_mask()
    frames = []
    for t in TIMES:
        st = grab(studio, t)
        frames.append((t, st, pair_at(insv, t)))
    res = {}
    for feather in (0.12, 0.06, 0.03):
        pj = Projector(lenses, dirs, stream_w=3840, flow=True, feather=feather)
        outs = [(t, st, pj(pair)) for t, st, pair in frames]
        R_cache = {}
        for dn, sh in itertools.product((0, 3, 5), (0, 0.4, 0.8)):
            m = {"sb": [], "ss": [], "nz": [], "sim": [], "sims": [], "L": []}
            for t, st, ou in outs:
                if t not in R_cache:
                    R_cache[t] = estimate_rotation(st, ou)[0]
                R = R_cache[t]
                ou2 = remap_rot(finish(ou, dn, sh), R.T)
                seam = remap_rot(seam0.astype(np.uint8) * 255, R.T) > 127
                lat = np.abs((np.mgrid[0:H, 0:W][0] + 0.5) / H * 180 - 90) < 60
                body = lat & ~seam
                gs, go = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (st, ou2))
                ss = ssim_map(gs, go)
                m["sb"].append(sharpness(go, body) / sharpness(gs, body))
                m["ss"].append(sharpness(go, seam) / sharpness(gs, seam))
                m["nz"].append(noise(go, body) / max(noise(gs, body), 1e-6))
                m["sim"].append(float(ss[body].mean())); m["sims"].append(float(ss[seam].mean()))
            k = f"feather={feather} dn={dn} sh={sh}"
            res[k] = {x: round(float(np.median(v)), 3) for x, v in m.items() if v}
            print(k, res[k], flush=True)
    json.dump(res, open(os.path.join(out, "tune.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
