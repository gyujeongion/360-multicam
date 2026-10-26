"""M7: 사람 정답 라벨 기준 컷 파라미터 튜닝. 앞 절반으로 고르고 뒤 절반으로 검증(과적합 확인).
usage: python re/tune_cut.py <cams_dir> <labels.json>"""
import glob, itertools, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbcut.cut import ASD_FPS, decide, person_scores, smooth
from orbcut.eval import HZ, label_sets

cams_dir, labels_p = sys.argv[1], sys.argv[2]
L = json.load(open(labels_p))
cam_dirs = sorted(d for d in glob.glob(os.path.join(cams_dir, "cam_*")) if os.path.isdir(d))
ids = [os.path.basename(d)[4:] for d in cam_dirs]
n = min(len(glob.glob(os.path.join(d, "pyframes", "*.jpg"))) for d in cam_dirs)
RAW = np.stack([person_scores(d, n) for d in cam_dirs])
t = np.arange(0, n / ASD_FPS, 1 / HZ)
truth = label_sets(L, t)
speak = np.array([bool(s) for s in truth])


def acc(cams, lo, hi):
    m = speak & (t >= lo) & (t < hi)
    return float(np.mean([cams[i] in truth[i] for i in np.where(m)[0]])) if m.any() else float("nan")


def cams_of(segs):
    c = np.array([None] * len(t), dtype=object)
    for s in segs:
        c[(t >= s["start"]) & (t < s["end"])] = s["cam"]
    return c


H = n / ASD_FPS / 2
# 상한: 프레임마다 점수 최댓값 인물(전환 제약 없음)
for sm in (0.2, 0.4, 0.8, 1.2, 2.0):
    S = np.stack([smooth(r, max(1, int(sm * ASD_FPS))) for r in RAW])
    idx = (t * ASD_FPS).astype(int).clip(0, n - 1)
    c = np.array([ids[k] for k in S[:, idx].argmax(0)], dtype=object)
    print(f"상한(평활 {sm}s, 제약 없음): 전체 {acc(c, 0, 1e9):.3f}")

grid = list(itertools.product((0.4, 0.8, 1.2, 2.0), (-1.0, -0.5, 0.0, 0.5), (0.3, 0.6, 1.0, 1.5), (1.0, 1.5, 2.5, 4.0)))
res = []
for sm, thr, hold, ms in grid:
    S = np.stack([smooth(r, max(1, int(sm * ASD_FPS))) for r in RAW])
    c = cams_of(decide(S, ids, ms, hold, thr))
    res.append((acc(c, 0, H), acc(c, H, 1e9), acc(c, 0, 1e9), sm, thr, hold, ms))
res.sort(key=lambda r: -r[0])
print("현재 기본값(0.4, 0.0, 0.5, 1.5):", [round(x, 3) for x in (lambda c: (acc(c, 0, H), acc(c, H, 1e9), acc(c, 0, 1e9)))(
    cams_of(decide(np.stack([smooth(r, int(0.4 * ASD_FPS)) for r in RAW]), ids, 1.5, 0.5, 0.0)))])
print("앞절반 기준 상위 8 (앞, 뒤, 전체 | smooth thr hold min_shot):")
for r in res[:8]:
    print(f"  {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} | {r[3]} {r[4]} {r[5]} {r[6]}")
