"""진단: 경로(paths.json) vs 파노라마 원시 얼굴 검출(캐시) — 경로가 얼굴보다 체계적으로 어긋나는지."""
import glob, json, pickle, sys
import numpy as np
out, pid = sys.argv[1], sys.argv[2]
d = pickle.load(open(glob.glob(f"{out}/*.dets_*.pkl")[0], "rb"))
rows = json.load(open(f"{out}/paths.json"))["paths"][pid]
P = np.array([(r[0], r[1], r[2]) for r in rows]); pres = np.array([r[4] for r in rows])
dy, dp = [], []
for t, dets in zip(d["ts"], d["obs"]):
    k = int(np.argmin(abs(P[:, 0] - t)))
    if not pres[k] or not dets: continue
    e = [((y - P[k, 1] + 180) % 360 - 180, p - P[k, 2]) for y, p, *_ in dets]
    b = min(e, key=lambda x: np.hypot(*x))
    if np.hypot(*b) < 25: dy.append(b[0]); dp.append(b[1])
dy, dp = np.array(dy), np.array(dp)
print(f"{pid}: n{len(dy)} 경로-검출 yaw 평균{dy.mean():+.1f} 중앙{np.median(dy):+.1f} 표준편차{dy.std():.1f} | pitch 평균{dp.mean():+.1f} 표준편차{dp.std():.1f}")
