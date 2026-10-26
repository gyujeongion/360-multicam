"""진단: 같은 시각, 경로 방향으로 파노라마에서 자른 뷰 vs cam 렌더 프레임 — 렌더 경로가 파노라마와 같은 그림을 내는지."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, ".")
from orbcut.engine import flat_dirs
from orbcut.track import smooth_path
out, pid, dst = sys.argv[1], sys.argv[2], sys.argv[3]
times = [float(x) for x in sys.argv[4].split(",")]
sm = smooth_path(json.load(open(f"{out}/paths.json"))["paths"][pid])
def grab(path, t):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * c.get(cv2.CAP_PROP_FPS)))); _, f = c.read(); return f
tiles = []
for t in times:
    y = float(np.interp(t, sm[:, 0], sm[:, 1])); p = float(np.interp(t, sm[:, 0], sm[:, 2]))
    eq = grab(f"{out}/panorama.mp4", t); H, W = eq.shape[:2]
    d = flat_dirs(y, p, 70, 480, 270).reshape(-1, 3)
    lon = np.arctan2(d[:, 0], d[:, 2]); lat = np.arcsin(np.clip(d[:, 1], -1, 1))
    mx = ((lon + np.pi) / (2*np.pi) * W - .5).astype(np.float32).reshape(270, 480); my = ((lat + np.pi/2) / np.pi * H - .5).astype(np.float32).reshape(270, 480)
    a = cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    b = cv2.resize(grab(f"{out}/cams/cam_{pid}.mp4", t), (480, 270))
    cv2.putText(a, f"pano t{t}", (8, 24), 0, .7, (0, 255, 255), 2); cv2.putText(b, "cam", (8, 24), 0, .7, (0, 255, 255), 2)
    tiles.append(np.vstack([a, b]))
cv2.imwrite(dst, np.hstack(tiles))
