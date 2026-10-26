"""진단: cam_X 렌더에서 얼굴 중심 오차(도)를 경로 각속도·found와 대조 — 지연(평활)인지 체계 편향인지 가른다."""
import json, math, sys
import cv2, numpy as np
sys.path.insert(0, ".")
from orbcut.faces import FaceKit, SAME_PERSON
from orbcut.track import smooth_path
out, pid = sys.argv[1], sys.argv[2]
r = json.load(open(f"{out}/render.json")) if False else None
paths = json.load(open(f"{out}/paths.json"))["paths"][pid]
sm = smooth_path(paths)
raw = np.array([(p[0], p[1], p[2]) for p in paths]); raw[:, 1] = np.degrees(np.unwrap(np.radians(raw[:, 1])))
found = np.array([p[3] for p in paths]); pres = np.array([p[4] if len(p) > 4 else True for p in paths])
kit = FaceKit()
cap = cv2.VideoCapture(f"{out}/cams/cam_{pid}.mp4"); vfps = cap.get(cv2.CAP_PROP_FPS); i = 0
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH); H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
hfov = 70.0; f = (W / 2) / math.tan(math.radians(hfov / 2))
rows = []
while cap.grab():
    if i % int(round(vfps / 2)) == 0:
        t = i / vfps; k = int(np.argmin(abs(raw[:, 0] - t)))
        if pres[k]:
            _, fr = cap.retrieve(); fs = kit.detect(fr)
            if fs:
                x1, y1, x2, y2, _, row = max(fs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                ex = math.degrees(math.atan(((x1+x2)/2 - W/2) / f)); ey = math.degrees(math.atan(((y1+y2)/2 - H/2) / f))
                vel = (sm[min(k+1, len(sm)-1), 1] - sm[max(k-1, 0), 1]) / max(sm[min(k+1, len(sm)-1), 0] - sm[max(k-1, 0), 0], 1e-3)
                rows.append((t, ex, ey, (y2-y1)/H, vel, raw[k, 1]-sm[k, 1], raw[k, 2]-sm[k, 2], found[k]))
    i += 1
a = np.array(rows)
print("n", len(a), "ex mean/med/std", a[:,1].mean().round(1), np.median(a[:,1]).round(1), a[:,1].std().round(1),
      "ey mean/std", a[:,2].mean().round(1), a[:,2].std().round(1))
print("corr ex~yaw_vel", np.corrcoef(a[:,1], a[:,4])[0,1].round(2), " corr ex~(raw-sm yaw)", np.corrcoef(a[:,1], a[:,5])[0,1].round(2))
print("found 비율", a[:,7].mean().round(2), "ex found/notfound", a[a[:,7]==1,1].mean().round(1), a[a[:,7]==0,1].mean().round(1) if (a[:,7]==0).any() else '-')
lim_x = math.degrees(math.atan((W/6)/f)); lim_y = math.degrees(math.atan((H/6)/f))
print(f"가운데 박스 한계 ±{lim_x:.1f}°(가로) ±{lim_y:.1f}°(세로)")
for r_ in a: print(" t%.1f ex%+.1f ey%+.1f h%.2f vel%+.0f d_raw%+.1f found%d" % (r_[0], r_[1], r_[2], r_[3], r_[4], r_[5], r_[7]))
