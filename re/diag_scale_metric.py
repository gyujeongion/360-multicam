"""가설 검증: cam 프레임 해상도를 바꿔 같은 얼굴을 검출하면 '가운데' 판정이 달라지는가(검출기 크기 의존)."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, ".")
from orbcut.faces import FaceKit
out, pid = sys.argv[1], sys.argv[2]
rows = json.load(open(f"{out}/paths.json"))["paths"][pid]
pres = lambda t: rows[min(range(len(rows)), key=lambda k: abs(rows[k][0] - t))][4]
kit = FaceKit()
for w in (1920, 960, 640, 480):
    c = cv2.VideoCapture(f"{out}/cams/cam_{pid}.mp4"); fps = c.get(cv2.CAP_PROP_FPS); i = seen = cen = 0
    while c.grab():
        if i % int(round(fps / 2)) == 0 and pres(i / fps):
            _, f = c.retrieve(); f = cv2.resize(f, (w, w * 9 // 16)); h = f.shape[0]
            fs = kit.detect(f)
            if fs:
                x1, y1, x2, y2, *_ = max(fs, key=lambda b: (b[2]-b[0])*(b[3]-b[1])); seen += 1
                cen += (w/3 <= (x1+x2)/2 <= 2*w/3) and (h/3 <= (y1+y2)/2 <= 2*h/3)
        i += 1
    print(f"{pid} 폭{w}: 얼굴 {seen} · 가운데 {cen/max(seen,1):.0%}")
