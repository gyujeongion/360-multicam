"""013 차 안 10~40초: 보정없음 / 자이로만 / 자이로+실내기준 — 실내 특징점 드리프트(첫 프레임 대비, ORB 중앙값 px)."""
import glob, sys
import cv2, numpy as np
sys.path.insert(0, '.')
from orbcut.engine import DynamicProjector, equirect_dirs, load_lenses
from orbcut.imu import read_imu
from orbcut.stabilize import orientations
from orbcut.anchor import SceneAnchor

raw_tr = open('re/car013.trailer', 'rb').read()
L = load_lenses('re/car013.trailer'); imu = read_imu(raw_tr)
files = sorted(glob.glob('re/car013/f_*.png'))
times = 10 + np.arange(len(files)) / 5 + 0.08   # fps 필터 프레임 시각 보정(30fps 실측 동기 0.01s)
frames = [cv2.imread(f) for f in files]
S = frames[0].shape[0]
Rg = orientations(imu, times, level=True)
rot_g = lambda t: Rg[min(len(Rg) - 1, int(round((t - times[0]) * 5)))]
anc = SceneAnchor(L, S).estimate(list(zip(times, frames)), rot_g)
print('anchor samples', len(anc.ts), 'inliers median', int(np.median(anc.inliers)))
pj = DynamicProjector(L, equirect_dirs(1024), stream_w=S)
orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
mask = np.zeros((512, 1024), np.uint8); mask[85:427] = 255
def drift(imgs):
    k0, d0 = orb.detectAndCompute(imgs[0], mask); out = []
    for im in imgs[1:]:
        k1, d1 = orb.detectAndCompute(im, mask); ms = bf.match(d0, d1)
        dx = np.array([k1[m.trainIdx].pt[0] - k0[m.queryIdx].pt[0] for m in ms]); dx = (dx + 512) % 1024 - 512
        dy = np.array([k1[m.trainIdx].pt[1] - k0[m.queryIdx].pt[1] for m in ms])
        out.append(np.hypot(np.median(dx), np.median(dy)))
    return float(np.mean(out))
g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
idx = list(range(0, len(frames), 5))
res = {}
for name, Rf in [('none', lambda t: np.eye(3)), ('gyro', rot_g), ('gyro+scene', lambda t: anc.W(t) @ rot_g(t))]:
    imgs = [g(pj(frames[i], Rf(times[i]).T)) for i in idx]
    res[name] = drift(imgs); print(name, round(res[name], 1), flush=True)
    sheet = np.hstack([cv2.resize(imgs[k], (256, 128)) for k in (0, 6, 12, 18, 24, 29)])
    cv2.imwrite(f'work/anchor_{name}.jpg', sheet)
cv2.imwrite('work/anchor_compare.jpg', np.vstack([cv2.imread(f'work/anchor_{n}.jpg') for n in ('none', 'gyro', 'gyro+scene')]))
