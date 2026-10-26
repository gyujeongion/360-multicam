"""축 대응 재탐색(24 고유 회전): 013 차 안 3축 움직임. 짧은 창(1초) 안 연속 프레임 일치도로 평가 → 드리프트 영향 배제."""
import glob, itertools, sys
import cv2, numpy as np
sys.path.insert(0, '.')
from orbcut.engine import DynamicProjector, equirect_dirs, load_lenses
from orbcut.imu import read_imu
from orbcut.stabilize import expm, gyro_bias

L = load_lenses('re/car013.trailer'); imu = read_imu(open('re/car013.trailer', 'rb').read())
fs = sorted(glob.glob('re/car013/f_*.png'))[:60]            # 10~22초
frames = [cv2.resize(cv2.imread(f), (960, 480)) for f in fs]
times = 10 + np.arange(len(frames)) / 5 + 0.1
pj = DynamicProjector(L, equirect_dirs(384), stream_w=480)
g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
mid = slice(48, 144)

def rel_rots(M):
    t = imu['t']; G = np.radians(imu['gyr']) @ M.T
    G = G - gyro_bias(imu, M)
    out = []
    for a, b in zip(times, times[1:]):
        R = np.eye(3)
        for i in np.where((t >= a) & (t < b))[0]:
            R = R @ expm(G[i] * 0.001001)
        out.append(R)                                     # 카메라(a) ← 카메라(b) 상대 회전
    return out

def score(M):
    errs = []
    for k, R in enumerate(rel_rots(M)):
        A = g(pj(frames[k], np.eye(3)))                    # 프레임 a를 자기 카메라 좌표로
        B = g(pj(frames[k + 1], R.T))                      # 프레임 b를 a 좌표로 되돌림
        errs.append(np.abs(A - B)[mid].mean())
    return float(np.mean(errs))

base = float(np.mean([np.abs(g(pj(frames[k], np.eye(3))) - g(pj(frames[k + 1], np.eye(3))))[mid].mean() for k in range(len(frames) - 1)]))
res = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        M = np.zeros((3, 3))
        for r, (c, s) in enumerate(zip(perm, signs)):
            M[r, c] = s
        if np.linalg.det(M) < 0:
            continue
        s = score(M); res.append((s, M)); print(round(s, 2), M.astype(int).tolist(), flush=True)
res.sort(key=lambda x: x[0])
print('BASE', round(base, 2)); print('BEST', round(res[0][0], 2), res[0][1].astype(int).tolist()); print('2nd', round(res[1][0], 2), res[1][1].astype(int).tolist())
np.save('re/imu_axis_M2.npy', res[0][1])
