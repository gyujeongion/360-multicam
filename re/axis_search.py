"""X6 IMU 축 → 렌즈0 카메라 축 대응 실측: 후보마다 자이로 적분 안정화 → 연속 프레임 일치도."""
import itertools, sys, glob
import cv2, numpy as np
sys.path.insert(0, '.')
from multicam360.engine import Projector, equirect_dirs, load_lenses
from multicam360.imu import read_imu

L = load_lenses('re/move006.trailer')
imu = read_imu(open('re/move006.trailer', 'rb').read())
files = sorted(glob.glob('re/move006_frames/f_*.png'))
times = 0.5 + np.arange(len(files)) / 10 + 0.05          # fps=10 프레임 중앙 시각 근사
frames = [cv2.resize(cv2.imread(f), (3840 * 2 // 4, 3840 // 4)) for f in files]   # 1/4 축소 pair
W = 512; D0 = equirect_dirs(W).reshape(-1, 3)
lat = np.arcsin(np.clip(D0[:, 1], -1, 1)); mid = (np.abs(lat) < np.radians(50)).reshape(W // 2, W)

def expm(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def integrate(M, sign):
    t, g = imu['t'], np.radians(imu['gyr']) @ M.T * sign
    sel = (t >= times[0] - 0.2) & (t <= times[-1] + 0.1)
    t, g = t[sel], g[sel]
    R, out, j = np.eye(3), [], 0
    for i in range(1, len(t)):
        R = R @ expm(g[i] * (t[i] - t[i - 1]))
        while j < len(times) and t[i] >= times[j]:
            out.append(R.copy()); j += 1
    return out

def score(Rs):
    imgs = []
    for fr, R in zip(frames, Rs):
        dirs = (D0 @ R).reshape(W // 2, W, 3)          # 월드 방향 → 카메라 방향 = Rᵀ d
        pj = Projector(L, dirs, stream_w=fr.shape[0], flow=False)
        imgs.append(cv2.cvtColor(pj(fr), cv2.COLOR_BGR2GRAY).astype(np.float32))
    return float(np.mean([np.abs(a - b)[mid].mean() for a, b in zip(imgs, imgs[1:])])), imgs

E = np.eye(3); res = []
for ysg in (1, -1):
    for xax, xsg in itertools.product((1, 2), (1, -1)):
        M = np.zeros((3, 3)); M[1, 0] = ysg; M[0, xax] = xsg
        M[2] = np.cross(M[0], M[1])                    # 오른손 좌표계 유지
        for sign in (1, -1):
            s, _ = score(integrate(M, sign))
            res.append((s, ysg, xax, xsg, sign, M.tolist()))
            print(round(s, 2), 'camY=', ysg, 'sx', 'camX=', xsg, 'sensor', 'yz'[xax - 1], 'gyro sign', sign, flush=True)
base, _ = score([np.eye(3)] * len(frames))
res.sort(); print('BASE(no stab)', round(base, 2)); print('BEST', res[0][:5]); print('2nd', res[1][:5])
np.save('re/imu_axis_M.npy', np.array(res[0][5]) * res[0][4])
