"""offset_v6 회전 규칙 후보를 겹침 띠 밝기 차이(MAE)로 평가한다."""
import itertools, sys
import cv2, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'multicam360')
from calib import read_metadata_bytes, offset_v6_from_metadata, parse_offset_v6, omni_project, euler

lenses = parse_offset_v6(offset_v6_from_metadata(read_metadata_bytes('re/tail8m.bin')))
S = 0.25  # 평가용 축소
imgs = [cv2.resize(cv2.imread(f're/l{i}.png', cv2.IMREAD_GRAYSCALE), None, fx=S, fy=S, interpolation=cv2.INTER_AREA).astype(np.float32) for i in (0, 1)]
W, H = 1440, 720
lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
lat = (np.arange(H) + 0.5) / H * np.pi - np.pi / 2
LON, LAT = np.meshgrid(lon, lat)
D = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1).reshape(-1, 3)  # y 아래, z 앞

def F(axis):
    return {'y': np.diag([-1., 1, -1]), 'x': np.diag([1., -1, -1])}[axis]

def sample(img, lens, M, k, stream_scale, crop, flip):
    rays = D @ M.T
    rays = rays * np.array([flip, 1, 1])
    px, valid = omni_project(rays, lens)
    lx = (px[:, 0] - k * lens['size'][1] - crop) * stream_scale * S
    ly = (px[:, 1] - crop) * stream_scale * S
    v = cv2.remap(img, lx.reshape(H, W).astype(np.float32), ly.reshape(H, W).astype(np.float32), cv2.INTER_LINEAR, borderValue=-1)
    ok = valid.reshape(H, W) & (v >= 0)
    return v, ok

res = []
for order, inv, faxis, swap, flip1, crop in itertools.product(['xyz','xzy','yxz','yzx','zxy','zyx'], [0,1], ['y','x'], [0,1], [1,-1], [32, 0]):
    Ms = []
    for k, L in enumerate(lenses):
        R = euler(*L['rot'], order)
        if inv: R = R.T
        Ms.append(R)
    R0 = Ms[0]
    M = [np.eye(3), Ms[1] @ R0.T @ F(faxis) if False else Ms[1] @ F(faxis) @ R0.T]
    # 월드 = 렌즈0 카메라 좌표 (렌즈0 회전 상쇄), 렌즈1 = 상대 회전 + 180도 반전
    M = [np.eye(3), Ms[1] @ R0.T]
    M[1] = M[1] @ F(faxis)
    src = [imgs[1], imgs[0]] if swap else imgs
    v0, o0 = sample(src[0], lenses[0], M[0], 0, 0.5, crop, 1)
    v1, o1 = sample(src[1], lenses[1], M[1], 1, 0.5, crop, flip1)
    both = o0 & o1
    if both.sum() < 1000: continue
    mae = float(np.abs(v0[both] - v1[both]).mean())
    res.append((mae, int(both.sum()), order, inv, faxis, swap, flip1, crop))
res.sort()
for r in res[:12]: print(r)
print('worst', res[-1])
