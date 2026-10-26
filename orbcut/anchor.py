"""장면 기준 고정(차 실내 등): 자이로 안정화 위에서 '화면 대부분을 차지하는 배경'의 느린 회전을 영상으로 추정.

자이로는 지구(관성) 기준이라 차가 회전하면 실내가 화면에서 흘러간다. 여기서는
  자이로 안정화 파노라마 → ORB 특징점(단위벡터) → 기준 키프레임과 회전 RANSAC(Kabsch)
으로 배경 회전 W(t)를 구해 부드럽게 다듬고, 최종 카메라→장면 회전 R_a = W·R_g 를 만든다.
빠른 흔들림은 자이로가, 느린 기준(차 회전·드리프트·수평)은 영상이 맡는 상보 구조.
"""
import math

import cv2
import numpy as np

from .engine import DynamicProjector, equirect_dirs

PANO_W = 1024
INLIER_DEG = 1.0
MIN_INLIERS = 40


def _px_to_dir(pts, W):
    H = W // 2
    lon = (pts[:, 0] + 0.5) / W * 2 * np.pi - np.pi
    lat = (pts[:, 1] + 0.5) / H * np.pi - np.pi / 2
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def kabsch(a, b):
    """a(N,3) → b(N,3) 최소제곱 회전."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def robust_rotation(a, b, iters=300, rng=np.random.default_rng(0)):
    best, best_in = np.eye(3), np.zeros(len(a), bool)
    thr = math.cos(math.radians(INLIER_DEG))
    for _ in range(iters):
        idx = rng.choice(len(a), 3, replace=False)
        R = kabsch(a[idx], b[idx])
        inl = np.einsum("ij,ij->i", a @ R.T, b) > thr
        if inl.sum() > best_in.sum():
            best, best_in = R, inl
    if best_in.sum() >= 3:
        best = kabsch(a[best_in], b[best_in])
    return best, best_in


def rotvec(R):
    v, _ = cv2.Rodrigues(R)
    return v.ravel()


def from_rotvec(v):
    R, _ = cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))
    return R


class SceneAnchor:
    """samples: [(t, pair_frame)] 저해상도 렌즈쌍. rot_gyro(t)→R_g. 결과 W(t) 보간 함수."""

    def __init__(self, lenses, stream_w):
        self.pj = DynamicProjector(lenses, equirect_dirs(PANO_W), stream_w=stream_w, seam="feather")
        self.orb = cv2.ORB_create(2500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        H = PANO_W // 2
        self.mask = np.zeros((H, PANO_W), np.uint8); self.mask[H // 6: H * 5 // 6] = 255   # 극 제외

    def stabilized(self, pair, R_g):
        return cv2.cvtColor(self.pj(pair, R_g.T), cv2.COLOR_BGR2GRAY)

    def estimate(self, samples, rot_gyro, smooth_s=0.6):
        ts, Ws, info = [], [], []
        key = None            # (dirs, desc, W_key: 키프레임 → 기준)
        for t, pair in samples:
            img = self.stabilized(pair, rot_gyro(t))
            kp, des = self.orb.detectAndCompute(img, self.mask)
            if des is None or len(kp) < MIN_INLIERS:
                continue
            dirs = _px_to_dir(np.array([k.pt for k in kp]), PANO_W)
            if key is None:
                key = (dirs, des, np.eye(3)); ts.append(t); Ws.append(np.eye(3)); info.append(len(kp)); continue
            ms = self.bf.match(des, key[1])
            if len(ms) < MIN_INLIERS:
                key = (dirs, des, Ws[-1]); continue                    # 장면 급변: 직전 추정을 이어 새 키프레임
            a = dirs[[m.queryIdx for m in ms]]; b = key[0][[m.trainIdx for m in ms]]
            R, inl = robust_rotation(a, b)                            # 현재 → 키프레임
            if inl.sum() < MIN_INLIERS:
                key = (dirs, des, Ws[-1]); continue
            W = key[2] @ R                                            # 현재 → 기준
            ts.append(t); Ws.append(W); info.append(int(inl.sum()))
            if inl.sum() < 2 * MIN_INLIERS:                           # 매칭이 줄면 키프레임 갱신
                key = (dirs, des, W)
        self.ts = np.array(ts)
        V = np.array([rotvec(W) for W in Ws])
        # 회전벡터 평활 (가우시안, smooth_s 초) — 느린 기준만 남김
        if len(V) > 3:
            dt = np.median(np.diff(self.ts))
            k = max(1, int(round(smooth_s / dt)))
            g = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2); g /= g.sum()
            Vp = np.pad(V, ((3 * k, 3 * k), (0, 0)), mode="edge")
            V = np.stack([np.convolve(Vp[:, i], g, "valid") for i in range(3)], 1)
        self.V, self.inliers = V, info
        return self

    def W(self, t):
        if len(self.ts) == 0:
            return np.eye(3)
        return from_rotvec([np.interp(t, self.ts, self.V[:, i]) for i in range(3)])
