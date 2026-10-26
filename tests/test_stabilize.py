"""M13: 자이로 안정화 — 차 안 테스트 클립(고속도로 직진), 카메라를 손으로 크게 흔든 50~53초.
직진 주행이라 차 실내 ≈ 지구 기준 → 회전 안정화만으로 실내가 고정돼야 한다.
(엘리베이터 006은 벽 1m 근거리 시차로 회전 보정 한계 39% — LEDGER 참조)
픽스처: re/out016.trailer, re/out016/ (VID_20261004_173903_00_016 50~53초, 10fps, 1920x960 렌즈쌍) — 로컬 전용."""
import glob, os, sys

import cv2
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from orbcut.engine import Projector, equirect_dirs, load_lenses
from orbcut.imu import read_imu
from orbcut.stabilize import IMU_TO_CAM, orientations, tilt_deg

TR = os.path.join(REPO, "re", "out016.trailer")
FR = sorted(glob.glob(os.path.join(REPO, "re", "out016", "f_*.png")))
T0 = 50.0
pytestmark = pytest.mark.skipif(not (os.path.exists(TR) and FR), reason="움직임 픽스처 없음")
W = 1024


def _pano(fr, R, lenses):
    dirs = (equirect_dirs(W).reshape(-1, 3) @ R).reshape(W // 2, W, 3)
    return cv2.cvtColor(Projector(lenses, dirs, stream_w=fr.shape[0], flow=False)(fr), cv2.COLOR_BGR2GRAY)


def _drift(imgs):
    """첫 프레임 대비 고정 장면의 위치 변화(px, 중앙값): ORB 특징점 매칭, 경도 순환 고려, 중위도 띠만."""
    orb = cv2.ORB_create(1500)
    h = imgs[0].shape[0]
    mask = np.zeros_like(imgs[0]); mask[h // 5: h * 4 // 5] = 255
    k0, d0 = orb.detectAndCompute(imgs[0], mask)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    out = []
    for im in imgs[1:]:
        k1, d1 = orb.detectAndCompute(im, mask)
        if d0 is None or d1 is None:
            out.append(np.nan); continue
        ms = bf.match(d0, d1)
        if len(ms) < 8:
            out.append(np.nan); continue
        dx = np.array([k1[m.trainIdx].pt[0] - k0[m.queryIdx].pt[0] for m in ms])
        dx = (dx + W / 2) % W - W / 2                     # 경도 순환
        dy = np.array([k1[m.trainIdx].pt[1] - k0[m.queryIdx].pt[1] for m in ms])
        out.append(float(np.hypot(np.median(dx), np.median(dy))))
    return float(np.nanmean(out))


@pytest.fixture(scope="module")
def data():
    lenses = load_lenses(TR)
    imu = read_imu(open(TR, "rb").read())
    frames = [cv2.imread(f) for f in FR]
    times = T0 + np.arange(len(frames)) / 10
    return lenses, imu, frames, times


def test_stabilized_scene_stays_put(data):
    lenses, imu, frames, times = data
    Rs = orientations(imu, times)
    raw = [_pano(f, np.eye(3), lenses) for f in frames]
    stab = [_pano(f, R, lenses) for f, R in zip(frames, Rs)]
    d_raw, d_stab = _drift(raw), _drift(stab)
    print(f"drift raw {d_raw:.1f}px → stab {d_stab:.1f}px ({d_stab / d_raw:.0%})")
    assert d_stab <= 0.2 * d_raw, (d_raw, d_stab)


def _vertical_tilts(img):
    """화면 안 거의 수직인 직선들의 기울기(도, 0=수직)."""
    import math
    e = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 60, 160)
    ls = cv2.HoughLinesP(e, 1, np.pi / 360, 60, minLineLength=90, maxLineGap=6)
    if ls is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(ls).reshape(-1, 4):
        a = (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 90) % 180 - 90
        if abs(a) < 25:
            out.append(abs(a))
    return out


EL_TR = os.path.join(REPO, "re", "move006.trailer")
EL_FR = sorted(glob.glob(os.path.join(REPO, "re", "move006_frames", "f_*.png")))


@pytest.mark.skipif(not (os.path.exists(EL_TR) and EL_FR), reason="엘리베이터 픽스처 없음")
def test_horizon_level_vertical_lines():
    """수평: 엘리베이터(문 모서리 = 진짜 수직선)를 크게 흔든 0.5~3.5초. 안정화 화면의 수직선 기울기 중앙값 ≤ 1°.
    차 안은 A필러 등이 원래 비스듬해 기준이 될 수 없음 — 가속도 기반 측정은 손 가속이 섞여 폐기 (LEDGER)."""
    from orbcut.engine import DynamicProjector, flat_dirs
    lenses = load_lenses(EL_TR)
    imu = read_imu(open(EL_TR, "rb").read())
    frames = [cv2.imread(f) for f in EL_FR]
    times = 0.5 + np.arange(len(frames)) / 10
    Rs = orientations(imu, times)
    views = [DynamicProjector(lenses, flat_dirs(y, 0, 90, 640, 480), stream_w=frames[0].shape[0]) for y in (0, 90, 180, -90)]
    raw, stab = [], []
    for f, R in zip(frames, Rs):
        for v in views:
            raw += _vertical_tilts(cv2.cvtColor(v(f, np.eye(3)), cv2.COLOR_BGR2GRAY))
            stab += _vertical_tilts(cv2.cvtColor(v(f, R.T), cv2.COLOR_BGR2GRAY))
    print(f"vertical tilt raw {np.median(raw):.2f}° → stab {np.median(stab):.2f}° (n={len(stab)})")
    assert len(stab) > 100 and np.median(stab) <= 1.0
