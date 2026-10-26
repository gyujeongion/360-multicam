import math, os, subprocess, sys, tempfile

import cv2
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from orbcut.engine import Projector, dir_to_yaw_pitch, equirect_dirs, flat_dirs, load_lenses
from orbcut.reframe import rect_vfov

L0, L1, META = (os.path.join(REPO, "re", f) for f in ("l0.png", "l1.png", "tail8m.bin"))
need_fixture = pytest.mark.skipif(not all(map(os.path.exists, (L0, L1, META))), reason="렌즈 스틸 픽스처 없음")


def test_flat_center_direction():
    d = flat_dirs(30, 10, 70, 64, 36)
    c = (d[17, 31] + d[18, 32]) / 2
    yaw, pitch = dir_to_yaw_pitch(c)
    assert abs(yaw - 30) < 0.5 and abs(pitch - 10) < 0.5


def test_flat_vertical_fov_exact():
    w, h, hfov = 1600, 900, 70
    d = flat_dirs(0, 0, hfov, w, h)
    top = math.degrees(-math.asin(d[0, w // 2, 1]))
    assert abs(2 * top - rect_vfov(hfov, w, h)) < 0.2   # hfov*9/16(=39.4)이 아니라 43.0


def test_equirect_yaw_column():
    W = 360
    d = equirect_dirs(W)
    yaw, pitch = dir_to_yaw_pitch(d[90, 270])            # 270열 = +90도
    assert abs(yaw - 90.5) < 1.1 and abs(pitch) < 1.1


@need_fixture
def test_matches_v360_convention():
    """엔진 flat 화면 == 엔진 equirect을 ffmpeg v360(yaw/pitch)으로 편 화면 → 각도 규약 일치 (people.json 호환)."""
    lenses = load_lenses(META)
    pair = np.hstack([cv2.imread(L0), cv2.imread(L1)])
    eq = Projector(lenses, equirect_dirs(3840), flow=False)(pair)
    yaw, pitch, hfov, w, h = -53.4, 8, 70, 640, 360
    mine = Projector(lenses, flat_dirs(yaw, pitch, hfov, w, h), flow=False)(pair)
    with tempfile.TemporaryDirectory() as t:
        cv2.imwrite(f"{t}/eq.png", eq)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{t}/eq.png", "-vf",
                        f"v360=input=e:output=flat:yaw={yaw}:pitch={pitch}:h_fov={hfov}:v_fov={rect_vfov(hfov, w, h)}"
                        f":w={w}:h={h}:interp=lanczos", f"{t}/ref.png"], check=True)
        ref = cv2.imread(f"{t}/ref.png")
    g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    mae = float(np.abs(g(mine) - g(ref)).mean())
    shifted = float(np.abs(g(mine)[:, 20:] - g(ref)[:, :-20]).mean())   # 대조군: 20px 어긋나면 커져야 함
    assert mae < 8, mae
    assert mae < shifted * 0.6, (mae, shifted)


@need_fixture
def test_seam_flow_reduces_ghost():
    """겹침띠에서 플로우 적용 시 두 렌즈 정렬 오차가 줄어든다 (근거리 이음새)."""
    lenses = load_lenses(META)
    pair = np.hstack([cv2.imread(L0), cv2.imread(L1)])
    pj = Projector(lenses, equirect_dirs(3840), flow=True)
    flows = [r for r in pj.regions if r[4]]
    assert flows, "플로우 영역 없음"
    out_flow = pj(pair)
    pj.flow_on = False
    out_plain = pj(pair)
    # 플로우가 실제로 화면을 바꾸는지: 플로우 영역 안은 바뀌고, 영역 밖은 그대로
    d = np.abs(out_flow.astype(int) - out_plain.astype(int)).sum(-1)
    inside = np.zeros(d.shape, bool)
    for y0, y1, x0, x1, _ in flows:
        inside[y0:y1, x0:x1] = True
    assert d[inside].mean() > 2.0, d[inside].mean()
    assert d[~inside].max() == 0


def test_seam_strip_cols_wrap():
    """보이는 θ가 ±180°를 걸쳐도 덮는 열이 연속(원형) 한 덩어리로 나온다."""
    from orbcut.engine import SEAM_MARGIN, SEAM_PPD, SeamStrip
    st = SeamStrip.__new__(SeamStrip)
    st.W = 360 * SEAM_PPD
    cols = st.cols_for(np.array([178.0, 179.5, -179.5, -178.0]))
    assert len(cols) == 4 * SEAM_PPD + 1 + 2 * SEAM_MARGIN, len(cols)
    assert (np.diff(cols) % st.W == 1).all()


@need_fixture
def test_dynamic_seam_flow_aligns():
    """동적 렌더(band2): 이음새 띠 플로우로 두 렌즈 정렬 오차가 줄고, 같은 뷰의 정적 렌더(band2)와 화면이 맞는다."""
    from orbcut.engine import DynamicProjector
    lenses = load_lenses(META)
    pair = np.hstack([cv2.imread(L0), cv2.imread(L1)])
    view = flat_dirs(90, 0, 70, 640, 360)                 # 이음새(yaw ±90°)를 가운데에
    pj = DynamicProjector(lenses, view)
    xs, ys, a0, _ = pj.maps(np.eye(3))                     # 변위 없음
    xf, yf, a1, conf = pj.maps(np.eye(3), pair)            # 띠 플로우 변위
    assert conf is not None and np.allclose(a0, a1)
    g = lambda x, y: cv2.cvtColor(cv2.remap(pair, x, y, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mid = (a0 > 0.3) & (a0 < 0.7)
    assert mid.sum() > 1000
    r_plain = float(np.abs(g(xs[0], ys[0]) - g(xs[1], ys[1]))[mid].mean())
    r_flow = float(np.abs(g(xf[0], yf[0]) - g(xf[1], yf[1]))[mid].mean())
    assert r_flow < r_plain * 0.85, (r_flow, r_plain)
    dyn = DynamicProjector(lenses, view)(pair, np.eye(3))
    st = Projector(lenses, view, flow=True)(pair)
    mae = float(np.abs(cv2.GaussianBlur(dyn, (5, 5), 0).astype(int) - cv2.GaussianBlur(st, (5, 5), 0).astype(int)).mean())
    assert mae < 4, mae
