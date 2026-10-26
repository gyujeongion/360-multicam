"""M17: 엔진 방향 → 프리미어 VR 투영 변환식. 실측값(tests/fixtures/vrproj_measured.json) 대비 각도 오차 ≤2°."""
import json, math, os

import numpy as np
import pytest

from orbcut.vrproj import engine_matrix, motion_scale, path_keys, premiere_matrix, to_vr

MEAS = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "vrproj_measured.json")))


def angle_deg(A, B):
    """두 회전 사이 각(도)."""
    c = (np.trace(A.T @ B) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@pytest.mark.parametrize("yaw,pitch", [(y, p) for y in range(-180, 181, 30) for p in (-60, -20, 0, 15, 45)])
def test_inverse_exact(yaw, pitch):
    assert np.abs(premiere_matrix(*to_vr(yaw, pitch)) - engine_matrix(yaw, pitch)).max() < 1e-9


def test_measured_targets_within_2deg():
    for c in MEAS["engine_targets"]:
        pan, tilt, roll = to_vr(c["yaw"], c["pitch"])
        assert np.allclose([pan, tilt, roll], c["premiere"], atol=0.01)        # 실측에 넣은 값 = 지금 식의 값
        err = angle_deg(engine_matrix(c["fit_yaw"], c["fit_pitch"]), engine_matrix(c["yaw"], c["pitch"]))
        assert err <= 2.0 and abs(c["fit_roll"]) <= 2.0


def test_measured_pan_only_model():
    for c in MEAS["pan_only"]:
        model = premiere_matrix(c["pan"], c["tilt"], c["roll"])
        assert angle_deg(model, engine_matrix(c["fit_yaw"], c["fit_pitch"])) <= 2.0


def test_motion_scale_matches_measured_fov():
    sh, sw = motion_scale(90)
    assert abs(sw - 400) < 1e-6 and abs(sh - 18000 / 50.625) < 1e-6


def test_path_keys_wrap_and_shape():
    rows = [[t, 170 + 4 * t, 0, True, True] for t in np.arange(0, 5, 0.25)]     # 170° → 186° (±180 넘김)
    keys = path_keys(rows, step=0.5, smooth_s=0.2)
    pans = [k[1] for k in keys]
    assert all(-180 <= p <= 180 for p in pans)
    jumps = [i for i in range(1, len(keys)) if abs(keys[i][1] - keys[i - 1][1]) > 180]
    assert len(jumps) == 1 and keys[jumps[0]][0] - keys[jumps[0] - 1][0] < 0.05   # 넘어가는 순간은 1프레임 안


def test_face_centered_measured():
    k = MEAS["keyframes_face_center"]
    assert max(map(abs, k["dx_pct"])) <= 5 and max(map(abs, k["dy_pct"])) <= 8


def test_vrkeys_cli(tmp_path):
    from orbcut.__main__ import main
    rows = [[t, 30 + t, -5, True, True] for t in np.arange(0, 3, 0.25)]
    (tmp_path / "paths.json").write_text(json.dumps({"paths": {"A": rows}}))
    assert main(["vrkeys", str(tmp_path), "A", "--hfov", "60"]) == 0
    js = (tmp_path / "vrkeys_A.jsx").read_text()
    assert "VR \\ud22c\\uc601" in js and "setValueAtKey" in js
    first = json.loads(js.split("var keys = ")[1].split(";")[0])[0]
    assert len(first) == 4 and np.allclose(first[1:], to_vr(30, -5), atol=0.6)     # 평활 오차 감안
