"""M23: 원본으로 바꿔 끼우기 — 덤프 파싱·키 보간·화각·계획, 생성 ExtendScript 문법(이스케이프) 점검.
실렌더 검증(얼굴 위치·선명도)은 tests/fixtures/swap_measured.json(원격 실측 기록)으로 확인."""
import json, os, subprocess

import numpy as np
import pytest

from multicam360.swap import dump_script, parse_dump, place_script, plan, view_fn
from multicam360.vrproj import premiere_matrix, to_vr, engine_matrix

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "swap_measured.json")


def test_parse_and_interp():
    d = parse_dump(json.dumps('{"clips":[{"pan":[[0,10],[2,30]],"tilt":[[null,5]],"roll":[[null,0]]}]}'))
    V = view_fn(d["clips"][0])
    assert np.allclose(V(1.0), premiere_matrix(20, 5, 0)) and np.allclose(V(5.0), premiere_matrix(30, 5, 0))


def test_roundtrip_engine_direction():
    """vrkeys로 쓴 키(엔진 방향 → 팬·기울기·롤)를 swap이 읽으면 원래 엔진 시선이 나온다."""
    p, t, r = to_vr(95.5, 20)
    V = view_fn({"pan": [[None, p]], "tilt": [[None, t]], "roll": [[None, r]]})
    assert np.abs(V(0) - engine_matrix(95.5, 20)).max() < 1e-9


def test_plan_hfov(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"src": "/r/work/x/source.mkv"}))
    dump = {"clips": [{"track": 1, "index": 2, "name": "p", "media": str(tmp_path / "panorama.mp4"), "start": 3.0, "end": 5.0,
                       "in_s": 10.0, "ticks_start": "1", "pan": [[None, 0]], "tilt": [[None, 0]], "roll": [[None, 0]],
                       "scale_h": 457.143, "scale_w": 514.286, "uniform": False, "position": [0.5, 0.5]}]}
    it = plan(dump)[0]
    assert abs(it["hfov"] - 70) < 0.01 and it["imu"] == "/r/work/x/out/imu.npz" and it["meta"].endswith(".trailer")
    assert it["dur"] == 2.0 and it["in_s"] == 10.0


@pytest.mark.skipif(subprocess.run(["which", "node"], capture_output=True).returncode != 0, reason="node 없음")
def test_scripts_parse_as_js(tmp_path):
    """생성 ExtendScript가 자바스크립트 문법으로 읽힌다(최상위 return 허용 위해 함수로 감싸서)."""
    for js in (dump_script('시퀀스 "따옴표"'), place_script([{"file": "/a/swap/b.mp4", "ticks_start": "0", "track": 0,
                                                              "index": 0, "name": "x"}], "s")):
        f = tmp_path / "s.js"
        f.write_text("(function(){" + js + "\n})")
        assert subprocess.run(["node", "--check", str(f)], capture_output=True).returncode == 0, js[:200]


@pytest.mark.skipif(not os.path.exists(FIX), reason="실렌더 기록 없음")
def test_measured_render():
    m = json.load(open(FIX))
    assert max(abs(x) for x in m["center_diff_pct"]) <= 3.0 and m["sharpness_ratio"] > 1.0
