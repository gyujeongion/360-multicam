import json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbcut.eval import evaluate

LAB = {"people": ["A", "B", "C"], "duration": 30.0, "segments": [
    {"start": 0, "end": 10, "speakers": ["A"]},
    {"start": 12, "end": 20, "speakers": ["B"]},
    {"start": 20, "end": 25, "speakers": ["B", "C"]},   # 겹침
    {"start": 25, "end": 30, "speakers": ["C"]}]}


def test_perfect():
    edit = {"segments": [{"start": 0, "end": 12, "cam": "A"}, {"start": 12, "end": 25, "cam": "B"},
                         {"start": 25, "end": 30, "cam": "C"}]}
    r = evaluate(edit, LAB)
    assert r["frame_acc"] == 1.0
    assert r["wrong_s"] == 0
    assert r["silence_s"] == 2.0
    assert r["n_gt_changes"] == 1          # A→B만 전환. B→B,C→C는 계속 말하는 사람이 있어 전환 아님
    assert r["cut_recall"] == 1.0          # 12.0 컷 ↔ 정답 12.0
    assert r["cut_precision"] == 0.5       # 25.0 컷은 정답 전환이 아님(허용되는 선택이지만 정밀도엔 반영)
    assert r["per_person"]["A"]["shown_while_talking"] == 1.0


def test_wrong_person():
    edit = {"segments": [{"start": 0, "end": 30, "cam": "A"}]}
    r = evaluate(edit, LAB)
    assert abs(r["frame_acc"] - 10 / 28) < 0.01
    assert abs(r["wrong_s"] - 18) < 0.2
    assert r["cut_precision"] is None


def test_cut_timing_tolerance():
    edit = {"segments": [{"start": 0, "end": 12.5, "cam": "A"}, {"start": 12.5, "end": 30, "cam": "B"}]}
    r = evaluate(edit, LAB, tol=0.75)
    assert r["cut_precision"] == 1.0       # 12.5 ↔ 정답 12.0
    r2 = evaluate(edit, LAB, tol=0.25)
    assert r2["cut_precision"] == 0.0


def test_cli(tmp_path):
    e, l = tmp_path / "e.json", tmp_path / "l.json"
    e.write_text(json.dumps({"segments": [{"start": 0, "end": 30, "cam": "B"}]}))
    l.write_text(json.dumps(LAB))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-m", "orbcut", "--json", "eval", str(e), str(l)], cwd=repo,
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out.strip().splitlines()[-1])["frame_acc"] > 0
