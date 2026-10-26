import json, os, struct, subprocess, sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from orbcut.calib import TRAILER_MAGIC
from orbcut.pipeline import parse_range
from orbcut.probe import ProbeError, probe

META = os.path.join(REPO, "re", "tail8m.bin")


def cli(*a):
    r = subprocess.run([sys.executable, "-m", "orbcut", "--json", *a], cwd=REPO, capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout.strip().splitlines()[-1])


def fake_insv(tmp_path, fields):
    """ExtraMetadata(0x101) 레코드만 든 최소 트레일러 파일."""
    pb = b""
    for fn, val in fields.items():
        v = val.encode()
        tag, L = (fn << 3) | 2, len(v)
        enc = lambda n: bytes([(n & 0x7F) | (0x80 if n > 0x7F else 0)]) + (enc(n >> 7) if n > 0x7F else b"")
        pb += enc(tag) + enc(L) + v
    body = pb + struct.pack("<HI", 0x101, len(pb))
    tail = b"\0" * 32 + struct.pack("<II", len(body) + 72, 3) + TRAILER_MAGIC
    p = tmp_path / "fake.insv"
    p.write_bytes(b"\0" * 100 + body + tail)
    return str(p)


@pytest.mark.skipif(not os.path.exists(META), reason="트레일러 픽스처 없음")
def test_x6_trailer():
    info = probe(META)
    assert info["camera"] == "Insta360 X6" and info["supported"] and info["offset_used"] == "offset_v6"
    assert info["lenses"] == 2 and abs(info["duration_s"] - 947.3) < 0.1


def test_unsupported_offset_version(tmp_path):
    f = fake_insv(tmp_path, {2: "Insta360 X4", 54: "2_1_2_3"})
    info = probe(f)
    assert info["camera"] == "Insta360 X4" and not info["supported"] and info["offsets_present"] == ["offset_v3"]
    rc, out = cli("probe", f)
    assert rc == 3 and out["supported"] is False


def test_no_trailer(tmp_path):
    f = tmp_path / "cut.mkv"; f.write_bytes(b"\0" * 4096)
    with pytest.raises(ProbeError) as e:
        probe(str(f))
    assert e.value.code == "no_trailer"
    rc, out = cli("probe", str(f))
    assert rc == 4 and out["error_code"] == "no_trailer"


def test_not_found():
    rc, out = cli("probe", "/없는/파일.insv")
    assert rc == 4 and out["error_code"] == "not_found"


def test_range_forms():
    assert parse_range("4-00:7-00") == (240, 180)
    assert parse_range("240:420") == (240, 180)
    assert parse_range("1-02-03:1-02-13") == (3723, 10)   # 시-분-초
    assert parse_range(None) == (None, None)


def test_nas_probe():
    from orbcut.config import nas_host, test_nas_src
    NAS = test_nas_src()
    if not NAS:
        pytest.skip("ORBCUT_TEST_NAS_SRC 미설정")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", nas_host(), "true"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("NAS 접속 불가")
    rc, out = cli("probe", NAS)
    assert rc == 0 and out["camera"] == "Insta360 X6" and out["source"] == NAS
