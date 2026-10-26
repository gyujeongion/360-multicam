"""insv(또는 트레일러 끝부분 파일)에서 기종·펌웨어·보정값 버전·길이를 읽는다."""
import json, os, shlex, subprocess, sys, tempfile

from .calib import parse_offset_v6, read_metadata_bytes
from .config import nas_host

# ExtraMetadata 필드 번호 (Studio 내장 스키마 기준, re/README.md E3)
FIELDS = {1: "serial_number", 2: "camera_type", 3: "fw_version", 10: "total_time", 93: "total_time_ms",
          5: "offset", 53: "offset_v2", 54: "offset_v3", 111: "offset_v6", 197: "offset_v8"}
SUPPORTED = ("offset_v6",)


class ProbeError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def _scan(buf):
    out, i = {}, 0
    def varint():
        nonlocal i
        r = s = 0
        while True:
            c = buf[i]; i += 1
            r |= (c & 0x7F) << s; s += 7
            if c < 0x80:
                return r
    while i < len(buf):
        tag = varint(); fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v = varint()
            if fn in FIELDS: out[FIELDS[fn]] = v
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
        elif wt == 2:
            L = varint(); v = buf[i:i + L]; i += L
            if fn in FIELDS: out[FIELDS[fn]] = v.decode("utf-8", "replace")
        else:
            raise ProbeError("bad_metadata", f"알 수 없는 wire type {wt}")
    return out


def fetch_nas_trailer(nas_path):
    """nas:/<경로> → 트레일러가 든 끝 8MB만 임시 파일로 (원본 전체 전송 없음)."""
    fd, tmp = tempfile.mkstemp(suffix=".trailer"); os.close(fd)
    with open(tmp, "wb") as f:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", nas_host(), f"tail -c 8388608 {shlex.quote(nas_path)}"],
                           stdout=f, stderr=subprocess.PIPE)
    if r.returncode != 0:
        os.unlink(tmp)
        raise ProbeError("nas_unreachable", f"NAS 읽기 실패: {r.stderr.decode(errors='replace').strip()[:200]}")
    return tmp


def probe(path):
    if path.startswith("nas:"):
        tmp = fetch_nas_trailer(path[4:])
        try:
            return {**probe(tmp), "source": path}
        finally:
            os.unlink(tmp)
    if not os.path.exists(path):
        raise ProbeError("not_found", f"파일 없음: {path}")
    try:
        raw = read_metadata_bytes(path)
    except AssertionError:
        raise ProbeError("no_trailer", "Insta360 트레일러 없음 — 원본 insv 또는 --meta로 트레일러 파일을 지정")
    f = _scan(raw)
    present = [k for k in ("offset_v8", "offset_v6", "offset_v3", "offset_v2", "offset") if f.get(k)]
    use = next((k for k in SUPPORTED if k in present), None)
    info = {"camera": f.get("camera_type"), "firmware": f.get("fw_version"), "serial": f.get("serial_number"),
            "duration_s": (f.get("total_time_ms") or 0) / 1000 or f.get("total_time"),
            "offsets_present": present, "offset_used": use, "supported": use is not None}
    if use:
        info["lenses"] = len(parse_offset_v6(f[use]))
    return info


def main(argv=None):
    argv = argv or sys.argv[1:]
    try:
        info = probe(argv[0])
    except ProbeError as e:
        print(json.dumps({"supported": False, "error_code": e.code, "error": str(e)}, ensure_ascii=False))
        return 4
    if len(argv) > 1:
        json.dump(info, open(argv[1], "w"), indent=2, ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False))
    if not info["supported"]:
        print(f"미지원 보정값 형식: {info['offsets_present']} (지원: {SUPPORTED})", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
