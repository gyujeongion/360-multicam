#!/bin/bash
# M1: 6초 구간을 전체 파이프라인으로 관통 → manifest status=ok + 핵심 산출물 존재
set -eo pipefail
cd "$(dirname "$0")/.."
if [ "$1" = "--remote" ]; then   # 원격 호스트(config remote_host)에서 검증
  exec .venv/bin/python -c 'import sys; sys.path.insert(0, "tests"); from _remote import delegate; rc = delegate("bash tests/smoke.sh"); sys.exit(2 if rc is None else rc)'
fi
SRC=work/src_240_180.mkv; META=re/tail8m.bin; OUT=work/smoke
[ -f "$SRC" ] && [ -f "$META" ] || { echo "픽스처 없음: $SRC / $META"; exit 1; }
rm -rf "$OUT"
.venv/bin/python -m orbcut --json all "$SRC" --out "$OUT" --range 30:36 --meta "$META" | tail -1
.venv/bin/python - "$OUT" <<'PY'
import json, os, sys, subprocess
out = sys.argv[1]
m = json.load(open(f"{out}/manifest.json"))
assert m["status"] == "ok", m.get("error")
for k in ("premiere_xml", "shots", "preview"):
    assert os.path.getsize(m["outputs"][k]) > 0, k
d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", m["outputs"]["preview"]]))
src_d = m['stages']['slice'].get('actual_duration_s') or 10
assert abs(d - src_d) < 0.5, (d, src_d)  # 무손실 자르기는 키프레임 단위 → 실제 길이 기준
subprocess.run(["xmllint", "--noout", m["outputs"]["premiere_xml"]], check=True)
print("SMOKE OK", {k: v for k, v in m["stages"].items() if False} or "", f"preview {d:.1f}s")
PY
