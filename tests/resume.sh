#!/bin/bash
# M3: 재실행 시 캐시 건너뜀 / 파라미터 바뀐 단계만 재실행 / 실패 단계·사유 기록
set -eo pipefail
cd "$(dirname "$0")/.."
if [ "$1" = "--remote" ]; then
  exec .venv/bin/python -c 'import sys; sys.path.insert(0, "tests"); from _remote import delegate; rc = delegate("bash tests/resume.sh"); sys.exit(2 if rc is None else rc)'
fi
PY=.venv/bin/python; SRC=work/src_240_180.mkv; META=re/tail8m.bin; OUT=work/resume
[ -f "$SRC" ] || { echo "픽스처 없음"; exit 1; }
# 준비 실행: 항상 한 번 — 코드·기본값이 바뀐 뒤면 바뀐 단계만 여기서 다시 돈다(이후 측정은 순수 캐시 재실행)
$PY -m orbcut --json all "$SRC" --out "$OUT" --range 60:66 --meta "$META" | tail -1
# 1) 같은 명령 재실행 → 전 단계 cached, 빠르게 종료
t0=$(date +%s)
$PY -m orbcut --json all "$SRC" --out "$OUT" --range 60:66 --meta "$META" | tail -1
t1=$(date +%s)
$PY - "$OUT" $((t1 - t0)) <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json")); el = int(sys.argv[2])
assert m["status"] == "ok"
notc = [k for k, v in m["stages"].items() if not v.get("cached")]
assert not notc, f"캐시 안 된 단계: {notc}"
assert el <= 15, f"재실행 {el}s"
print(f"1) 전 단계 cached, {el}s")
PY
# 2) xml 파라미터만 변경 → xml만 재실행
$PY -m orbcut --json all "$SRC" --out "$OUT" --range 60:66 --meta "$META" --set xml.face_ratio=0.29 | tail -1
$PY - "$OUT" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))
ran = sorted(k for k, v in m["stages"].items() if not v.get("cached"))
assert ran == ["xml"], ran
assert m["params"]["xml"]["face_ratio"] == 0.29
print("2) xml만 재실행")
PY
# 원상 복구(다음 실행 대비)
$PY -m orbcut --json all "$SRC" --out "$OUT" --range 60:66 --meta "$META" > /dev/null
# 3) 실패 기록
BAD=work/resume_bad; rm -rf "$BAD"
set +e; $PY -m orbcut --json all work/없는파일.mkv --out "$BAD" > /dev/null; rc=$?; set -e
$PY - "$BAD" $rc <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))
assert int(sys.argv[2]) != 0 and m["status"] == "failed" and m["error"]
failed = [k for k, v in m["stages"].items() if v.get("status") == "failed"]
assert failed == ["probe"], failed
print("3) 실패 단계·사유 기록:", failed, m["error"][:80])
PY
echo RESUME OK
