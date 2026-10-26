"""M5: 4:00~7:00 테이블 모드 결과물 검증 — 인물별 4K 마스터, 샷별 Scale/Position(≤100%, 크롭이 마스터 안), XML, 프리뷰.
usage: python tests/check_v5.py [--remote]   (--remote: config의 원격 호스트에서 실행·검증)"""
import json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "work/v5"

if "--remote" in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _remote import delegate
    rc = delegate(".venv/bin/python tests/check_v5.py")
    if rc is None:
        print("원격 호스트 미설정(MULTICAM360_REMOTE_HOST) — 로컬에서 실행하려면 --remote 없이"); sys.exit(2)
    sys.exit(rc)

os.chdir(REPO)
if not (os.path.exists(os.path.join(OUT, "manifest.json")) and json.load(open(os.path.join(OUT, "manifest.json"))).get("status") == "ok"):
    r = subprocess.run([sys.executable, "-m", "multicam360", "--json", "all", "work/src_240_180.mkv", "--out", OUT,
                        "--meta", "re/tail8m.bin"], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-800:])

fails = []
m = json.load(open(os.path.join(OUT, "manifest.json")))
if m.get("status") != "ok":
    print("manifest 실패:", m.get("error")); sys.exit(1)
dur = lambda p: float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                               "-of", "csv=p=0", p]).decode())
people = json.load(open(os.path.join(OUT, "people.json")))["people"]
mr = json.load(open(os.path.join(OUT, "masters", "render.json")))
if len(people) < 2: fails.append(f"인물 {len(people)}명")
if (mr["w"], mr["h"]) != (3840, 2160): fails.append(f"마스터 해상도 {mr['w']}x{mr['h']}")
if mr.get("proj") != "pannini": fails.append("마스터 투영이 pannini 아님")
src_d = m["stages"]["slice"]["actual_duration_s"]
for p in people:
    f = os.path.join(OUT, "masters", f"cam_{p['id']}.mp4")
    if not os.path.exists(f): fails.append(f"마스터 없음 {f}"); continue
    if abs(dur(f) - src_d) > 1.0: fails.append(f"마스터 길이 {p['id']} {dur(f):.1f} vs {src_d}")
shots = json.load(open(os.path.join(OUT, "premiere_shots.json")))
for s in shots:
    cx, cy, cw, ch = s["crop"]
    if s["scale"] > 100.5: fails.append(f"확대율 {s['scale']}% > 100 (cam_{s['cam']} {s['start']})")
    if cx - cw / 2 < -1 or cx + cw / 2 > mr["w"] + 1 or cy - ch / 2 < -1 or cy + ch / 2 > mr["h"] + 1:
        fails.append(f"크롭이 마스터 밖 (cam_{s['cam']} {s['start']})")
if abs(dur(os.path.join(OUT, "preview.mp4")) - shots[-1]["end"]) > 1.0: fails.append("프리뷰 길이 불일치")
if subprocess.run(["xmllint", "--noout", os.path.join(OUT, "premiere.xml")]).returncode: fails.append("XML 무효")
print("\n".join(fails) if fails else f"V5 OK — 인물 {len(people)}명, 샷 {len(shots)}개, 확대율 최대 {max(s['scale'] for s in shots):.1f}%, 길이 {src_d:.0f}s")
sys.exit(1 if fails else 0)
