"""M15: Insta360 Studio 360 내보내기 대비 orbcut 파노라마 화질 — 기본 설정으로 5초 구간을 8K 렌더해 비교한다(goalset 검증 600초 한도 안).
기준(Studio = 1.0): 본문 선명도 ≥0.9, 이음새 선명도 ≥0.7, 노이즈 ≤1.0, 본문 구조 유사도 ≥0.8, 밝기 차 |ΔL| ≤2.
재료: ORBCUT_STUDIO_DIR(기본 ~/orbcut_cmp)에 원본 VID_*_017.insv 와 Studio 내보내기 studio_017.mp4.
usage: python tests/check_studio_compare.py [--remote]   (8K 디코딩이라 맥북 로컬 실행은 막는다)"""
import glob, json, os, platform, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "--remote" in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _remote import delegate
    rc = delegate(".venv/bin/python tests/check_studio_compare.py")
    if rc is None:
        print("원격 호스트 미설정(ORBCUT_REMOTE_HOST) — 이 검증은 8K라 원격에서만"); sys.exit(2)
    sys.exit(rc)

if os.environ.get("ORBCUT_ON_REMOTE") != "1" and "MacBook" in subprocess.run(
        ["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout + platform.node():
    print("맥북에서 8K 비교 실행 금지 — --remote로"); sys.exit(2)

os.chdir(REPO)
D = os.path.expanduser(os.environ.get("ORBCUT_STUDIO_DIR", "~/orbcut_cmp"))
src = sorted(glob.glob(os.path.join(D, "VID_*_017.insv")))
studio = os.path.join(D, "studio_017.mp4")
if not src or not os.path.exists(studio):
    print(f"재료 없음: {D}/VID_*_017.insv, studio_017.mp4"); sys.exit(1)
seg, out = os.path.join(D, "seg_default.mp4"), os.path.join(D, "vf_default")
r = subprocess.run([sys.executable, "-m", "orbcut.render", "panorama", src[0], seg, "--meta", src[0],
                    "--width", "7680", "--start", "10", "--dur", "5"], capture_output=True, text=True)
if r.returncode:
    print("렌더 실패:", r.stderr[-800:]); sys.exit(1)
subprocess.run([sys.executable, "re/studio_compare.py", studio, seg, out, "--times", "11,12,13,14", "--offset", "10"],
               check=True, capture_output=True)
s = json.load(open(os.path.join(out, "studio_compare.json")))["summary"]
json.dump(s, open(os.path.join(REPO, "tests", "studio_compare.json"), "w"), indent=1)
fails = [m for ok, m in [
    (s["sharp_body_ratio"] >= 0.9, f"본문 선명도 {s['sharp_body_ratio']:.2f} < 0.9"),
    (s["sharp_seam_ratio"] >= 0.7, f"이음새 선명도 {s['sharp_seam_ratio']:.2f} < 0.7"),
    (s["noise_ratio"] <= 1.0, f"노이즈 {s['noise_ratio']:.2f} > 1.0"),
    (s["ssim_body"] >= 0.8, f"구조 유사도 {s['ssim_body']:.2f} < 0.8"),
    (abs(s["lab_delta"][0]) <= 2, f"밝기 차 {s['lab_delta'][0]}"),
] if not ok]
print("\n".join(fails) if fails else
      f"STUDIO OK — 본문 선명도 {s['sharp_body_ratio']:.2f}, 이음새 {s['sharp_seam_ratio']:.2f}, "
      f"노이즈 {s['noise_ratio']:.2f}, 구조 유사도 {s['ssim_body']:.2f}, ΔL {s['lab_delta'][0]} (Studio=1.0)")
sys.exit(1 if fails else 0)
