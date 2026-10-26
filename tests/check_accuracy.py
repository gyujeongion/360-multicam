"""M7: 사람 정답 라벨 대비 화자 컷 프레임 정확도 ≥ 85% (원형 테이블 4:00~7:00, 기본 파라미터).
usage: python tests/check_accuracy.py   (config에 원격 호스트가 있으면 그쪽에서 실행)"""
import json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
SRC, OUT = os.path.join(REPO, "work", "src_240_180.mkv"), os.path.join(REPO, "work", "v5")
LABELS = os.path.join(REPO, "tests", "fixtures", "labels_v4_240_180.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _remote import ON_REMOTE, delegate, remote_host
# 원격 호스트가 설정돼 있으면 항상 그쪽(화자 감지 결과 캐시가 있는 곳)에서 잰다.
# 로컬에 구간 파일만 있고 결과가 낡으면 다른 화자 점수로 재게 된다(실측: 로컬 82.8% vs 원격 86.4%)
if (remote_host() and not ON_REMOTE) or not os.path.exists(SRC):
    rc = delegate(".venv/bin/python tests/check_accuracy.py")
    if rc is None:
        print("픽스처 없음·원격 미설정 — 건너뜀"); sys.exit(0)
    sys.exit(rc)

r = subprocess.run([sys.executable, "-m", "multicam360", "--json", "all", SRC, "--out", OUT, "--meta",
                    os.path.join(REPO, "re", "tail8m.bin"), "--until", "cut"], cwd=REPO, capture_output=True, text=True)
m = json.load(open(os.path.join(OUT, "manifest.json")))
assert m["stages"]["cut"]["status"] == "ok", r.stdout[-400:] + r.stderr[-400:]
from multicam360.eval import evaluate
res = evaluate(json.load(open(os.path.join(OUT, "edit.json"))), json.load(open(LABELS)), 0.5)
print(f"frame_acc {res['frame_acc']:.3f} · 잘못 비춤 {res['wrong_s']:.1f}s · 컷 {res['n_cuts']}개 · 인물별 "
      + ", ".join(f"{k} {v['shown_while_talking']:.0%}" for k, v in res["per_person"].items()))
sys.exit(0 if res["frame_acc"] >= 0.85 else 1)
