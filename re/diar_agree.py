"""화자 컷 두 개가 '같은 사람을 비추는가'를 화면 방향으로 비교(인물 ID가 실행마다 달라도 됨).
기준 컷(예: 사람이 확인한 시트 = v2 ID)과 대상 컷(v3 ID)을 각자의 paths.json으로 방향(yaw)으로 바꿔
0.1초마다 비교: 둘 다 와이드거나, 둘 다 인물이고 방향 차 < 20°면 일치. 기준이 비추는 인물이 그 시각에 화면에 없으면(present
거짓) 기준 쪽이 틀린 시각이라 집계에서 뺀다.
usage: python re/diar_agree.py <기준 out_dir> <기준 cut.json> <대상 out_dir> <대상 cut.json>  (cut.json = segments[start,end,cam])"""
import json, sys

import numpy as np

WIDE = {"W", "wide"}


def path_fn(out_dir):
    P = json.load(open(f"{out_dir}/paths.json"))["paths"]
    tab = {k: np.array([[r[0], r[1], 1.0 if (len(r) < 5 or r[4]) else 0.0] for r in v]) for k, v in P.items()}

    def at(cam, t):
        a = tab.get(cam)
        if a is None:
            return None, False
        if t < a[0, 0] - 0.3 or t > a[-1, 0] + 0.3:          # 경로 기록 밖 = 그 ID가 아직(이미) 없음
            return None, False
        i = min(np.searchsorted(a[:, 0], t), len(a) - 1)
        return float(a[i, 1]), bool(a[i, 2] > 0.5)
    return at


def cam_at(segs, t):
    return next((s["cam"] for s in segs if s["start"] <= t < s["end"]), segs[-1]["cam"])


def agree(ref_dir, ref_cut, tgt_dir, tgt_cut, step=0.1):
    rs, ts = json.load(open(ref_cut))["segments"], json.load(open(tgt_cut))["segments"]
    ra, ta = path_fn(ref_dir), path_fn(tgt_dir)
    end = min(rs[-1]["end"], ts[-1]["end"])
    ok = n = skip = 0
    bad = []
    for t in np.arange(0, end, step):
        a, b = cam_at(rs, t), cam_at(ts, t)
        if a in WIDE or b in WIDE:
            n += 1; same = (a in WIDE) and (b in WIDE); ok += same
        else:
            ya, pa = ra(a, t); yb, pb = ta(b, t)
            if ya is None or not pa:
                skip += 1; continue
            n += 1
            same = yb is not None and pb and abs((ya - yb + 180) % 360 - 180) < 20
            ok += same
        if not same:
            if bad and abs(bad[-1][1] - t) < step * 1.5 and bad[-1][2:] == [a, b]:
                bad[-1][1] = round(t + step, 2)
            else:
                bad.append([round(float(t), 2), round(float(t + step), 2), a, b])
    return {"agree": round(ok / max(n, 1), 3), "frames": n, "ref_wrong_skipped": skip,
            "disagree": [x for x in bad if x[1] - x[0] >= 0.5]}


if __name__ == "__main__":
    r = agree(*sys.argv[1:5])
    print(json.dumps({k: r[k] for k in ("agree", "frames", "ref_wrong_skipped")}, ensure_ascii=False),
          "불일치(≥0.5s):", r["disagree"][:12])
