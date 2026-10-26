"""화자 컷 정확도 평가: 사람 정답(labels.json) vs 예측(edit.json).

labels.json 형식 (라벨링 도구 출력):
  {"people": ["A","B",...], "duration": 180.0,
   "segments": [{"start": 1.2, "end": 4.8, "speakers": ["C"]}, ...]}   # 겹쳐 말하면 speakers 여러 명
edit.json: multicam360.cut 출력 {"segments": [{"start","end","cam"}...]}

지표
- frame_acc: 누군가 말하는 시간 중, 화면(cam)이 말하는 사람 중 하나인 비율
- silence_s: 아무도 안 말하는 시간(평가 제외, 참고)
- cut_precision / cut_recall: 화자 전환 시점 ↔ 컷 시점 매칭(±tol초)
- wrong_s: 말하는 사람이 아닌 사람을 비춘 총 시간
"""
import argparse, json, sys

import numpy as np

HZ = 10


def _grid(duration):
    return np.arange(0, duration, 1 / HZ)


def label_sets(labels, t):
    out = [set() for _ in t]
    for s in labels["segments"]:
        i0, i1 = int(np.floor(s["start"] * HZ)), int(np.ceil(s["end"] * HZ))
        for i in range(max(i0, 0), min(i1, len(t))):
            out[i] |= set(s["speakers"])
    return out


def pred_cams(edit, t):
    cams = np.array([None] * len(t), dtype=object)
    for s in edit["segments"]:
        cams[(t >= s["start"]) & (t < s["end"])] = s["cam"]
    return cams


def change_points(seq_sets):
    """정답 화자 전환 시점: 말하는 사람 집합이 비어있지 않은 상태에서 바뀌는 시점(침묵 구간은 건너뜀)."""
    pts, prev = [], None
    for i, s in enumerate(seq_sets):
        if not s:
            continue
        if prev is not None and not (s & prev):
            pts.append(i / HZ)
        prev = s
    return pts


def match(a, b, tol):
    """a 각 점에 대해 b 안에 tol 이내 점이 있으면 매칭(일대일)."""
    used, hit = set(), 0
    for x in a:
        cand = [j for j, y in enumerate(b) if abs(x - y) <= tol and j not in used]
        if cand:
            used.add(min(cand, key=lambda j: abs(b[j] - x))); hit += 1
    return hit


def evaluate(edit, labels, tol=0.75):
    dur = labels.get("duration") or max(s["end"] for s in labels["segments"])
    dur = min(dur, max(s["end"] for s in edit["segments"]))
    t = _grid(dur)
    gt, pr = label_sets(labels, t), pred_cams(edit, t)
    speaking = np.array([bool(s) for s in gt])
    correct = np.array([pr[i] in gt[i] for i in range(len(t))])
    frame_acc = float(correct[speaking].mean()) if speaking.any() else float("nan")
    gt_cuts = change_points(gt)
    pr_cuts = [s["start"] for s in edit["segments"][1:]]
    tp_r = match(gt_cuts, pr_cuts, tol)
    tp_p = match(pr_cuts, gt_cuts, tol)
    per = {}
    for p in labels.get("people", []):
        mask = np.array([p in s for s in gt])
        if mask.any():
            per[p] = {"talk_s": round(mask.sum() / HZ, 1), "shown_while_talking": round(float((pr[mask] == p).mean()), 3)}
    return {"frame_acc": round(frame_acc, 4), "speaking_s": round(speaking.sum() / HZ, 1),
            "silence_s": round((~speaking).sum() / HZ, 1), "wrong_s": round((speaking & ~correct).sum() / HZ, 1),
            "cut_precision": round(tp_p / len(pr_cuts), 3) if pr_cuts else None,
            "cut_recall": round(tp_r / len(gt_cuts), 3) if gt_cuts else None,
            "n_cuts": len(pr_cuts), "n_gt_changes": len(gt_cuts), "per_person": per}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("edit"); ap.add_argument("labels")
    ap.add_argument("--tol", type=float, default=0.75)
    a = ap.parse_args(argv)
    r = evaluate(json.load(open(a.edit)), json.load(open(a.labels)), a.tol)
    print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
