"""앵글별 Light-ASD 결과로 화자 전환 컷 리스트를 만들고, 프리미어용 FCP7 XML로 내보낸다.

각 cam_X 앵글의 '화면 중앙 인물' 트랙만 그 인물의 발화 점수로 쓴다
(가장자리에 걸친 옆사람 얼굴은 그 사람 본인 앵글에서 따로 잡히므로 무시).

usage: python edit_plan.py <cams_dir> <out_prefix> [--min-shot 1.5 --hold 0.5]
"""
import argparse, glob, json, os, pickle, urllib.parse
from xml.sax.saxutils import escape

import numpy as np

ASD_FPS = 25
SEQ_RATE = 30000 / 1001  # 원본 29.97


def person_scores(cam_dir, n_frames, frame_w=1920):
    """중앙 인물 트랙들의 프레임별 발화 점수(없으면 nan)."""
    tracks = pickle.load(open(os.path.join(cam_dir, "pywork", "tracks.pckl"), "rb"))
    scores = pickle.load(open(os.path.join(cam_dir, "pywork", "scores.pckl"), "rb"))
    out = np.full(n_frames, np.nan)
    for tr, sc in zip(tracks, scores):
        xs, ss = np.array(tr["proc_track"]["x"]), np.array(tr["proc_track"]["s"])
        if abs(xs.mean() - frame_w / 2) > frame_w * 0.2:
            continue
        for i, f in enumerate(tr["track"]["frame"].tolist()):
            if f < n_frames and i < len(sc):
                out[f] = sc[i] if np.isnan(out[f]) else max(out[f], sc[i])
    return out


def smooth(x, win):
    k = np.ones(win) / win
    filled = np.nan_to_num(x, nan=-5.0)
    return np.convolve(filled, k, mode="same")


def decide(S, ids, min_shot, hold, thr):
    """히스테리시스 전환: 새 화자가 hold 이상 지속 + 현재 샷이 min_shot 이상일 때만 컷."""
    n = S.shape[1]
    active = np.where(S.max(0) > thr, S.argmax(0), -1)
    hold_f, min_f = int(hold * ASD_FPS), int(min_shot * ASD_FPS)
    first = next((a for a in active if a >= 0), 0)
    cur, shot_start, segs = first, 0, []
    run_who, run_start = -1, 0
    for f in range(n):
        a = active[f]
        if a != run_who:
            run_who, run_start = a, f
        if a >= 0 and a != cur and f - run_start + 1 >= hold_f and run_start - shot_start >= min_f:
            segs.append((shot_start, run_start, ids[cur]))
            cur, shot_start = a, run_start
    segs.append((shot_start, n, ids[cur]))
    return [{"start": s / ASD_FPS, "end": e / ASD_FPS, "cam": c} for s, e, c in segs]


def fr(sec):
    return int(round(sec * SEQ_RATE))


def file_el(fid, path, dur_f, defined):
    if fid in defined:
        return f'<file id="{fid}"/>'
    defined.add(fid)
    url = "file://localhost" + urllib.parse.quote(os.path.abspath(path))
    return (f'<file id="{fid}"><name>{escape(os.path.basename(path))}</name><pathurl>{url}</pathurl>'
            f'<rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate><duration>{dur_f}</duration>'
            f'<media><video><samplecharacteristics><width>1920</width><height>1080</height></samplecharacteristics></video>'
            f'<audio><channelcount>2</channelcount></audio></media></file>')


def to_xmeml(segs, cams_dir, name, audio_cam):
    rate = "<rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate>"
    total = fr(segs[-1]["end"])
    defined, vclips = set(), []
    for i, s in enumerate(segs):
        a, b = fr(s["start"]), fr(s["end"])
        path = os.path.join(cams_dir, f"cam_{s['cam']}.mp4")
        vclips.append(f'<clipitem id="v{i}"><name>cam_{s["cam"]}</name>{rate}<start>{a}</start><end>{b}</end>'
                      f'<in>{a}</in><out>{b}</out>{file_el("f" + s["cam"], path, total, defined)}</clipitem>')
    apath = os.path.join(cams_dir, f"cam_{audio_cam}.mp4")
    aclips = [f'<clipitem id="a{ch}"><name>cam_{audio_cam}</name>{rate}<start>0</start><end>{total}</end>'
              f'<in>0</in><out>{total}</out>{file_el("f" + audio_cam, apath, total, defined)}'
              f'<sourcetrack><mediatype>audio</mediatype><trackindex>{ch}</trackindex></sourcetrack></clipitem>'
              for ch in (1, 2)]
    return ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE xmeml><xmeml version="4">'
            f'<sequence id="seq1"><name>{escape(name)}</name>{rate}<duration>{total}</duration><media>'
            '<video><format><samplecharacteristics><width>1920</width><height>1080</height>'
            f'{rate}</samplecharacteristics></format><track>{"".join(vclips)}</track></video>'
            f'<audio><track>{aclips[0]}</track><track>{aclips[1]}</track></audio>'
            '</media></sequence></xmeml>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cams_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--min-shot", type=float, default=1.0, help="한 샷 최소 길이(초)")
    ap.add_argument("--hold", type=float, default=1.0, help="새 화자가 이만큼 이어져야 전환(초)")
    ap.add_argument("--smooth", type=float, default=0.8, help="점수 평활 창(초)")
    ap.add_argument("--thr", type=float, default=-0.5, help="발화 판정 점수 임계")
    ap.add_argument("--paths", help="추적 모드 paths.json — 인물이 없는(카메라 이동으로 시야 밖) 구간은 컷 후보에서 제외")
    a = ap.parse_args()

    cam_dirs = sorted(d for d in glob.glob(os.path.join(a.cams_dir, "cam_*")) if os.path.isdir(d))
    ids = [os.path.basename(d)[4:] for d in cam_dirs]
    n = min(len(glob.glob(os.path.join(d, "pyframes", "*.jpg"))) for d in cam_dirs)
    S = np.stack([smooth(person_scores(d, n), int(a.smooth * ASD_FPS)) for d in cam_dirs])
    if a.paths:
        P = json.load(open(a.paths))["paths"]
        tt = np.arange(n) / ASD_FPS
        for k, pid in enumerate(ids):
            rows = P.get(pid)
            if rows and len(rows[0]) >= 5:
                pres = np.interp(tt, [r[0] for r in rows], [1.0 if r[4] else 0.0 for r in rows], left=0.0, right=0.0) > 0.5   # 경로 기록 밖 = 없음
                S[k, ~pres] = -5.0                     # 존재하지 않으면 발화 불가로 취급

    segs = decide(S, ids, a.min_shot, a.hold, a.thr)
    talk = {i: round(float((S[k] > a.thr).sum()) / ASD_FPS, 1) for k, i in enumerate(ids)}
    json.dump({"talk_seconds": talk, "segments": segs}, open(a.out_prefix + ".json", "w"), indent=2)
    open(a.out_prefix + ".xml", "w").write(
        to_xmeml(segs, os.path.abspath(a.cams_dir), os.path.basename(a.out_prefix), ids[0]))
    print(f"{len(segs)} shots, talk seconds {talk}")
    for s in segs[:15]:
        print(f"  {s['start']:7.2f}-{s['end']:7.2f}  cam_{s['cam']}")


if __name__ == "__main__":
    main()
