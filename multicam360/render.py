"""엔진으로 원본(듀얼 어안)에서 직접 렌더: 파노라마 / 인물 앵글 / 4K 마스터.

usage:
  python -m multicam360.render panorama <src> <out.mp4> --meta M [--width 3840]
  python -m multicam360.render views <src> <people.json> <out_dir> --meta M --hfov 70 --w 1920 --h 1080 --pitch 0
        [--bitrate 12M] [--flow] [--jobs N]
views는 인물마다 cam_<ID>.mp4 + render.json(실제 화각 기록). --jobs: 병렬 프로세스 수(인물 단위 분할).
"""
import argparse, json, os, subprocess, sys

import math

from .engine import Projector, equirect_dirs, flat_dirs, load_lenses, pannini_dirs, pannini_scale, render_many
from .reframe import rect_vfov


def render_json(a, out_dir, tracked=False):
    """후속 단계(얼굴 좌표 → 각도 역산·XML 크롭)가 쓰는 실제 렌더 기하."""
    pann = getattr(a, "proj", "flat") == "pannini"
    vfov = (2 * math.degrees(math.atan((a.h / 2) / pannini_scale(a.hfov, a.w, a.pannini_d))) if pann
            else rect_vfov(a.hfov, a.w, a.h))
    json.dump({"hfov": a.hfov, "vfov": vfov, "pitch": a.pitch, "w": a.w, "h": a.h, "src": a.src,
               "engine": "multicam360.engine", "flow": getattr(a, "flow", False), "proj": getattr(a, "proj", "flat"),
               "pannini_d": a.pannini_d if pann else 0, "tracked": tracked},
              open(os.path.join(out_dir, "render.json"), "w"), indent=2)


MAX_HW_ENC = int(os.environ.get("MULTICAM360_MAX_ENC", 4))   # 동시 하드웨어 인코딩 수 상한(track.py와 동일. M2 Ultra 실측: 10비트 -q:v는 4, 비트레이트 5, 8비트 6+)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("panorama")
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--meta", required=True); p.add_argument("--width", type=int, default=3840)
    p.add_argument("--bitrate", default="20M"); p.add_argument("--no-flow", action="store_true")
    p.add_argument("--start", type=float); p.add_argument("--dur", type=float)
    v = sub.add_parser("views")
    v.add_argument("src"); v.add_argument("people"); v.add_argument("out_dir")
    v.add_argument("--meta", required=True)
    v.add_argument("--hfov", type=float, default=70); v.add_argument("--pitch", type=float, default=0)
    v.add_argument("--w", type=int, default=1920); v.add_argument("--h", type=int, default=1080)
    v.add_argument("--bitrate", default="12M"); v.add_argument("--flow", action="store_true")
    v.add_argument("--proj", choices=["flat", "pannini"], default="flat", help="pannini = 가장자리 늘어짐 감소")
    v.add_argument("--pannini-d", type=float, default=0.7)
    v.add_argument("--only", help="쉼표로 구분한 인물 ID만 (병렬 분할용)")
    v.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    lenses = load_lenses(a.meta)

    if a.kind == "panorama":
        pj = Projector(lenses, equirect_dirs(a.width), flow=not a.no_flow)
        n = render_many(a.src, [(pj, a.out, a.bitrate)], start=a.start, dur=a.dur)
        print(json.dumps({"frames": n, "out": a.out}))
        return 0

    people = json.load(open(a.people))["people"]
    os.makedirs(a.out_dir, exist_ok=True)
    if a.jobs > 1 and not a.only:   # 인물 단위로 나눠 하위 프로세스 병렬 실행
        ids = [p["id"] for p in people]
        clean, skip = [sys.executable, "-m", "multicam360.render"], False
        for x in (argv or sys.argv[1:]):         # 하위 프로세스엔 --jobs N 을 빼고 그대로 전달
            if skip: skip = False; continue
            if x == "--jobs": skip = True; continue
            clean.append(x)
        for w in range(0, len(ids), MAX_HW_ENC):   # 전체 동시 인코더 수를 VideoToolbox 한도 안으로(10비트 품질모드 = 4)
            wave = ids[w:w + MAX_HW_ENC]
            chunks = [wave[i::a.jobs] for i in range(a.jobs) if wave[i::a.jobs]]
            procs = [subprocess.Popen(clean + ["--only", ",".join(c)]) for c in chunks]
            rcs = [pr.wait() for pr in procs]
            if any(rcs):
                print(f"병렬 렌더 실패 {rcs}", file=sys.stderr); return 1
    else:
        sel = [p for p in people if not a.only or p["id"] in a.only.split(",")]
        outs = []
        for p in sel:
            hfov, pitch = p.get("hfov", a.hfov), p.get("pitch", a.pitch)
            dirs = (pannini_dirs(p["yaw"], pitch, hfov, a.w, a.h, a.pannini_d) if a.proj == "pannini"
                    else flat_dirs(p["yaw"], pitch, hfov, a.w, a.h))
            pj = Projector(lenses, dirs, flow=a.flow)
            outs.append((pj, os.path.join(a.out_dir, f"cam_{p['id']}.mp4"), a.bitrate))
        for w in range(0, len(outs), MAX_HW_ENC):
            render_many(a.src, outs[w:w + MAX_HW_ENC])
    if not a.only:
        render_json(a, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
