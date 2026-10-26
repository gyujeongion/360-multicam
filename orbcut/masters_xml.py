"""인물별 광각 마스터(4K) + 화자 컷 → 프리미어용 XML(샷마다 Scale/Position 모션 값).

샷 구도는 reframe_shots.compose(얼굴 기준)로 정하고, 그 방향·화각을 마스터 프레임 안의
크롭 창으로 바꿔 Basic Motion(scale %, center)으로 기록한다. 렌더 없이 프리미어에서 바로
Scale/Position만 만져 수정할 수 있다. --preview는 같은 크롭을 ffmpeg로 적용해 프리미어 없이 검수.

FCP7 XML center: 시퀀스 중앙 (0,0), 세로 +가 아래. 프리미어 26.5 실측: Position(0~1) = 0.5 + 2×center
(center 1 = 프레임 절반) → 의도한 이동(프레임 비율) δ를 center = δ/2로 기록. shots.json의 horiz/vert는 δ 그대로(UXP 패널이 Position에 직접 씀).

usage: python masters_xml.py <people.json> <asd_cams_dir> <edit.json> <masters_dir> <out.xml> [--preview out.mp4]
"""
import argparse, json, math, os, subprocess, urllib.parse
from types import SimpleNamespace
from xml.sax.saxutils import escape

import numpy as np

from .engine import pannini_project
from .reframe import compose, face_dirs, load_cam_render

SEQ_W, SEQ_H, SEQ_RATE = 1920, 1080, 30000 / 1001
WIDE = "wide"           # 실내 광각 마스터 id(track.WIDE_ID) — 화자 분리 컷의 와이드 샷
RATE = "<rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate>"


def fr(sec):
    return int(round(sec * SEQ_RATE))


def direction(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])  # x 오른쪽, y 위, z 앞


def _to_cam(d, m):
    """월드 방향 → 마스터 카메라 좌표 (engine._orient의 역: yaw 되돌리고 pitch 되돌림)."""
    ym, pm = math.radians(m["yaw"]), math.radians(m["pitch"])
    x = d[0] * math.cos(ym) - d[2] * math.sin(ym)
    z = d[0] * math.sin(ym) + d[2] * math.cos(ym)
    y = -d[1]                                   # direction()은 y 위+, 화면 좌표는 y 아래+
    y, z = y * math.cos(pm) + z * math.sin(pm), -y * math.sin(pm) + z * math.cos(pm)
    return np.array([x, y, z])


def _px(cam, m):
    """마스터 카메라 좌표 → 마스터 픽셀 (render.json의 투영 그대로)."""
    if m.get("proj") == "pannini":
        return pannini_project(cam, m["hfov"], m["w"], m["h"], m.get("pannini_d", 0.7))
    f = (m["w"] / 2) / math.tan(math.radians(m["hfov"] / 2))
    return m["w"] / 2 + f * cam[0] / cam[2], m["h"] / 2 + f * cam[1] / cam[2]


def to_master(c, m):
    """샷 구도 c(yaw,pitch,hfov) → 마스터 m 안의 크롭 중심(px)·폭(px). 폭은 샷 좌우 끝 방향의 실제 픽셀 거리."""
    cx, cy = _px(_to_cam(direction(c["yaw"], c["pitch"]), m), m)
    xl, _ = _px(_to_cam(direction(c["yaw"] - c["hfov"] / 2, c["pitch"]), m), m)
    xr, _ = _px(_to_cam(direction(c["yaw"] + c["hfov"] / 2, c["pitch"]), m), m)
    cw = min(abs(xr - xl), m["w"])
    ch = cw * SEQ_H / SEQ_W
    cx = float(np.clip(cx, cw / 2, m["w"] - cw / 2))  # 마스터 밖으로 나가지 않게
    cy = float(np.clip(cy, ch / 2, m["h"] - ch / 2))
    return cx, cy, cw, ch


def motion(cx, cy, cw, m):
    k = SEQ_W / cw
    return {"scale": round(100 * k, 2),
            "horiz": round((m["w"] / 2 - cx) * k / SEQ_W, 5),
            "vert": round((m["h"] / 2 - cy) * k / SEQ_H, 5)}


def file_el(fid, path, w, h, dur_f, defined):
    if fid in defined:
        return f'<file id="{fid}"/>'
    defined.add(fid)
    url = "file://localhost" + urllib.parse.quote(os.path.abspath(path))
    return (f'<file id="{fid}"><name>{escape(os.path.basename(path))}</name><pathurl>{url}</pathurl>{RATE}'
            f'<duration>{dur_f}</duration><media><video><samplecharacteristics><width>{w}</width>'
            f'<height>{h}</height></samplecharacteristics></video><audio><channelcount>1</channelcount></audio></media></file>')


def motion_filter(mo):
    return ('<filter><effect><name>Basic Motion</name><effectid>basic</effectid><effectcategory>motion</effectcategory>'
            '<effecttype>motion</effecttype><mediatype>video</mediatype>'
            f'<parameter><parameterid>scale</parameterid><name>Scale</name><valuemin>0</valuemin><valuemax>1000</valuemax>'
            f'<value>{mo["scale"]}</value></parameter>'
            f'<parameter><parameterid>center</parameterid><name>Center</name><value><horiz>{round(mo["horiz"] / 2, 6)}</horiz>'
            f'<vert>{round(mo["vert"] / 2, 6)}</vert></value></parameter></effect></filter>')


def main():
    ap = argparse.ArgumentParser()
    for k in ("people", "asd_cams", "edit", "masters", "out"):
        ap.add_argument(k)
    ap.add_argument("--preview")
    ap.add_argument("--face-ratio", type=float, default=0.32)
    ap.add_argument("--eye-line", type=float, default=0.38)
    ap.add_argument("--hfov-min", type=float, default=55)
    ap.add_argument("--hfov-max", type=float, default=85)
    a = ap.parse_args()
    cargs = SimpleNamespace(face_ratio=a.face_ratio, eye_line=a.eye_line, hfov_min=a.hfov_min,
                            hfov_max=a.hfov_max, cam_hfov=70, cam_pitch=0)

    people = {p["id"]: p for p in json.load(open(a.people))["people"]}
    segs = json.load(open(a.edit))["segments"]
    asd_cam = load_cam_render(a.asd_cams, cargs)
    faces = {pid: face_dirs(os.path.join(a.asd_cams, f"cam_{pid}"), p["yaw"], asd_cam) for pid, p in people.items()}
    mr = json.load(open(os.path.join(a.masters, "render.json")))

    shots, defined, vclips = [], set(), []
    total = fr(segs[-1]["end"])
    wide_ok = os.path.exists(os.path.join(a.masters, f"cam_{WIDE}.mp4"))
    for i, s in enumerate(segs):
        pid = s["cam"]
        if pid == WIDE and not wide_ok:              # 실내 광각 마스터가 없으면 앞 샷 인물 유지
            pid = s["cam"] = shots[-1]["cam"] if shots else sorted(people)[0]
        if pid == WIDE:                               # 실내 광각: 마스터 전체 화면
            cw, ch = mr["w"], mr["w"] * SEQ_H / SEQ_W
            cx, cy = mr["w"] / 2, mr["h"] / 2
            mo = motion(cx, cy, cw, mr)
            shots.append({**s, "crop": [round(cx), round(cy), round(cw), round(ch)], **mo})
        else:
            m = {**mr, "yaw": people[pid]["yaw"], "pitch": people[pid].get("pitch", mr["pitch"])}
            c = compose(faces[pid], s["start"], s["end"], people[pid], cargs)
            cx, cy, cw, ch = to_master(c, m)
            mo = motion(cx, cy, cw, m)
            shots.append({**s, **c, "crop": [round(cx), round(cy), round(cw), round(ch)], **mo})
        st, en = fr(s["start"]), fr(s["end"])
        path = os.path.join(a.masters, f"cam_{pid}.mp4")
        vclips.append(f'<clipitem id="v{i}"><name>cam_{pid}</name>{RATE}<start>{st}</start><end>{en}</end>'
                      f'<in>{st}</in><out>{en}</out>{file_el("m" + pid, path, mr["w"], mr["h"], total, defined)}'
                      f'{motion_filter(mo)}</clipitem>')

    first = sorted(people)[0]
    apath = os.path.join(a.masters, f"cam_{first}.mp4")
    aclip = (f'<clipitem id="a1"><name>cam_{first}</name>{RATE}<start>0</start><end>{total}</end><in>0</in>'
             f'<out>{total}</out>{file_el("m" + first, apath, mr["w"], mr["h"], total, defined)}'
             '<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack></clipitem>')
    name = os.path.splitext(os.path.basename(a.out))[0]
    open(a.out, "w").write(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE xmeml><xmeml version="4">'
        f'<sequence id="seq1"><name>{escape(name)}</name>{RATE}<duration>{total}</duration><media>'
        f'<video><format><samplecharacteristics><width>{SEQ_W}</width><height>{SEQ_H}</height>{RATE}'
        f'</samplecharacteristics></format><track>{"".join(vclips)}</track></video>'
        f'<audio><track>{aclip}</track></audio></media></sequence></xmeml>')
    json.dump(shots, open(os.path.splitext(a.out)[0] + "_shots.json", "w"), indent=2)
    print(f"{len(shots)} shots → {a.out}")
    for s in shots[:8]:
        print(f"  cam_{s['cam']} {s['start']:6.2f}-{s['end']:6.2f} hfov {s.get('hfov', 0):5.1f} scale {s['scale']:6.2f}% "
              f"center ({s['horiz']:+.3f},{s['vert']:+.3f})")

    if a.preview:  # 프리미어와 같은 크롭을 ffmpeg로 — 모션 값 검수용
        ids = sorted(people) + ([WIDE] if any(s["cam"] == WIDE for s in shots) else [])
        idx = {p: k for k, p in enumerate(ids)}
        ins = sum([["-i", os.path.join(a.masters, f"cam_{p}.mp4")] for p in ids], [])
        parts = [f"[{idx[s['cam']]}:v]trim={s['start']}:{s['end']},setpts=PTS-STARTPTS,"
                 f"crop={s['crop'][2]}:{s['crop'][3]}:{s['crop'][0] - s['crop'][2] // 2}:{s['crop'][1] - s['crop'][3] // 2},"
                 f"scale={SEQ_W}:{SEQ_H}:flags=lanczos[p{k}]" for k, s in enumerate(shots)]
        fc = ";".join(parts) + ";" + "".join(f"[p{k}]" for k in range(len(shots))) + \
             f"concat=n={len(shots)}:v=1:a=0[v];[0:a]atrim=0:{segs[-1]['end']}[a]"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                        "-c:v", "h264_videotoolbox", "-b:v", "10M", "-c:a", "aac", a.preview], check=True)


if __name__ == "__main__":
    main()
