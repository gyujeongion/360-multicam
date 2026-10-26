"""컷마다 그 구간의 얼굴 위치·크기로 구도를 다시 잡아 샷 단위로 렌더한다 (샷 안에서는 고정 = 멀티캠 느낌).

얼굴 위치는 Light-ASD 트랙(cam_X 원근 영상의 픽셀 좌표)을 구면 각도로 되돌려 쓴다.
구도 규칙: 얼굴 높이가 화면 세로의 FACE_RATIO, 얼굴 중심이 위에서 EYE_LINE 지점.

usage: python reframe_shots.py <people.json> <cams_dir> <edit.json> <out_dir> --start 240
       [--face-ratio 0.22 --hfov-min 55 --hfov-max 100 --cam-hfov 70 --cam-pitch 0]
"""
import argparse, json, math, os, pickle, subprocess

import numpy as np

ASD_FPS = 25


def rect_vfov(hfov, w=16, h=9):
    """직선 원근(flat)에서 가로 화각에 대응하는 세로 화각(도). hfov*h/w는 틀림(세로 늘어남)."""
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov / 2)) * h / w))


def load_cam_render(cams_dir, args):
    """render_angles.py가 남긴 실제 렌더 화각. 없으면(구버전 렌더) 당시 규칙 hfov*9/16."""
    p = os.path.join(cams_dir, "render.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"hfov": args.cam_hfov, "vfov": args.cam_hfov * 9 / 16, "pitch": args.cam_pitch, "w": 1920, "h": 1080}


def face_dirs(cam_dir, base_yaw, cam):
    """중앙 인물 트랙의 프레임별 (yaw, pitch, 얼굴 각높이). 키 = ASD 프레임 번호."""
    w, h = cam["w"], cam["h"]
    f = (w / 2) / math.tan(math.radians(cam["hfov"] / 2))
    fy = (h / 2) / math.tan(math.radians(cam["vfov"] / 2))
    cam_pitch = cam["pitch"]
    tracks = pickle.load(open(os.path.join(cam_dir, "pywork", "tracks.pckl"), "rb"))
    out = {}
    for tr in tracks:
        xs, ys, ss = (np.array(tr["proc_track"][k]) for k in ("x", "y", "s"))
        if abs(xs.mean() - w / 2) > w * 0.2:
            continue
        for i, fr in enumerate(tr["track"]["frame"].tolist()):
            dx, dy = (xs[i] - w / 2) / f, (h / 2 - ys[i]) / fy
            yaw = base_yaw + math.degrees(math.atan(dx))
            pitch = cam_pitch + math.degrees(math.atan(dy / math.sqrt(1 + dx * dx)))
            size = math.degrees(2 * math.atan(ss[i] / fy))  # s = 박스 반변 → 얼굴 각높이
            out[fr] = (yaw, pitch, size)
    return out


def compose(faces, a, b, person, args):
    pts = [faces[k] for k in range(int(a * ASD_FPS), int(b * ASD_FPS)) if k in faces]
    if len(pts) < 5:
        return {"yaw": person["yaw"], "pitch": 8.0, "hfov": 70.0, "auto": False}
    yaw, pitch, size = np.median(np.array(pts), axis=0)
    # 샷 안에서 얼굴이 움직인 범위까지 담기 (상위 90% 지점)
    spread = np.percentile(np.abs(np.array(pts)[:, 0] - yaw), 90)
    want_vfov = size / args.face_ratio
    want_hfov = 2 * math.degrees(math.atan(math.tan(math.radians(want_vfov / 2)) * 16 / 9))
    hfov = float(np.clip(max(want_hfov, (size + 2 * spread) * 1.6), args.hfov_min, args.hfov_max))
    vfov = rect_vfov(hfov)
    yaw = (yaw + 180) % 360 - 180  # E처럼 ±180 경계에 앉은 인물
    return {"yaw": round(float(yaw), 2), "pitch": round(float(pitch - (0.5 - args.eye_line) * vfov), 2),
            "hfov": round(hfov, 1), "auto": True}


def render(src, t0, dur, lens_fov, c, out, equirect):
    proj = "pannini" if c["hfov"] > 85 else "flat"  # 광각은 가장자리 늘어짐 덜한 투영
    inp = "[0:v]v360=input=e" if equirect else f"[0:0][0:1]hstack,v360=input=dfisheye:ih_fov={lens_fov}:iv_fov={lens_fov}"
    vf = (f"{inp}:output={proj}:yaw={c['yaw']}:pitch={c['pitch']}:h_fov={c['hfov']}:"
          f"v_fov={round(rect_vfov(c['hfov']), 3)}:w=1920:h=1080:interp=lanczos[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", src,
                    "-filter_complex", vf, "-map", "[v]", "-map", "0:a:0", "-c:v", "hevc_videotoolbox",
                    "-b:v", "16M", "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "192k", out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("people"); ap.add_argument("cams_dir"); ap.add_argument("edit"); ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=0, help="cams가 원본의 몇 초부터 렌더됐는지")
    ap.add_argument("--src", help="스튜디오에서 뽑은 equirect 파일(있으면 이걸로 렌더)")
    ap.add_argument("--face-ratio", type=float, default=0.32, help="얼굴 높이 / 화면 세로")
    ap.add_argument("--eye-line", type=float, default=0.38, help="얼굴 중심 위치(위에서 비율)")
    ap.add_argument("--hfov-min", type=float, default=55)
    ap.add_argument("--hfov-max", type=float, default=85)
    ap.add_argument("--cam-hfov", type=float, default=70, help="ASD에 쓴 cam_X 렌더 화각")
    ap.add_argument("--cam-pitch", type=float, default=0, help="ASD에 쓴 cam_X 렌더 pitch")
    args = ap.parse_args()

    meta = json.load(open(args.people))
    people = {p["id"]: p for p in meta["people"]}
    segs = json.load(open(args.edit))["segments"]
    os.makedirs(args.out_dir, exist_ok=True)
    cam = load_cam_render(args.cams_dir, args)
    faces = {pid: face_dirs(os.path.join(args.cams_dir, f"cam_{pid}"), p["yaw"], cam)
             for pid, p in people.items()}

    src = args.src or meta["insv"]
    shots = []
    for i, s in enumerate(segs):
        c = compose(faces[s["cam"]], s["start"], s["end"], people[s["cam"]], args)
        out = os.path.join(args.out_dir, f"shot_{i:03d}_{s['cam']}.mp4")
        render(src, args.start + s["start"], s["end"] - s["start"], meta["lens_fov"], c, out, bool(args.src) or meta.get("equirect", False))
        shots.append({**s, **c, "file": out})
        print(f"{i:3d} cam_{s['cam']} {s['start']:7.2f}-{s['end']:7.2f} yaw {c['yaw']:7.1f} "
              f"pitch {c['pitch']:5.1f} hfov {c['hfov']:5.1f} {'auto' if c['auto'] else 'default'}", flush=True)
    json.dump(shots, open(os.path.join(args.out_dir, "shots.json"), "w"), indent=2)

    lst = os.path.join(args.out_dir, "concat.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(s['file'])}'\n" for s in shots))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(args.out_dir, "edit_reframed.mp4")], check=True)


if __name__ == "__main__":
    main()
