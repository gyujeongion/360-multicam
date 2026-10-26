"""360 영상에서 테이블에 앉은 인물들의 yaw 각도를 자동 검출한다.

여러 시점의 equirect 프레임을 45도 간격 원근 뷰로 펼쳐 S3FD로 얼굴을 찾고,
yaw로 환산한 뒤 군집화한다. 큰 얼굴(가까운 사람)이 여러 시점에 반복 등장하면 '테이블 인원'으로 본다.

usage: python detect_people.py <insv> <out.json> [--times 60,180,...]
"""
import argparse, json, math, os, subprocess, sys, tempfile

import cv2
import numpy as np

from .config import asd_dir

VIEW_W, VIEW_H, VIEW_HFOV, VIEW_VFOV = 960, 720, 90, 70
LENS_FOV = 190  # ffmpeg dfisheye 스티칭용 렌즈 화각 (프로토타입 값)


def stitch_filter(extra=""):
    return f"[0:0][0:1]hstack,v360=input=dfisheye:output=e:ih_fov={LENS_FOV}:iv_fov={LENS_FOV}:w=3072:h=1536{extra}"


def grab_views(src, t, tmp, equirect):
    equi = os.path.join(tmp, f"e_{t}.png")
    vf = ["-vf", "scale=3072:1536"] if equirect else ["-filter_complex", stitch_filter()]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", src] + vf + ["-frames:v", "1", equi], check=True)
    views = []
    for yaw in range(-180, 180, 45):
        out = os.path.join(tmp, f"v_{t}_{yaw}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", equi, "-vf",
                        f"v360=input=e:output=flat:yaw={yaw}:h_fov={VIEW_HFOV}:v_fov={VIEW_VFOV}:w={VIEW_W}:h={VIEW_H}",
                        out], check=True)
        views.append((yaw, out))
    return views


def face_yaws(det, view_yaw, path, min_h):
    img = cv2.imread(path)
    boxes = det.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf_th=0.7, scales=[0.5, 1.0])
    f = (VIEW_W / 2) / math.tan(math.radians(VIEW_HFOV / 2))
    out = []
    for x1, y1, x2, y2, conf in boxes:
        if (y2 - y1) < min_h:
            continue
        cx = (x1 + x2) / 2
        yaw = view_yaw + math.degrees(math.atan((cx - VIEW_W / 2) / f))
        yaw = (yaw + 180) % 360 - 180
        out.append({"yaw": yaw, "h": float(y2 - y1), "conf": float(conf)})
    return out


def cluster(hits, n_times, gap=12.0, min_ratio=0.5):
    """원형 yaw를 정렬해 gap 이내끼리 묶고, 등장 시점 비율이 낮은 군집(지나가는 사람)은 버린다."""
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: h["yaw"])
    groups = [[hits[0]]]
    for h in hits[1:]:
        if h["yaw"] - groups[-1][-1]["yaw"] <= gap:
            groups[-1].append(h)
        else:
            groups.append([h])
    # -180/180 경계 이어붙이기
    if len(groups) > 1 and (groups[0][0]["yaw"] + 360) - groups[-1][-1]["yaw"] <= gap:
        groups[0] = groups.pop() + groups[0]
    people = []
    for g in groups:
        times = {h["t"] for h in g}
        if len(times) / n_times < min_ratio:
            continue
        ang = np.radians([h["yaw"] for h in g])
        yaw = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean()))
        people.append({"yaw": round(yaw, 1), "face_h": round(float(np.median([h["h"] for h in g])), 1),
                       "seen": len(times)})
    return sorted(people, key=lambda p: p["yaw"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("insv")
    ap.add_argument("out")
    ap.add_argument("--times", default="60,180,300,420,540,660,780,900")
    ap.add_argument("--equirect", action="store_true", help="입력이 보정 스티칭된 equirect 영상(stitch_calibrated.py)")
    ap.add_argument("--min-face", type=float, default=60, help="원근 뷰 기준 최소 얼굴 높이(px)")
    a = ap.parse_args()
    a.out, a.insv = os.path.abspath(a.out), os.path.abspath(a.insv)

    ASD_DIR = asd_dir()
    os.chdir(ASD_DIR)  # S3FD가 cwd 기준으로 가중치를 찾음
    sys.path.insert(0, ASD_DIR)
    from model.faceDetector.s3fd import S3FD
    import torch
    det = S3FD(device="mps" if torch.backends.mps.is_available() else "cpu")

    times = [float(x) for x in a.times.split(",")]
    hits = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            for yaw, path in grab_views(a.insv, t, tmp, a.equirect):
                for h in face_yaws(det, yaw, path, a.min_face):
                    h["t"] = t
                    hits.append(h)
    # 45도 간격 + 90도 화각이라 같은 얼굴이 인접 뷰 두 장에 잡힐 수 있음 → 군집화가 흡수
    people = cluster(hits, len(times))
    for i, p in enumerate(people):
        p["id"] = chr(ord("A") + i)
    json.dump({"insv": a.insv, "lens_fov": LENS_FOV, "equirect": a.equirect, "people": people}, open(a.out, "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(people, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
