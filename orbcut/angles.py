"""people.json의 인물별 yaw로 고정 앵글 영상을 렌더한다 (디코딩 1회 → split → 인물 수만큼 출력).

dfisheye → flat 한 번에 투영해 재샘플링을 1회로 줄인다.
usage: python render_angles.py <people.json> <out_dir> [--start 240 --dur 180 --hfov 70 --pitch 8]
화각: --hfov 전역값(50=타이트, 70=기본, 100+=광각). people.json 인물 항목에 "hfov"/"pitch"를 넣으면 그 인물만 덮어씀.
"""
import argparse, json, os, subprocess

from .reframe import rect_vfov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("people")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--hfov", type=float, default=70, help="수평 화각(도). 50 타이트 / 70 기본 / 100+ 광각")
    ap.add_argument("--pitch", type=float, default=8, help="상하 각도(도, +가 위). 가까운 인물 머리 잘림 보정")
    ap.add_argument("--bitrate", default="12M", help="4K 마스터면 40M 이상")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    a = ap.parse_args()

    meta = json.load(open(a.people))
    people, fov = meta["people"], meta["lens_fov"]
    os.makedirs(a.out_dir, exist_ok=True)

    eq = meta.get("equirect", False)
    src_in = "[0:v]" if eq else "[0:0][0:1]hstack,"
    proj_in = "input=e" if eq else f"input=dfisheye:ih_fov={fov}:iv_fov={fov}"
    chains = [f"{src_in}split={len(people)}" + "".join(f"[s{i}]" for i in range(len(people)))]
    outs = []
    for i, p in enumerate(people):
        hfov, pitch = p.get("hfov", a.hfov), p.get("pitch", a.pitch)
        vfov = round(rect_vfov(hfov, a.w, a.h), 3)
        chains.append(f"[s{i}]v360={proj_in}:output=flat:interp=lanczos:"
                      f"yaw={p['yaw']}:pitch={pitch}:h_fov={hfov}:v_fov={vfov}:w={a.w}:h={a.h}[v{i}]")
        outs += ["-map", f"[v{i}]", "-map", "0:a:0", "-c:v", "hevc_videotoolbox", "-b:v", a.bitrate,
                 "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "192k",
                 os.path.join(a.out_dir, f"cam_{p['id']}.mp4")]

    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-ss", str(a.start)]
    if a.dur:
        cmd += ["-t", str(a.dur)]
    cmd += ["-i", meta["insv"], "-filter_complex", ";".join(chains)] + outs
    subprocess.run(cmd, check=True)
    # 후속 단계(얼굴 좌표 → 각도 역산)가 실제 렌더 화각을 쓰도록 기록. 인물별 덮어쓰기는 people.json에 남음
    json.dump({"hfov": a.hfov, "vfov": rect_vfov(a.hfov, a.w, a.h), "pitch": a.pitch, "w": a.w, "h": a.h,
               "src": meta["insv"]}, open(os.path.join(a.out_dir, "render.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
