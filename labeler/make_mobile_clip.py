"""라벨링용 모바일 영상: 파노라마에서 인물별 얼굴 영역을 잘라 3열 타일로(이름표는 라벨 도구가 영상 위에 그림, 순서 = people.json) — 폰에서 입 모양으로 화자 판단.
usage: python labeler/make_mobile_clip.py <out_dir>   (out_dir/panorama.mp4, people.json → out_dir/label_mobile.mp4, ~10MB/3분)"""
import json, os, subprocess, sys

TILE, CROP, CY = 320, 560, 1010          # 타일 px, 파노라마에서 자를 정사각형 px(약 52°), 얼굴 높이 중심(적도 약간 아래)


def main(out):
    P = json.load(open(os.path.join(out, "people.json")))["people"]
    src = os.path.join(out, "panorama.mp4")
    W, H = [int(x) for x in subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                     "stream=width,height", "-of", "csv=p=0", src]).decode().split(",")]
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src]))
    n = len(P)
    fc = ["[0:v]split=2[a][b]", f"[b]crop={CROP}:{H}:0:0[w]",            # 180° 경계를 넘는 인물용으로 앞부분을 뒤에 붙임
          f"[a][w]hstack=2,split={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i, p in enumerate(P):
        x = (int((p["yaw"] + 180) / 360 * W) - CROP // 2) % W
        fc.append(f"[s{i}]crop={CROP}:{CROP}:{x}:{CY - CROP // 2},scale={TILE}:{TILE}[t{i}]")   # 이름표는 HTML이 겹쳐 그림
    tiles = "".join(f"[t{i}]" for i in range(n))
    cells = -(-n // 3) * 3
    for k in range(cells - n):
        fc.append(f"color=c=black:s={TILE}x{TILE}:d={dur:.2f}[k{k}]"); tiles += f"[k{k}]"
    layout = "|".join(f"{(k % 3) * TILE}_{(k // 3) * TILE}" for k in range(cells))
    fc.append(f"{tiles}xstack=inputs={cells}:layout={layout}:shortest=1[v]")
    dst = os.path.join(out, "label_mobile.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "0:a:0",
                    "-c:v", "libx264", "-preset", "slow", "-b:v", "420k", "-maxrate", "600k", "-bufsize", "1200k",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "24", "-c:a", "aac", "-b:a", "64k", "-ac", "1", dst],
                   check=True)
    print(dst, os.path.getsize(dst))


if __name__ == "__main__":
    main(sys.argv[1])
