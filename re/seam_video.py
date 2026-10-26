"""M15-2: Studio vs multicam360 이음새 구간 나란히 비교 영상(육안 판정용).
Studio 영상은 자체 안정화로 프레임마다 방향이 달라지므로 프레임마다 특징점 회전을 추정(5프레임 평활)해 우리 좌표에 맞춘 뒤, 양쪽 렌즈 경계(경도 ±90°) 둘레를 잘라
2배 확대해 [Studio | multicam360] 가로, 위=왼쪽 이음새·아래=오른쪽 이음새로 쌓는다.
usage: python re/seam_video.py <studio.mp4> <ours.mp4> <out.mp4> --start 10 --dur 10 [--offset 10]"""
import argparse, os, subprocess, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_compare import H, W, estimate_rotation, grab, remap_rot

CW, CH, ZOOM = 560, 420, 2           # 크롭(3840폭 기준 px) · 확대


def frames(path, start, dur):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
                          "-vf", f"scale={W}:{H}:flags=area", "-r", "30000/1001", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.wait()


def label(img, text):
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("studio"); ap.add_argument("ours"); ap.add_argument("out")
    ap.add_argument("--start", type=float, default=10); ap.add_argument("--dur", type=float, default=10)
    ap.add_argument("--offset", type=float, default=10, help="ours 영상이 원본 몇 초부터인지")
    a = ap.parse_args()
    # 1차: 프레임별 회전(studio 방향 = R · ours 방향) 추정
    Rs = []
    for st, ou in zip(frames(a.studio, a.start, a.dur), frames(a.ours, a.start - a.offset, a.dur)):
        R, inl, n = estimate_rotation(st, ou, iters=200)
        Rs.append(R if inl > 40 else (Rs[-1] if Rs else np.eye(3)))
    print(f"회전 {len(Rs)}프레임 추정, 마지막 inlier {inl}/{n}", flush=True)
    # 5프레임 이동평균 뒤 직교화(극분해) — 추정 지터 제거
    def polar(M):
        U, _, Vt = np.linalg.svd(M); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); return U @ D @ Vt
    Rs = [polar(np.mean(Rs[max(0, i - 2): i + 3], 0)) for i in range(len(Rs))]
    crops = [(W // 4, H // 2), (3 * W // 4, H // 2)]            # 왼쪽·오른쪽 렌즈 경계
    ow, oh = CW * ZOOM * 2, CH * ZOOM * 2
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
                            "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    k = 0
    for st, ou in zip(frames(a.studio, a.start, a.dur), frames(a.ours, a.start - a.offset, a.dur)):
        st = remap_rot(st, Rs[min(k, len(Rs) - 1)])               # Studio를 우리 좌표로
        rows = []
        for cx, cy in crops:
            tiles = []
            for img, name in ((st, "Insta360 Studio"), (ou, "multicam360")):
                c = img[cy - CH // 2: cy + CH // 2, cx - CW // 2: cx + CW // 2]
                tiles.append(label(cv2.resize(c, (CW * ZOOM, CH * ZOOM), interpolation=cv2.INTER_CUBIC), name))
            rows.append(np.hstack(tiles))
        enc.stdin.write(np.vstack(rows).tobytes())
        k += 1
    enc.stdin.close(); enc.wait()
    print(f"{k} 프레임 → {a.out}")


if __name__ == "__main__":
    main()
