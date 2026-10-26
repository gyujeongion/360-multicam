"""insv 공장 보정값으로 equirect 360 영상을 만든다 (Insta360 Studio 내보내기 대체).

- 보정/투영: insta_calib.py (Studio OmniProjectionPro 재구현)
- ffmpeg: 두 렌즈 hstack → 렌즈별 remap → 겹침 띠 maskedmerge 페더 블렌딩
- 회전 규칙: 렌즈1 상대회전 = R1ᵀ·R0 (오일러 zxy, 도) + Y축 180° — 겹침 띠 MAE 탐색으로 선정(re/search_convention.py)

usage: python stitch_calibrated.py <insv> <out.mp4> [--start 240 --dur 180 --width 7680 --meta tail.bin]
"""
import argparse, os, subprocess, tempfile

import numpy as np

from .calib import euler, offset_v6_from_metadata, omni_project, parse_offset_v6, read_metadata_bytes

CROP, STREAM_SCALE = 32, 0.5  # 보정 좌표(7744) → 센서 크롭 7680 → 스트림 3840
FEATHER = 0.12  # 광축 cos 기준 페더 폭


def build_maps(lenses, W, H, stream_w):
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = (np.arange(H) + 0.5) / H * np.pi - np.pi / 2
    LON, LAT = np.meshgrid(lon, lat)
    D = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1).reshape(-1, 3)
    R = [euler(*l["rot"], "zxy").T for l in lenses]
    M = [np.eye(3), R[1] @ R[0].T @ np.diag([-1.0, 1, -1])]
    maps, weights = [], []
    for k, (L, Mk) in enumerate(zip(lenses, M)):
        rays = D @ Mk.T
        px, valid = omni_project(rays, L)
        lx = (px[:, 0] - k * L["size"][1] - CROP) * STREAM_SCALE
        ly = (px[:, 1] - CROP) * STREAM_SCALE
        inside = valid & (lx >= 0) & (lx < stream_w - 1) & (ly >= 0) & (ly < stream_w - 1)
        z = rays[:, 2] / np.linalg.norm(rays, axis=1)
        w = np.clip((z - np.cos(L["max_fov"])) / FEATHER, 0, 1) * inside
        maps.append((np.clip(lx + k * stream_w, 0, 2 * stream_w - 1).reshape(H, W),  # hstack 좌표
                     np.clip(ly, 0, stream_w - 1).reshape(H, W)))
        weights.append(w.reshape(H, W))
    alpha = weights[1] / (weights[0] + weights[1] + 1e-6)  # maskedmerge: 0=렌즈0, 1=렌즈1
    return maps, alpha


def write_pgm(path, arr, maxval):
    a = np.round(arr).astype(">u2" if maxval > 255 else "u1")
    with open(path, "wb") as f:
        f.write(f"P5 {arr.shape[1]} {arr.shape[0]} {maxval}\n".encode())
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("insv"); ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--meta", help="트레일러 끝부분 파일(원본이 원격일 때)")
    ap.add_argument("--bitrate", default="120M")
    a = ap.parse_args()

    lenses = parse_offset_v6(offset_v6_from_metadata(read_metadata_bytes(a.meta or a.insv)))
    W, H = a.width, a.width // 2
    (m0, m1), alpha = build_maps(lenses, W, H, 3840)
    tmp = tempfile.mkdtemp(prefix="stitch_")
    for name, arr, mx in [("x0", m0[0], 65535), ("y0", m0[1], 65535), ("x1", m1[0], 65535),
                          ("y1", m1[1], 65535), ("alpha", alpha * 255, 255)]:
        write_pgm(os.path.join(tmp, f"{name}.pgm"), arr, mx)

    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-ss", str(a.start)]
    if a.dur:
        cmd += ["-t", str(a.dur)]
    cmd += ["-i", a.insv] + sum([["-i", os.path.join(tmp, f"{n}.pgm")] for n in ("x0", "y0", "x1", "y1", "alpha")], [])
    fc = ("[0:0][0:1]hstack,format=gbrp10le,split[h0][h1];"
          "[1:v]format=gray16[x0];[2:v]format=gray16[y0];[3:v]format=gray16[x1];[4:v]format=gray16[y1];"
          "[5:v]format=gray,format=gbrp10le[m];"
          "[h0][x0][y0]remap[r0];[h1][x1][y1]remap[r1];"
          "[r0][r1][m]maskedmerge=planes=7,format=yuv420p10le[v]")
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "0:a:0", "-c:v", "hevc_videotoolbox",
            "-b:v", a.bitrate, "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "256k",
            "-metadata:s:v", "spherical=equirectangular", a.out]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
