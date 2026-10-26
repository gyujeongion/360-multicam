"""공장 보정 + 겹침 띠 옵티컬 플로우로 근거리 이음새까지 맞춘 equirect 스티처.

stitch_calibrated.py(ffmpeg remap, 최근접·플로우 없음)의 상위 버전.
- 투영: insta_calib (Studio OmniProjectionPro 재구현), cv2.remap 선형 보간
- 이음새: 두 렌즈가 모두 보이는 띠에서 DIS 옵티컬 플로우 F(I0→I1)를 구하고
  out(p) = (1-a)·I0(p - a·F) + a·I1(p + (1-a)·F)  (a = 렌즈1 가중치) 로 서로 끌어당겨 블렌딩.
  플로우는 시간축 EMA로 평활해 깜빡임을 줄인다.

usage: python stitch_flow.py <insv|구간.mkv> <out.mp4> --meta <트레일러> [--start --dur --width 7680]
       python stitch_flow.py <l0.png> <out.jpg> --still <l1.png> --meta ...   (스틸 테스트)
"""
import argparse, hashlib, os, subprocess, sys

import cv2
import numpy as np

from .calib import offset_v6_from_metadata, parse_offset_v6, read_metadata_bytes
from .stitch_ff import build_maps

DEC_COLOR = "scale=in_color_matrix=bt709:in_range=tv"   # engine.DEC_COLOR와 같음(M15)

FLOW_SCALE = 0.5   # 플로우 계산 해상도
FLOW_EMA = 0.6     # 이전 프레임 플로우 비중


class FlowStitcher:
    def __init__(self, lenses, W, stream_w=3840, cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")):
        self.W, self.H, self.S = W, W // 2, stream_w
        key = hashlib.sha1((repr([(l["xi"], tuple(l["f"]), tuple(l["c"]), tuple(l["rot"]), tuple(l["dist"]))
                                  for l in lenses]) + str(W)).encode()).hexdigest()[:12]
        cpath = os.path.join(cache_dir, f"maps_{key}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath); mx0, my0, mx1, my1, alpha = (z[k] for k in ("mx0", "my0", "mx1", "my1", "alpha"))
        else:
            (m0, m1), alpha = build_maps(lenses, self.W, self.H, stream_w)
            mx0, my0, mx1, my1 = (a.astype(np.float32) for a in (m0[0], m0[1], m1[0], m1[1]))
            alpha = alpha.astype(np.float32)
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cpath, mx0=mx0, my0=my0, mx1=mx1, my1=my1, alpha=alpha)
        self.alpha = alpha
        # 전체 화면: 가중치 큰 렌즈 하나만 샘플 (hstack 좌표) → remap 1회
        use1 = alpha >= 0.5
        self.full = cv2.convertMaps(np.where(use1, mx1, mx0), np.where(use1, my1, my0), cv2.CV_16SC2)
        self.mx = (mx0, mx1); self.my = (my0, my1)
        band = (alpha > 0.001) & (alpha < 0.999)
        # 이음새 원은 양 극을 지나 극지방에선 띠가 전폭으로 퍼짐 → 플로우는 위도 ±55°(사람 높이)만, 극은 페더만
        lat_lim = int(self.H * (90 - 55) / 180)
        self.regions = []  # (y0, y1, x0, x1, use_flow)
        for r0, r1, flow in ((0, lat_lim, False), (lat_lim, self.H - lat_lim, True), (self.H - lat_lim, self.H, False)):
            for c0 in range(r0, r1, 256):
                c1 = min(c0 + 256, r1)
                cols = np.where(band[c0:c1].any(0))[0]
                if not len(cols):
                    continue
                for g in np.split(cols, np.where(np.diff(cols) > 1)[0] + 1):
                    self.regions.append((c0, c1, max(g[0] - 16, 0), min(g[-1] + 17, self.W), flow))
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev = {}

    def __call__(self, pair, use_flow=True):
        """pair = 두 렌즈 hstack 이미지 (S, 2S, 3) uint8"""
        out = cv2.remap(pair, *self.full, cv2.INTER_LINEAR)
        for i, (y0, y1, x0, x1, flow) in enumerate(self.regions):
            a = self.alpha[y0:y1, x0:x1]
            I0 = cv2.remap(pair, self.mx[0][y0:y1, x0:x1], self.my[0][y0:y1, x0:x1], cv2.INTER_LINEAR)
            I1 = cv2.remap(pair, self.mx[1][y0:y1, x0:x1], self.my[1][y0:y1, x0:x1], cv2.INTER_LINEAR)
            if flow and use_flow:
                g0 = cv2.resize(cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY), None, fx=FLOW_SCALE, fy=FLOW_SCALE)
                g1 = cv2.resize(cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY), None, fx=FLOW_SCALE, fy=FLOW_SCALE)
                # 두 렌즈 노출 차이가 플로우를 흔들지 않게 밝기 정규화
                g1 = np.clip((g1.astype(np.float32) - g1.mean()) * (g0.std() / (g1.std() + 1e-6)) + g0.mean(),
                             0, 255).astype(np.uint8)
                F = cv2.resize(self.dis.calc(g0, g1, None), (x1 - x0, y1 - y0)) / FLOW_SCALE
                if i in self.prev:
                    F = FLOW_EMA * self.prev[i] + (1 - FLOW_EMA) * F
                self.prev[i] = F
                gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
                I0 = cv2.remap(I0, gx - a * F[..., 0], gy - a * F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                I1 = cv2.remap(I1, gx + (1 - a) * F[..., 0], gy + (1 - a) * F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            a3 = a[..., None]
            blended = (I0 * (1 - a3) + I1 * a3).astype(np.uint8)
            m = ((a > 0.001) & (a < 0.999))[..., None]
            out[y0:y1, x0:x1] = np.where(m, blended, out[y0:y1, x0:x1])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--still", help="스틸 테스트: 렌즈1 이미지(src는 렌즈0 이미지)")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--bitrate", default="120M")
    a = ap.parse_args()

    lenses = parse_offset_v6(offset_v6_from_metadata(read_metadata_bytes(a.meta)))
    st = FlowStitcher(lenses, a.width)

    if a.still:
        pair = np.hstack([cv2.imread(a.src), cv2.imread(a.still)])
        cv2.imwrite(a.out, st(pair, use_flow=not a.no_flow))
        return

    S = 3840
    dec = ["ffmpeg", "-v", "error", "-ss", str(a.start)] + (["-t", str(a.dur)] if a.dur else []) + \
          ["-i", a.src, "-filter_complex", f"[0:0][0:1]hstack,{DEC_COLOR},format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"]
    enc = ["ffmpeg", "-v", "error", "-stats", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{st.W}x{st.H}",
           "-r", "30000/1001", "-i", "-", "-ss", str(a.start)] + (["-t", str(a.dur)] if a.dur else []) + \
          ["-i", a.src, "-map", "0:v", "-map", "1:a:0", "-c:v", "hevc_videotoolbox", "-b:v", a.bitrate,
           "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "256k", "-shortest", a.out]
    pd = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=10 ** 8)
    pe = subprocess.Popen(enc, stdin=subprocess.PIPE, bufsize=10 ** 8)
    fsz = S * 2 * S * 3
    n = 0
    while True:
        buf = pd.stdout.read(fsz)
        if len(buf) < fsz:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(S, 2 * S, 3)
        pe.stdin.write(st(fr, use_flow=not a.no_flow).tobytes())
        n += 1
    pe.stdin.close(); pe.wait(); pd.wait()
    print(f"{n} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
