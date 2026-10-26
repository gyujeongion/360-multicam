"""통합 렌더 엔진: 출력 픽셀별 월드 방향 → 공장보정 렌즈 좌표 샘플링 + 겹침띠 플로우 블렌딩.

equirect·인물 앵글·4K 마스터가 모두 이 엔진 하나로 원본(듀얼 어안)에서 직접 렌더된다.
월드 좌표: x 오른쪽, y 아래, z 정면(렌즈0 광축). 경도(yaw)는 +x 쪽이 +, 위도(pitch)는 위가 +
— ffmpeg v360(yaw/pitch)과 같은 규약이라 people.json 각도를 그대로 쓴다.
"""
import math, os, subprocess

import cv2
import numpy as np

from .calib import euler, offset_v6_from_metadata, omni_project, parse_offset_v6, read_metadata_bytes

CROP, STREAM_SCALE = 32, 0.5      # 보정 좌표(7744) → 센서 크롭 7680 → 스트림 3840
FEATHER = float(os.environ.get("MULTICAM360_FEATHER", "0.12"))   # 광축 cos 기준 페더 폭
# 어안 → 출력 보간. 선형은 저역통과처럼 뭉개져 Studio 대비 선명도 72%(본체)·53%(이음새)로 측정됨(M15) → 큐빅 기본
INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[
    os.environ.get("MULTICAM360_INTERP", "cubic")]
# 인코딩 전 마감 필터(ffmpeg -vf). 노이즈가 많으면 인코더가 비트를 노이즈에 써서 디테일을 잃는다(M15 실측:
# 프레임 선명도 Studio 대비 1.24 → 인코딩 후 0.87). 빈 값이면 끔
# M15 실측(Studio 대비): 약한 시간축 디노이즈 + 약한 언샤프가 인코딩 손실을 가장 잘 보완
FINISH_VF = os.environ.get("MULTICAM360_FINISH_VF", "hqdn3d=6:4.5:9:6.75,unsharp=5:5:0.6:5:5:0")


# 색 변환은 BT.709로 고정: 기본값(BT.601)으로 풀고 다시 묶으면 밝기·색이 틀어지고 태그도 비어
# Studio 대비 밝기 −4로 측정됐다(M15). 원본 insv·Studio 출력은 BT.709·TV 범위
DEC_COLOR = "scale=in_color_matrix=bt709:in_range=tv"
ENC_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


TEN_BIT = os.environ.get("MULTICAM360_10BIT", "1") == "1"      # Studio 출력과 같은 10비트(main10)
ENC_Q = os.environ.get("MULTICAM360_ENC_Q", "70")                 # 품질 고정(VideoToolbox -q:v 1~100). 빈 값이면 비트레이트 지정


def _enc_vf():
    fmt = "p010le" if TEN_BIT else "yuv420p"
    vf = ",".join(x for x in (FINISH_VF, "scale=out_color_matrix=bt709:out_range=tv", f"format={fmt}") if x)
    return ["-vf", vf] + ENC_TAGS + (["-profile:v", "main10"] if TEN_BIT else [])


def _enc_rate(br):
    return ["-q:v", ENC_Q] if ENC_Q else ["-b:v", br]
FLOW_SCALE = 0.5
FLOW_EMA = float(os.environ.get("MULTICAM360_FLOW_EMA", "0.8"))   # M15-3: 0.6→0.8, 프레임 간 플로우 변화 6.1→0.8px
FLOW_MAX_ELEV = float(os.environ.get("MULTICAM360_FLOW_MAX_ELEV", "55"))   # 이 고도 밖(천장·바닥 극)은 플로우 생략, 페더만
FLOW_TILE = int(os.environ.get("MULTICAM360_FLOW_TILE", "0"))              # 플로우 영역 세로 분할(0 = 세로 전체 한 스트립, M15-2 실측 최선)
# 이음새 합성 방식: feather = 선형 알파 / band2 = 저주파 넓게·고주파 좁게 + 플로우 신뢰도 낮은 곳은 한쪽 렌즈만
SEAM_MODE = os.environ.get("MULTICAM360_SEAM_MODE", "band2")
BAND2_K = float(os.environ.get("MULTICAM360_BAND2_K", "10"))       # 고주파 알파 기울기(클수록 좁은 전환)
BAND2_SIGMA = float(os.environ.get("MULTICAM360_BAND2_SIGMA", "6"))  # 저/고주파 분리 가우시안 σ(px)
CONF_FB = float(os.environ.get("MULTICAM360_CONF_FB", "1.5"))      # 전후방 플로우 불일치 허용(px, 출력 해상도)
CONF_RES = float(os.environ.get("MULTICAM360_CONF_RES", "24"))     # 정렬 후 밝기 잔차 허용(0~255)
FLOW_SMOOTH = float(os.environ.get("MULTICAM360_FLOW_SMOOTH", "32"))   # 플로우 공간 평활 σ(px, 출력 해상도). 0=끔. 패치 단위 물결(워프 출렁임) 억제
FLOW_VCLAMP = float(os.environ.get("MULTICAM360_FLOW_VCLAMP", "3"))   # 수직 플로우 성분 상한(px). 시차는 거의 수평이라 수직은 보정 잔차 수준만. 0=끔
FLOW_MAX = float(os.environ.get("MULTICAM360_FLOW_MAX", "0"))         # 플로우 크기 상한(px). 0=끔
EMA_ADAPT = float(os.environ.get("MULTICAM360_EMA_ADAPT", "0"))    # >0이면 플로우 변화량(px)에 따라 EMA 감쇠(σ). 기본 0 — DIS 잡음(~6px)이 σ를 넘어 평활이 꺼지던 실측


# ---------- 방향 격자 ----------

def equirect_dirs(W):
    H = W // 2
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = (np.arange(H) + 0.5) / H * np.pi - np.pi / 2          # 위쪽 행 = 음수 = 위
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)


def flat_dirs(yaw, pitch, hfov, w, h):
    """직선 원근(rectilinear) 화면. 세로 화각은 hfov와 화면비로 정확히 결정."""
    f = (w / 2) / math.tan(math.radians(hfov / 2))
    xs = (np.arange(w) + 0.5 - w / 2) / f
    ys = (np.arange(h) + 0.5 - h / 2) / f
    X, Y = np.meshgrid(xs, ys)
    r = np.stack([X, Y, np.ones_like(X)], -1)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return _orient(r, yaw, pitch)[0]   # 위로 들기(pitch+) 후 +x 쪽으로 돌기(yaw+)


def _orient(r, yaw, pitch):
    p, y = math.radians(pitch), math.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return r @ (Ry @ Rx).T, Ry @ Rx


def pannini_scale(hfov, w, d):
    """파니니 화면에서 가장자리(±hfov/2)가 화면 폭 끝에 오도록 하는 픽셀 배율."""
    phi = math.radians(hfov / 2)
    xmax = (d + 1) / (d + math.cos(phi)) * math.sin(phi)
    return (w / 2) / xmax


def pannini_dirs(yaw, pitch, hfov, w, h, d=0.7):
    """파니니 투영 화면 (Sharpless 일반식). d=0 → 직선 원근, d=1 → 표준 파니니.
    세로선은 곧게 유지하고 가로 가장자리 늘어짐을 줄인다 — 넓은 화각 인물 화면용."""
    f = pannini_scale(hfov, w, d)
    xs = (np.arange(w) + 0.5 - w / 2) / f
    ys = (np.arange(h) + 0.5 - h / 2) / f
    X, Y = np.meshgrid(xs, ys)
    k = X ** 2 / (d + 1) ** 2
    disc = np.clip(k ** 2 * d ** 2 - (k + 1) * (k * d ** 2 - 1), 0, None)
    cphi = (-k * d + np.sqrt(disc)) / (k + 1)
    S = (d + 1) / (d + cphi)
    phi = np.arctan2(X, S * cphi)
    theta = np.arctan(Y / S)                       # y 아래 + (화면 좌표)
    r = np.stack([np.sin(phi) * np.cos(theta), np.sin(theta), np.cos(phi) * np.cos(theta)], -1)
    return _orient(r, yaw, pitch)[0]


def pannini_project(cam_dir, hfov, w, h, d=0.7):
    """카메라 좌표 방향 → 파니니 화면 픽셀 (pannini_dirs의 역)."""
    x, y, z = cam_dir
    phi = math.atan2(x, z)
    theta = math.atan2(y, math.hypot(x, z))
    S = (d + 1) / (d + math.cos(phi))
    f = pannini_scale(hfov, w, d)
    return w / 2 + f * S * math.sin(phi), h / 2 + f * S * math.tan(theta)


def dir_to_yaw_pitch(d):
    return math.degrees(math.atan2(d[0], d[2])), math.degrees(-math.asin(d[1] / np.linalg.norm(d)))


# ---------- 렌즈 ----------

def load_lenses(meta_path):
    return parse_offset_v6(offset_v6_from_metadata(read_metadata_bytes(meta_path)))


def lens_rotations(lenses):
    """렌즈별 월드→카메라 회전. 규칙 근거: re/README.md (오일러 zxy, 렌즈1 = 상대회전 + Y축 180°).
    z(약 90°, 센서 장착각)는 두 렌즈에 렌즈1 값을 같이 쓴다 — 렌즈별 z를 그대로 쓰면 두 z 차이(X6 0.77°)만큼
    렌즈1이 광축 둘레로 돌아 이음새에 세로 단차가 남았다. 9개 클립 45프레임 실측 잔차 −0.796°(클립별 −0.78~−0.82)와
    이 규칙의 보정량 −0.765°가 0.03° 안에서 일치하고, 기울기 성분은 0(re/lens_roll_est.py, 2026-10-26)."""
    z = lenses[1]["rot"][2]
    R = [euler(l["rot"][0], l["rot"][1], z, "zxy").T for l in lenses]
    M1 = R[1] @ R[0].T @ np.diag([-1.0, 1, -1])
    corr = os.environ.get("MULTICAM360_LENS1_CORR")            # 실험용: "롤,x,y"(도) — 렌즈1 샘플링 방향에 추가 회전
    if corr:
        r, x, y = (float(v) for v in corr.split(","))
        M1 = M1 @ euler(x, y, r, "xyz")
    return [np.eye(3), M1]


class Projector:
    """하나의 출력 화면(방향 격자)에 대한 샘플링 맵·블렌딩 가중치·플로우 영역."""

    def __init__(self, lenses, dirs, stream_w=3840, flow=True, feather=None):
        feather = FEATHER if feather is None else feather
        self.h, self.w = dirs.shape[:2]
        self.S, self.flow_on = stream_w, flow
        scale = STREAM_SCALE * stream_w / 3840     # 입력이 축소 프레임이면 좌표도 같이 축소
        D = dirs.reshape(-1, 3)
        mx, my, wts = [], [], []
        for k, (L, M) in enumerate(zip(lenses, lens_rotations(lenses))):
            rays = D @ M.T
            px, valid = omni_project(rays, L)
            lx = (px[:, 0] - k * L["size"][1] - CROP) * scale
            ly = (px[:, 1] - CROP) * scale
            inside = valid & (lx >= 0) & (lx < stream_w - 1) & (ly >= 0) & (ly < stream_w - 1)
            z = rays[:, 2] / np.linalg.norm(rays, axis=1)
            wts.append((np.clip((z - np.cos(L["max_fov"])) / feather, 0, 1) * inside).reshape(self.h, self.w))
            mx.append(np.clip(lx + k * stream_w, 0, 2 * stream_w - 1).reshape(self.h, self.w).astype(np.float32))
            my.append(np.clip(ly, 0, stream_w - 1).reshape(self.h, self.w).astype(np.float32))
        self.alpha = (wts[1] / (wts[0] + wts[1] + 1e-6)).astype(np.float32)   # 0=렌즈0, 1=렌즈1
        use1 = self.alpha >= 0.5
        self.full = cv2.convertMaps(np.where(use1, mx[1], mx[0]), np.where(use1, my[1], my[0]), cv2.CV_16SC2)
        self.mx, self.my = mx, my
        band = (self.alpha > 0.001) & (self.alpha < 0.999)
        elev = np.degrees(np.abs(np.arcsin(np.clip(dirs[..., 1], -1, 1))))
        self.regions = []                     # (y0, y1, x0, x1, flow)
        self.zone = (elev < FLOW_MAX_ELEV)    # 영역이 겹치면(고도 경계 타일·세로 스트립) 플로우 결과를 페더가 덮지 않게 구역으로 가른다
        for flow_zone in (True, False):
            zone = band & (self.zone == flow_zone)
            if FLOW_TILE:
                row_spans = [(r0, min(r0 + FLOW_TILE, self.h)) for r0 in range(0, self.h, FLOW_TILE)]
            else:                                  # 세로 스트립: 그 구역이 실제로 차지하는 행 범위만(극 쪽 행은 페더 구역으로)
                rows = np.where(zone.any(1))[0]
                row_spans = [(g[0], g[-1] + 1) for g in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)] if len(rows) else []
            for r0, r1 in row_spans:
                cols = np.where(zone[r0:r1].any(0))[0]
                if not len(cols):
                    continue
                for g in np.split(cols, np.where(np.diff(cols) > 1)[0] + 1):
                    self.regions.append((r0, r1, max(g[0] - 16, 0), min(g[-1] + 17, self.w), flow_zone))
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if flow else None
        self.prev = {}

    def __call__(self, pair):
        """pair = 두 렌즈 hstack (S, 2S, 3) uint8 → 출력 (h, w, 3) uint8"""
        out = cv2.remap(pair, *self.full, INTERP)
        for i, (y0, y1, x0, x1, flow) in enumerate(self.regions):
            a = self.alpha[y0:y1, x0:x1]
            m = (a > 0.001) & (a < 0.999) & (self.zone[y0:y1, x0:x1] == flow)
            if not m.any():
                continue
            m0x, m0y = self.mx[0][y0:y1, x0:x1], self.my[0][y0:y1, x0:x1]
            m1x, m1y = self.mx[1][y0:y1, x0:x1], self.my[1][y0:y1, x0:x1]
            conf = None
            if flow and self.flow_on:
                I0 = cv2.remap(pair, m0x, m0y, cv2.INTER_LINEAR)        # 플로우 추정용(저해상도로 줄여 씀)
                I1 = cv2.remap(pair, m1x, m1y, cv2.INTER_LINEAR)
                g0 = cv2.resize(cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY), None, fx=FLOW_SCALE, fy=FLOW_SCALE)
                g1 = cv2.resize(cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY), None, fx=FLOW_SCALE, fy=FLOW_SCALE)
                if min(g0.shape) >= 16:
                    # 두 렌즈 노출 차이가 플로우를 흔들지 않게 밝기 정규화
                    g1 = np.clip((g1.astype(np.float32) - g1.mean()) * (g0.std() / (g1.std() + 1e-6)) + g0.mean(),
                                 0, 255).astype(np.uint8)
                    Fs = self.dis.calc(g0, g1, None)
                    if FLOW_SMOOTH > 0:
                        Fs = cv2.GaussianBlur(Fs, (0, 0), FLOW_SMOOTH * FLOW_SCALE)
                    if FLOW_VCLAMP > 0:
                        Fs[..., 1] = np.clip(Fs[..., 1], -FLOW_VCLAMP * FLOW_SCALE, FLOW_VCLAMP * FLOW_SCALE)
                    if FLOW_MAX > 0:
                        mag = np.hypot(Fs[..., 0], Fs[..., 1]); lim = FLOW_MAX * FLOW_SCALE
                        Fs *= np.where(mag > lim, lim / (mag + 1e-6), 1.0)[..., None]
                    F = cv2.resize(Fs, (x1 - x0, y1 - y0)) / FLOW_SCALE
                    if SEAM_MODE == "band2":
                        conf = self._confidence(g0, g1, Fs, (x1 - x0, y1 - y0))
                    if i in self.prev:
                        P = self.prev[i]
                        if EMA_ADAPT > 0:        # 움직임이 큰 곳은 이전 플로우를 버린다(지연 고스트 방지)
                            d2 = ((F - P) ** 2).sum(-1)
                            w = (FLOW_EMA * np.exp(-d2 / (EMA_ADAPT ** 2)))[..., None]
                            F = w * P + (1 - w) * F
                        else:
                            F = FLOW_EMA * P + (1 - FLOW_EMA) * F
                    self.prev[i] = F
                    gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
                    # 플로우를 좌표 맵에 합성한 뒤 원본에서 한 번만 샘플링 — 두 번 보간하면 이음새가 뭉개짐(M15 실측)
                    w0 = (gx - a * F[..., 0], gy - a * F[..., 1]); w1 = (gx + (1 - a) * F[..., 0], gy + (1 - a) * F[..., 1])
                    m0x, m0y = (cv2.remap(m, *w0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for m in (m0x, m0y))
                    m1x, m1y = (cv2.remap(m, *w1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for m in (m1x, m1y))
            I0 = cv2.remap(pair, m0x, m0y, INTERP)
            I1 = cv2.remap(pair, m1x, m1y, INTERP)
            if SEAM_MODE == "band2":
                blended = self._band2(I0, I1, a, conf, i)
            else:
                a3 = a[..., None]
                blended = (I0 * (1 - a3) + I1 * a3).astype(np.uint8)
            out[y0:y1, x0:x1] = np.where(m[..., None], blended, out[y0:y1, x0:x1])
        return out

    def _confidence(self, g0, g1, Fs, size):
        return cv2.resize(flow_confidence(self.dis, g0, g1, Fs), size)

    def _band2(self, I0, I1, a, conf, i):
        if conf is not None:
            key = ("conf", i)
            if key in self.prev:                      # 신뢰도 맵도 시간축으로 눌러 전환 플리커 억제
                conf = 0.5 * self.prev[key] + 0.5 * conf
            self.prev[key] = conf
        return band2_blend(I0, I1, a, conf)


def flow_confidence(dis, g0, g1, Fs):
    """플로우 신뢰도 0~1 (플로우 해상도). 전후방 불일치가 크거나 정렬 뒤 밝기 잔차가 큰 곳(가림·오대응)은 0."""
    Fb = dis.calc(g1, g0, None)
    gy, gx = np.mgrid[0:g0.shape[0], 0:g0.shape[1]].astype(np.float32)
    Fbw = cv2.remap(Fb, gx + Fs[..., 0], gy + Fs[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb_err = np.hypot(*(Fs + Fbw).transpose(2, 0, 1)) / FLOW_SCALE
    g1w = cv2.remap(g1, gx + Fs[..., 0], gy + Fs[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    res = cv2.GaussianBlur(cv2.absdiff(g0, g1w).astype(np.float32), (0, 0), 1.5)
    c = np.exp(-(fb_err / CONF_FB) ** 2) * np.exp(-(res / CONF_RES) ** 2)
    return cv2.GaussianBlur(c.astype(np.float32), (0, 0), 4)


def band2_blend(I0, I1, a, conf):
    """저주파는 넓은 알파로(노출·비네팅 단차 숨김), 고주파는 좁은 전환으로(잔여 플로우 오차 블러 방지).
    신뢰도 낮은 곳은 고주파를 한쪽 렌즈에서만 가져온다(고스트 대신 한쪽 원본 선명도).
    out = L0(1−a) + L1·a + H0(1−a_hi) + H1·a_hi  =  I0 + (I1−I0)·a_hi + (L1−L0)·(a−a_hi)"""
    h, w = a.shape
    a_hi = np.clip((a - 0.5) * BAND2_K + 0.5, 0, 1)
    if conf is not None:
        a_hi = conf * a_hi + (1 - conf) * (a >= 0.5).astype(np.float32)
    # 저주파는 1/4 해상도에서 블러 후 확대(σ/4) — 같은 결과에 블러 비용 1/16
    small = (max(w // 4, 1), max(h // 4, 1))
    lo = lambda img: cv2.GaussianBlur(cv2.resize(img, small, interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), BAND2_SIGMA / 4)
    dL = cv2.resize(lo(I1) - lo(I0), (w, h), interpolation=cv2.INTER_LINEAR)          # 작은 해상도에서 float 변환
    out = cv2.blendLinear(I0, I1, 1 - a_hi, a_hi).astype(np.float32)                  # I0 + (I1−I0)·a_hi
    out += dL * (a - a_hi)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- 영상 렌더: 디코딩 1회 → 출력 여러 개 ----------

def render_many(src, outputs, start=None, dur=None, fps="30000/1001", stream_w=3840):
    """outputs: [(Projector, out_path, bitrate)]. 원본을 한 번만 디코딩해 모든 출력에 동시에 쓴다."""
    S = stream_w
    dec = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + (["-ss", str(start)] if start is not None else []) + \
          (["-t", str(dur)] if dur is not None else []) + \
          ["-i", src, "-filter_complex", f"[0:0][0:1]hstack,{DEC_COLOR},format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"]
    encs = []
    for pj, path, br in outputs:
        encs.append(subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{pj.w}x{pj.h}", "-r", fps,
             "-i", "-"] + (["-ss", str(start)] if start is not None else []) + (["-t", str(dur)] if dur is not None else []) +
            ["-i", src, "-map", "0:v", "-map", "1:a:0"] + _enc_vf() + ["-c:v", "hevc_videotoolbox"] + _enc_rate(br) + ["-tag:v", "hvc1",
             "-c:a", "aac", "-b:a", "192k", "-shortest", path], stdin=subprocess.PIPE, bufsize=10 ** 8))
    pd = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=10 ** 8)
    fsz, n = S * 2 * S * 3, 0
    try:
        while True:
            buf = pd.stdout.read(fsz)
            if len(buf) < fsz:
                break
            pair = np.frombuffer(buf, np.uint8).reshape(S, 2 * S, 3)
            for (pj, _, _), e in zip(outputs, encs):
                e.stdin.write(pj(pair).tobytes())
            n += 1
    finally:
        for e in encs:
            e.stdin.close()
        rcs = [e.wait() for e in encs]
        pd.wait()
    if any(rcs) or pd.returncode:
        raise RuntimeError(f"인코딩/디코딩 실패 dec={pd.returncode} enc={rcs}")
    return n


# 동적 렌더 이음새: band2 = 카메라에 붙은 이음새 띠에서 플로우 정렬 + 2밴드 합성 / feather = 가중치 페더만
DYN_SEAM = os.environ.get("MULTICAM360_DYN_SEAM", "band2")
SEAM_PPD = 12          # 이음새 띠 플로우 해상도(px/도) — 정적 렌더 플로우 해상도(70°·1920폭 출력의 절반)와 같다
SEAM_HALF = 12         # 띠 반폭(도). 겹침 띠(약 ±7°)를 덮는다
SEAM_MARGIN = 24       # 보이는 열 앞뒤 여유(px, 2°) — DIS 가장자리 오차가 쓰는 곳까지 오지 않게
_STRIP_MAPS = {}


def _strip_dirs(th_deg, ph_deg):
    """이음새 좌표(θ = 렌즈0 광축에 수직인 고리 둘레 각, φ = 고리에서 광축 쪽으로 벗어난 각) → 카메라 좌표 방향."""
    th, ph = np.radians(th_deg), np.radians(ph_deg)
    return np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)


class SeamStrip:
    """카메라 좌표에 고정된 이음새 띠에서 두 렌즈 사이 플로우를 추정한다.

    띠는 카메라에 붙어 있어 뷰가 매 프레임 돌아도 같은 자리 — 정적 렌더처럼 시간축 EMA가 그대로 성립한다.
    열 = θ(이음새를 따라), 행 = φ(이음새를 가로질러 = 두 렌즈 시차 방향). 고리 전체 상태를 들고
    매 프레임 뷰에 보이는 열만 계산한다. 단위는 정적 렌더 플로우(출력 절반 해상도)와 같아 같은 상수를 쓴다.
    """

    def __init__(self, lenses, stream_w=3840):
        p = SEAM_PPD
        self.W, self.H = 360 * p, 2 * SEAM_HALF * p
        key = (stream_w, repr(lenses))
        if key not in _STRIP_MAPS:           # 렌즈 샘플링 맵(2배 해상도로 뽑아 절반으로 줄임)은 뷰끼리 공유
            th = (np.arange(2 * self.W) + 0.5) / (2 * p) - 180
            ph = (np.arange(2 * self.H) + 0.5) / (2 * p) - SEAM_HALF
            D = _strip_dirs(*np.meshgrid(th, ph)).reshape(-1, 3)
            scale = STREAM_SCALE * stream_w / 3840
            maps = []
            for k, (L, M) in enumerate(zip(lenses, lens_rotations(lenses))):
                px, _ = omni_project(D @ M.T, L)
                lx = np.clip((px[:, 0] - k * L["size"][1] - CROP) * scale, 0, stream_w - 1) + k * stream_w
                ly = np.clip((px[:, 1] - CROP) * scale, 0, stream_w - 1)
                sh = (2 * self.H, 2 * self.W)
                maps.append(cv2.convertMaps(np.nan_to_num(lx).reshape(sh).astype(np.float32),
                                            np.nan_to_num(ly).reshape(sh).astype(np.float32), cv2.CV_16SC2))
            th1 = (np.arange(self.W) + 0.5) / p - 180
            ph1 = (np.arange(self.H) + 0.5) / p - SEAM_HALF
            elev = np.degrees(np.arcsin(np.abs(_strip_dirs(*np.meshgrid(th1, ph1))[..., 1])))   # 카메라 세로축 기준 고도
            taper = np.clip((FLOW_MAX_ELEV + 5 - elev) / 10, 0, 1).astype(np.float32)          # 천장·바닥 극은 플로우 끔
            _STRIP_MAPS[key] = (maps, taper)
        self.maps, self.taper = _STRIP_MAPS[key]
        self.F = np.zeros((self.H, self.W, 2), np.float32)
        self.C = np.ones((self.H, self.W), np.float32)
        self.seen = np.zeros(self.W, bool)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def cols_for(self, th_deg):
        """보이는 이음새 점들의 θ → 덮는 열 번호(원형, 연속). 없으면 None."""
        occ = np.zeros(self.W, bool)
        occ[np.floor((th_deg + 180) * SEAM_PPD).astype(int) % self.W] = True
        idx = np.where(occ)[0]
        if not len(idx):
            return None
        gaps = np.diff(np.r_[idx, idx[0] + self.W])          # 원형으로 이어 가장 큰 빈칸 바깥이 덮는 구간
        g = int(np.argmax(gaps))
        start, n = idx[(g + 1) % len(idx)] - SEAM_MARGIN, self.W - int(gaps[g]) + 1 + 2 * SEAM_MARGIN
        return np.arange(start, start + min(n, self.W)) % self.W

    def update(self, pair, cols):
        """현재 프레임에서 cols 열의 플로우·신뢰도를 갱신하고 (F, C) 그 열만 돌려준다."""
        c2 = np.stack([2 * cols, 2 * cols + 1], 1).ravel()
        g = []
        for m1, m2 in self.maps:
            I = cv2.remap(pair, np.ascontiguousarray(m1[:, c2]), np.ascontiguousarray(m2[:, c2]), cv2.INTER_LINEAR)
            g.append(cv2.resize(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), (len(cols), self.H), interpolation=cv2.INTER_AREA))
        g0, g1 = g
        g1 = np.clip((g1.astype(np.float32) - g1.mean()) * (g0.std() / (g1.std() + 1e-6)) + g0.mean(), 0, 255).astype(np.uint8)
        Fs = self.dis.calc(g0, g1, None)
        if FLOW_SMOOTH > 0:
            Fs = cv2.GaussianBlur(Fs, (0, 0), FLOW_SMOOTH * FLOW_SCALE)
        if FLOW_VCLAMP > 0:            # 시차는 이음새를 가로지르는 방향(행) — 따라가는 성분(열)은 보정 잔차 수준만
            Fs[..., 0] = np.clip(Fs[..., 0], -FLOW_VCLAMP * FLOW_SCALE, FLOW_VCLAMP * FLOW_SCALE)
        if FLOW_MAX > 0:
            mag = np.hypot(Fs[..., 0], Fs[..., 1]); lim = FLOW_MAX * FLOW_SCALE
            Fs *= np.where(mag > lim, lim / (mag + 1e-6), 1.0)[..., None]
        C = flow_confidence(self.dis, g0, g1, Fs)
        tp = self.taper[:, cols]
        F = Fs * tp[..., None]
        seen = self.seen[cols][None, :]
        F = np.where(seen[..., None], FLOW_EMA * self.F[:, cols] + (1 - FLOW_EMA) * F, F)
        C = np.where(seen, 0.5 * self.C[:, cols] + 0.5 * C, C)
        self.F[:, cols], self.C[:, cols], self.seen[cols] = F, C, True
        self.last_cols = cols
        return F, C


class DynamicProjector:
    """프레임마다 바뀌는 회전 Q(뷰 좌표 → 카메라 좌표)로 렌더 (안정화·추적용).

    투영 맵은 부드럽게 변하므로 1/MAP_DOWN 해상도로 계산 후 확대 — 계산량 1/16.
    seam="band2": 이음새 플로우는 카메라에 붙은 띠(SeamStrip)에서 추정·평활하고(뷰가 돌아도 이력 유효),
    그 변위를 샘플링 방향에 합성해 원본에서 한 번만 샘플링한 뒤 2밴드로 합성한다(정적 Projector와 같은 방식).
    seam="feather": 가중치 페더만(이력 없는 띄엄띄엄 샘플 — 안정화 앵커 추정 등).
    """
    MAP_DOWN = 4

    def __init__(self, lenses, view_dirs, stream_w=3840, seam=None):
        self.lenses, self.S = lenses, stream_w
        self.h, self.w = view_dirs.shape[:2]
        d = self.MAP_DOWN
        small = cv2.resize(view_dirs.astype(np.float32), (max(2, self.w // d), max(2, self.h // d)),
                           interpolation=cv2.INTER_LINEAR)
        self.small = small / np.linalg.norm(small, axis=-1, keepdims=True)
        self.rots = lens_rotations(lenses)
        self.scale = STREAM_SCALE * stream_w / 3840
        self.strip = SeamStrip(lenses, stream_w) if (seam or DYN_SEAM) == "band2" else None

    def _project(self, D, k):
        L, M = self.lenses[k], self.rots[k]
        rays = D @ M.T
        px, valid = omni_project(rays, L)
        lx = (px[:, 0] - k * L["size"][1] - CROP) * self.scale + k * self.S
        ly = (px[:, 1] - CROP) * self.scale
        z = rays[:, 2] / np.linalg.norm(rays, axis=1)
        return np.nan_to_num(lx), np.nan_to_num(ly), np.clip((z - np.cos(L["max_fov"])) / FEATHER, 0, 1) * valid

    def maps(self, Q, pair=None):
        sh, sw = self.small.shape[:2]
        D = self.small.reshape(-1, 3) @ Q.T                  # 뷰 → 카메라(렌즈0) 좌표
        (x0, y0, w0), (x1, y1, w1) = self._project(D, 0), self._project(D, 1)
        a = (w1 / (w0 + w1 + 1e-6)).astype(np.float32)
        conf = None
        if self.strip is not None and pair is not None:
            band = (a > 0.001) & (a < 0.999)
            th = np.degrees(np.arctan2(D[:, 1], D[:, 0]))
            ph = np.degrees(np.arcsin(np.clip(D[:, 2], -1, 1)))
            cols = self.strip.cols_for(th[band & (np.abs(ph) < SEAM_HALF)])
            if cols is not None and len(cols) >= 16:
                F, C = self.strip.update(pair, cols)
                # θ → 이번에 계산한 열 안의 위치(원형), φ → 행. 띠 밖·계산 안 한 열은 플로우 0·신뢰도 1
                cx = ((th + 180) * SEAM_PPD - 0.5 - cols[0]) % self.strip.W
                cy = (ph + SEAM_HALF) * SEAM_PPD - 0.5
                mx, my = cx.reshape(sh, sw).astype(np.float32), cy.reshape(sh, sw).astype(np.float32)
                Fv = cv2.remap(F, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, 2)
                conf = cv2.remap(C, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
                dth, dph = Fv[:, 0] / SEAM_PPD, Fv[:, 1] / SEAM_PPD     # 띠 px → 도
                # 정적 렌더와 같은 규약: 렌즈0은 −a·F, 렌즈1은 +(1−a)·F 만큼 옮겨 같은 장면 점을 본다
                x0, y0, _ = self._project(_strip_dirs(th - a * dth, ph - a * dph), 0)
                x1, y1, _ = self._project(_strip_dirs(th + (1 - a) * dth, ph + (1 - a) * dph), 1)
        up = lambda v: cv2.resize(v.reshape(sh, sw).astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return [up(x0), up(x1)], [up(y0), up(y1)], up(a), (None if conf is None else up(conf))

    def __call__(self, pair, Q):
        xs, ys, alpha, conf = self.maps(Q, pair)
        self.last = (xs, ys, alpha, conf)          # 측정용(re/dyn_seam_ab.py)
        I0 = cv2.remap(pair, xs[0], ys[0], INTERP)
        I1 = cv2.remap(pair, xs[1], ys[1], INTERP)
        out = cv2.blendLinear(I0, I1, 1 - alpha, alpha)
        if conf is None:
            return out
        m = (alpha > 0.001) & (alpha < 0.999)
        rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
        if not len(rows):
            return out
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        b = band2_blend(I0[y0:y1, x0:x1], I1[y0:y1, x0:x1], alpha[y0:y1, x0:x1], conf[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = np.where(m[y0:y1, x0:x1, None], b, out[y0:y1, x0:x1])
        return out


def render_dynamic(src, outputs, rot_fn, start=None, dur=None, fps="30000/1001", fps_f=30000 / 1001, stream_w=3840):
    """outputs: [(DynamicProjector, path, bitrate, view_fn)]. rot_fn(t) → 카메라→세계 R,
    view_fn(t) → 뷰→세계 V. 프레임 회전 Q = Rᵀ·V (뷰 → 카메라)."""
    S = stream_w
    dec = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + (["-ss", str(start)] if start is not None else []) + \
          (["-t", str(dur)] if dur is not None else []) + \
          ["-i", src, "-filter_complex", f"[0:0][0:1]hstack,{DEC_COLOR},format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"]
    encs = []
    for pj, path, br, _ in outputs:
        encs.append(subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{pj.w}x{pj.h}", "-r", fps,
             "-i", "-"] + (["-ss", str(start)] if start is not None else []) + (["-t", str(dur)] if dur is not None else []) +
            ["-i", src, "-map", "0:v", "-map", "1:a:0"] + _enc_vf() + ["-c:v", "hevc_videotoolbox"] + _enc_rate(br) + ["-tag:v", "hvc1",
             "-c:a", "aac", "-b:a", "192k", "-shortest", path], stdin=subprocess.PIPE, bufsize=10 ** 8))
    pd = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=10 ** 8)
    fsz, n = S * 2 * S * 3, 0
    try:
        while True:
            buf = pd.stdout.read(fsz)
            if len(buf) < fsz:
                break
            pair = np.frombuffer(buf, np.uint8).reshape(S, 2 * S, 3)
            t = n / fps_f
            R = rot_fn(t)
            for (pj, _, _, view_fn), e in zip(outputs, encs):
                e.stdin.write(pj(pair, R.T @ view_fn(t)).tobytes())
            n += 1
    finally:
        for e in encs:
            e.stdin.close()
        rcs = [e.wait() for e in encs]
        pd.wait()
    if any(rcs) or pd.returncode:
        raise RuntimeError(f"인코딩/디코딩 실패 dec={pd.returncode} enc={rcs}")
    return n
