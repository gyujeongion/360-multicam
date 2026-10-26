"""추적 모드: 손에 든·차 안 360 클립에서 사람(또는 지정 대상)마다 '고정된 앵글'을 만든다.

흐름: 자이로+수평 안정화 자세 R(t) → 안정화 파노라마 → 인물별 얼굴 방향을 초당 TRACK_FPS회 재검출해 경로
→ 경로 평활 → 경로를 따라가는 뷰 V_i(t)로 앵글·마스터 렌더(engine.render_dynamic).
각 앵글은 '자기 경로 기준 좌표'로 렌더되므로 이후 화자 감지·XML 구도는 고정 앵글과 똑같이 동작한다
(people_view.json = 모든 인물 yaw 0, pitch 0).

usage:
  python -m multicam360.track imu <src> <out.npz> --meta M            프레임별 자세 + 시간차
  python -m multicam360.track paths <panorama.mp4> <people.json> <out.json> [--fps 4]
  python -m multicam360.track render <src> <imu.npz> <paths.json> <out_dir> --meta M --hfov 70 --w 1920 --h 1080
        [--proj flat|pannini] [--bitrate 12M] [--only A,B]
"""
import argparse, json, math, os, subprocess, sys

import cv2
import numpy as np

from .config import asd_dir
from .engine import DynamicProjector, flat_dirs, load_lenses, pannini_dirs, render_dynamic
from .imu import read_imu
from .stabilize import estimate_time_offset, orientations

TRACK_FPS = 4
SMOOTH_S = 0.8          # 경로 평활(가우시안 시그마, 초)
GATE_DEG = 20           # 직전 방향에서 이 각도 안의 얼굴만 같은 사람 후보
MAX_HW_ENC = int(os.environ.get("MULTICAM360_MAX_ENC", 4))   # 동시 하드웨어 인코딩 수 상한
MAX_HW_DEC = int(os.environ.get("MULTICAM360_MAX_DEC", 2))   # 동시 하드웨어 디코딩(프로세스) 수 상한
WIDE_ID = "wide"
WIDE_HFOV = 110         # 실내 고정 광각 화각
LINK_SIM = 0.5          # 구간 간 신원 연결 임계(실측 같은 사람 0.93~0.95, 다른 사람 0.26~0.28)
GALLERY_N = 20          # 신원 기준 = 최근 정면 임베딩 20개 평균
GALLERY_LR = 0.1        # 신원 임베딩 갱신율 (확실히 같은 사람일 때만)
BURST_DPS = 120.0       # 카메라 급회전(°/s, 0.25초 평균) — 건네기 후보. 앉은 채 흔드는 손떨림 p99는 100~280이라 단독 판정엔 못 쓴다
BURST_MIN_S = 0.2
BURST_LOOKBACK_S = 4.0  # 분할이 걸리면 그 전 4초 안의 급회전 묶음 시작으로 경계를 당긴다
BURST_GAP_S = 1.5       # 이보다 가까운 급회전들은 한 번의 건네기 동작으로 묶는다
VIEW_HFOV, VIEW_W, VIEW_H = 100, 960, 540


# ---------- 1. 자세 ----------

def frame_motion(src, start, dur, size=160):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + (["-ss", str(start)] if start else []) +
                         (["-t", str(dur)] if dur else []) +
                         ["-i", src, "-map", "0:0", "-vf", f"scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, size, size).astype(np.float32)
    return np.abs(np.diff(fr, axis=0)).mean((1, 2))


def build_imu(src, meta, out, start=0.0, fps=30000 / 1001, search=3.0):
    """src 프레임 n의 시각 = start + n/fps (원본 기준). 구간 자르기 오차는 search초 안에서 상관으로 보정."""
    imu = read_imu(meta)
    n = int(float(subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                                           "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", src])))
    vt = start + np.arange(n) / fps
    motion = frame_motion(src, None, None)
    off, corr = (estimate_time_offset(imu, vt[1:len(motion) + 1], motion, search=search)
                 if motion.std() > 0.5 else (0.0, float("nan")))
    if not (corr > 0.5):                       # 움직임이 적어 상관이 약하면 0 (정지 장면은 시간차 영향도 작음)
        off = 0.0
    Rs = np.array(orientations(imu, vt, time_offset=off))
    np.savez_compressed(out, t=vt - start, R=Rs, time_offset=off, corr=corr)
    return {"frames": n, "time_offset": off, "corr": corr}


def rot_lookup(npz):
    d = np.load(npz)
    t, R = d["t"], d["R"]
    return lambda x: R[min(max(int(round(np.interp(x, t, np.arange(len(t))))), 0), len(t) - 1)]


# ---------- 2. 경로 ----------

def imu_bursts(npz, thr=BURST_DPS, min_dur=BURST_MIN_S, avg_s=0.25):
    """IMU 자세 R(t)의 회전 속도가 thr°/s를 넘는 구간 [(시작, 끝)] — 카메라를 건네는 순간의 후보.
    (실측 2026-10-24: trk016 건네기 화면 19.5~20.0초 ↔ 급회전 19.79초 시작)"""
    if not npz or not os.path.exists(npz):
        return []
    z = np.load(npz)
    t, R = z["t"], z["R"]
    if len(t) < 3:
        return []
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    w = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))) / np.maximum(np.diff(t), 1e-6)
    k = max(1, int(avg_s / np.median(np.diff(t))))
    w = np.convolve(w, np.ones(k) / k, "same")
    out, on = [], None
    for ti, v in zip(t[1:], w):
        if v > thr and on is None:
            on = ti
        elif v <= thr and on is not None:
            if ti - on >= min_dur:
                out.append((float(on), float(ti)))
            on = None
    return out


def _yaw_pitch_to_dir(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])   # y 아래+


def _dir_to_yaw_pitch(d):
    return math.degrees(math.atan2(d[0], d[2])), math.degrees(-math.asin(np.clip(d[1], -1, 1)))


def _view_basis(yaw, pitch):
    p, y = math.radians(pitch), math.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return Ry @ Rx                                         # 뷰 → 세계


def _equirect_view(eq, yaw, pitch):
    """안정화 파노라마(equirect)에서 원근 뷰 추출 (검출용)."""
    H, W = eq.shape[:2]
    d = flat_dirs(yaw, pitch, VIEW_HFOV, VIEW_W, VIEW_H).reshape(-1, 3)
    lon = np.arctan2(d[:, 0], d[:, 2]); lat = np.arcsin(np.clip(d[:, 1], -1, 1))
    mx = ((lon + np.pi) / (2 * np.pi) * W - 0.5).astype(np.float32).reshape(VIEW_H, VIEW_W)
    my = ((lat + np.pi / 2) / np.pi * H - 0.5).astype(np.float32).reshape(VIEW_H, VIEW_W)
    return cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _detector():
    cwd = os.getcwd()
    os.chdir(asd_dir()); sys.path.insert(0, asd_dir())
    try:
        from model.faceDetector.s3fd import S3FD
        import torch
        return S3FD(device="mps" if torch.backends.mps.is_available() else "cpu")
    finally:
        os.chdir(cwd)


def track_paths(panorama, people, fps=TRACK_FPS):
    """인물마다 시간별 얼굴 방향 [(t, yaw, pitch, found)] — 직전 방향 주변 뷰에서 재검출.
    동일인 유지: 각도 게이트 + 얼굴 임베딩(SFace) 유사도. 못 찾으면 직전 위치 유지(안정화 화면이라 사람은 제자리)."""
    from .faces import SAME_PERSON, FaceKit
    kit = FaceKit()
    f = (VIEW_W / 2) / math.tan(math.radians(VIEW_HFOV / 2))
    cap = cv2.VideoCapture(panorama)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30000 / 1001
    step = max(1, int(round(vfps / fps)))
    cur = {p["id"]: (p["yaw"], p.get("face_pitch", 5.0)) for p in people}
    paths = {p["id"]: [] for p in people}
    gallery = {p["id"]: None for p in people}
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            _, eq = cap.retrieve()
            t = i / vfps
            for pid, (yaw, pitch) in cur.items():
                view = _equirect_view(eq, yaw, pitch)
                best = None
                gate = GATE_DEG if gallery[pid] is not None else 30.0
                for x1, y1, x2, y2, score, row in kit.detect(view):
                    cx, cy = (x1 + x2) / 2 - VIEW_W / 2, (y1 + y2) / 2 - VIEW_H / 2
                    d = _view_basis(yaw, pitch) @ (np.array([cx, cy, f]) / np.linalg.norm([cx, cy, f]))
                    ang = math.degrees(math.acos(np.clip(d @ _yaw_pitch_to_dir(yaw, pitch), -1, 1)))
                    if ang > gate:
                        continue
                    emb = kit.embed(view, row)
                    sim = kit.sim(gallery[pid], emb) if gallery[pid] is not None else 1.0
                    if sim < SAME_PERSON:
                        continue                                  # 다른 사람
                    s_ = sim - 0.5 * ang / gate
                    if best is None or s_ > best[0]:
                        best = (s_, *_dir_to_yaw_pitch(d), emb, sim)
                if best:
                    cur[pid] = (best[1], best[2])
                    emb, sim = best[3], best[4]
                    if gallery[pid] is None:
                        gallery[pid] = emb
                    elif sim > 0.5:
                        g = (1 - GALLERY_LR) * gallery[pid] + GALLERY_LR * emb
                        gallery[pid] = g / np.linalg.norm(g)
                paths[pid].append((round(t, 3), round(cur[pid][0], 2), round(cur[pid][1], 2), bool(best)))
        i += 1
    cap.release()
    return paths


def _detect_all(kit, eq, f):
    """안정화 파노라마 전 방향 얼굴: [(yaw, pitch, emb|None, frontal)]."""
    out = []
    for vy in range(-180, 180, 60):                       # 60° 간격 100° 뷰 → 전 방향 겹쳐 덮음
        view = _equirect_view(eq, vy, 0)
        for x1, y1, x2, y2, score, row in kit.detect(view):
            cx, cy = (x1 + x2) / 2 - VIEW_W / 2, (y1 + y2) / 2 - VIEW_H / 2
            if abs(cx) > VIEW_W * 0.35:                    # 뷰 가장자리 중복은 이웃 뷰에 맡김
                continue
            d = _view_basis(vy, 0) @ (np.array([cx, cy, f]) / np.linalg.norm([cx, cy, f]))
            frontal = float(np.hypot(row[4] - row[6], row[5] - row[7])) / max(row[2], 1) >= 0.35
            out.append((*_dir_to_yaw_pitch(d), kit.embed(view, row) if frontal else None, frontal))
    return out


def _cluster_seats(samples, min_ratio=0.4, radius=12.0):
    """구간 초반 표본들의 얼굴 방향을 묶어 좌석 [(yaw, pitch, [emb...])]."""
    pts = [(y, p, e) for dets in samples for (y, p, e, _) in dets]
    seats = []
    for y, p, e in sorted(pts, key=lambda x: x[0]):
        for st in seats:
            if abs((y - st["y"] + 180) % 360 - 180) < radius and abs(p - st["p"]) < radius:
                st["n"] += 1; st["y"] += ((y - st["y"] + 180) % 360 - 180) / st["n"]; st["p"] += (p - st["p"]) / st["n"]
                if e is not None: st["e"].append(e)
                break
        else:
            seats.append({"y": y, "p": p, "n": 1, "e": [e] if e is not None else []})
    return [s for s in seats if s["n"] >= min_ratio * len(samples)]


def _detections(kit, panorama, fps, f):
    """파노라마 전 방향 얼굴 검출(가장 무거운 단계) — 파노라마 옆에 캐시. 파노라마가 바뀌면(크기·시각) 다시 계산."""
    import pickle
    st = os.stat(panorama)
    cache = f"{panorama}.dets_{fps}.pkl"
    key = (st.st_size, int(st.st_mtime), fps)
    if os.path.exists(cache):
        try:
            d = pickle.load(open(cache, "rb"))
            if d["key"] == key:
                return d["ts"], d["obs"]
        except Exception:
            pass
    cap = cv2.VideoCapture(panorama)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30000 / 1001
    step = max(1, int(round(vfps / fps)))
    ts, allobs = [], []
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            _, eq = cap.retrieve()
            ts.append(i / vfps); allobs.append(_detect_all(kit, eq, f))
        i += 1
    cap.release()
    ts = np.array(ts)
    pickle.dump({"key": key, "ts": ts, "obs": allobs}, open(cache, "wb"))
    return ts, allobs


def track_seats(panorama, people, fps=TRACK_FPS, sigma_drift=1.0, sigma_res=0.7, res_max=35.0,
                fail_s=2.0, settle_s=3.0, with_interior=False, interior_hfov=WIDE_HFOV, bursts=None,
                return_segments=False):
    """좌석 모델(차·테이블처럼 사람이 제자리인 장면): 경로_i(t) = 좌석_i + 공통 회전 δ(t) + 개인 잔차 r_i(t).
    - 매 시각 전 방향 얼굴 검출 → 좌석과 1:1 최적 배정(헝가리안, 게이트) → δ = 배정 오차의 중앙값.
    - 카메라 위치 이동(건네주기 등): 배정이 fail_s초 연속 실패하면 구간을 끊고, 새 구간 초반 settle_s초로
      좌석을 다시 잡은 뒤 SFace 얼굴 임베딩으로 이전 인물 ID에 연결(처음 보는 사람은 새 ID).
    - IMU 급회전(bursts, 기본: 파노라마 옆 imu.npz): 분할이 걸리면 경계를 직전 급회전 묶음의 시작으로 당기고, 좌석은
      급회전이 끝난 뒤로 다시 잡는다. 급회전 동안(손이 카메라를 옮기는 중)은 아무도 '존재'하지 않음 → 앵글 정지·컷 제외.
      판정 자체는 배정 실패로만 한다 — 앉은 채 흔드는 급회전도 많아 급회전만으로 자르면 오검출(실측 F017 3건).
      (실측: trk016 분할 22.9초 → 건네기 시작 19.8초(화면 19.5~20.0초), A013 4.9초 → 2.7초)
    얼굴이 안 보이는 사람(운전자 뒤통수)도 δ를 따라 자리를 지키고, 배정이 1:1이라 사람이 바뀌지 않는다."""
    from scipy.optimize import linear_sum_assignment
    from .faces import SAME_PERSON, FaceKit
    kit = FaceKit()
    f = (VIEW_W / 2) / math.tan(math.radians(VIEW_HFOV / 2))
    ts, allobs = _detections(kit, panorama, fps, f)

    ids = [p["id"] for p in people]
    seat = {p["id"]: np.array([p["yaw"], p.get("face_pitch", 5.0)], float) for p in people}
    gallery = {pid: [] for pid in ids}
    seg_of = np.zeros(len(ts), int)
    drift = np.zeros((len(ts), 2))
    obs = [dict() for _ in ts]                 # k → {pid: (yaw, pitch)}
    if bursts is None:
        bursts = imu_bursts(os.path.join(os.path.dirname(os.path.abspath(panorama)), "imu.npz"))
    groups = []                                      # 가까운 급회전 묶기 → (묶음 시작, 묶음 끝)
    for b in bursts:
        if groups and b[0] - groups[-1][1] < BURST_GAP_S:
            groups[-1] = (groups[-1][0], b[1])
        else:
            groups.append(b)
    transit = np.zeros(len(ts), bool)                # 카메라가 옮겨지는 중(급회전 구간)
    d_prev, fails, seg, k = np.zeros(2), 0, 0, 0
    segments = [{"start": 0.0, "seat": dict(seat)}]
    while k < len(ts):
        if transit[k]:
            seg_of[k] = seg; obs[k] = {}; drift[k] = d_prev
            k += 1
            continue
        dets = allobs[k]
        active = [pid for pid in ids if pid in seat]
        pred = np.array([seat[pid] + d_prev for pid in active])
        got = {}
        if dets and active:
            D = np.array([(y, p) for y, p, _, _ in dets])
            dy = (D[None, :, 0] - pred[:, None, 0] + 180) % 360 - 180
            cost = np.hypot(dy, D[None, :, 1] - pred[:, None, 1])
            # 신원 게이트: 정면 얼굴이고 갤러리가 쌓인 인물이면 다른 사람 얼굴엔 배정하지 않는다
            # (근거리 인물이 크게 움직일 때 옆 사람 얼굴로 끌려가는 것 방지)
            cent = {pid: np.mean(gallery[pid][-GALLERY_N:], 0) for pid in active if len(gallery[pid]) >= 3}
            for a, pid in enumerate(active):
                if pid not in cent:
                    continue
                for b, det in enumerate(dets):
                    if det[2] is not None and kit.sim(cent[pid], det[2]) < SAME_PERSON:
                        cost[a, b] = 1e6
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] <= GATE_DEG:
                    got[active[a]] = (pred[a, 0] + dy[a, b], D[b, 1])
                    e = dets[b][2]
                    # 모호한 배정(게이트 안에 다른 얼굴도 있음)은 신원 갤러리에 넣지 않는다 — 옆·뒷자리 사람의
                    # 정면 얼굴이 옆얼굴뿐인 사람의 갤러리를 채워 신원이 바뀌는 것 방지(실측: 016 뒷좌석 D)
                    alone = int((cost[a] <= GATE_DEG).sum()) == 1
                    if e is not None and alone and (active[a] not in cent or kit.sim(cent[active[a]], e) >= SAME_PERSON):
                        gallery[active[a]].append(e)
        # 카메라 이동 판정: 얼굴은 2개 이상 보이는데 배정이 절반 미만
        if len(dets) >= 2 and len(got) < max(1, len(dets) // 2):
            fails += 1
        else:
            fails = 0
        if fails * (1 / fps) >= fail_s:
            k0 = k - fails + 1                                   # 실패 시작 지점부터 새 구간
            k_settle = k0
            cand = [g for g in groups if ts[k0] - BURST_LOOKBACK_S <= g[0] <= ts[k0] and g[1] >= ts[k0] - BURST_LOOKBACK_S]
            if cand:                                             # 경계를 급회전 묶음 시작으로, 좌석은 급회전 끝난 뒤로 잡는다
                on, off = cand[-1]
                k0 = int(np.searchsorted(ts, on))
                k_settle = max(k0, int(np.searchsorted(ts, off)))
                transit[k0:k_settle] = True
            win = [allobs[j] for j in range(k_settle, min(len(ts), k_settle + int(settle_s * fps)))]
            new = _cluster_seats(win)
            # 신원 연결: 새 좌석의 정면 임베딩 평균 ↔ 기존 인물 갤러리 평균
            # 구간을 넘는 연결은 되돌릴 수 없어 더 엄격하게: 양쪽 정면 임베딩 3개 이상 + LINK_SIM 이상
            olds = [pid for pid in ids if len(gallery[pid]) >= 3]
            nseat = {}
            if new and olds:
                S = np.array([[kit.sim(np.mean(gallery[o][-GALLERY_N:], 0), np.mean(st["e"], 0)) if len(st["e"]) >= 3 else -1
                               for o in olds] for st in new])
                r, c = linear_sum_assignment(-S)
                for a, b in zip(r, c):
                    if S[a, b] >= LINK_SIM:
                        nseat[olds[b]] = np.array([new[a]["y"], new[a]["p"]])
                used = {a for a, b in zip(r, c) if S[a, b] >= LINK_SIM}
                if os.environ.get("MULTICAM360_DEBUG"):
                    print(f"[link] t={ts[k0]:.2f} olds={olds} gal={[len(gallery[o]) for o in olds]} "
                          f"new={[(round(st['y']), len(st['e'])) for st in new]}\n{np.round(S, 2)}", file=sys.stderr)
            else:
                used = set()
            for a, st in enumerate(new):
                if a not in used:
                    pid = chr(ord("A") + len(ids)); ids.append(pid); gallery[pid] = list(st["e"])
                    nseat[pid] = np.array([st["y"], st["p"]])
            seat = nseat                                          # 새 구간에 안 보이는 사람은 좌석 없음(앵글 정지)
            seg += 1; segments.append({"start": float(ts[k0]), "settle": float(ts[min(k_settle, len(ts) - 1)]),
                                       "seat": {p: v.tolist() for p, v in nseat.items()}})
            d_prev, fails = np.zeros(2), 0
            for j in range(k0, k + 1):
                obs[j] = {}
            k = k0
            seg_of[k:] = seg
            continue                                             # 새 좌석으로 k0부터 다시 배정
        if got:
            res = np.array([[got[p][0] - seat[p][0], got[p][1] - seat[p][1]] for p in got if p in seat])
            if len(res):
                d_prev = np.median(res, axis=0) if len(res) >= 2 else d_prev * 0.5 + res[0] * 0.5
        drift[k] = d_prev; obs[k] = got; seg_of[k] = seg
        k += 1

    def gsmooth(x, sig, idx):
        y = x.copy()
        dt = np.median(np.diff(ts)) if len(ts) > 1 else 1
        kk = max(1, int(round(sig / dt)))
        g = np.exp(-0.5 * (np.arange(-3 * kk, 3 * kk + 1) / kk) ** 2); g /= g.sum()
        for sgi in np.unique(seg_of):                            # 구간 경계를 넘어 평활하지 않음
            m = np.where(seg_of == sgi)[0]
            y[m] = np.convolve(np.pad(x[m], 3 * kk, mode="edge"), g, "valid")
        return y

    drift_s = np.stack([gsmooth(drift[:, c], sigma_drift, None) for c in (0, 1)], 1)
    seat_at = []                                                 # 시각별 좌석표
    for k in range(len(ts)):
        seat_at.append({p: np.array(v) for p, v in segments[seg_of[k]]["seat"].items()})
    paths = {}
    for pid in ids:
        has = np.array([pid in seat_at[k] and not transit[k] for k in range(len(ts))])
        if not has.any():
            continue
        seen_k = np.array([pid in obs[k] for k in range(len(ts))])
        # 다음 구간에서 사라지는 인물: 그 구간 마지막 검출 이후는 이미 카메라가 넘어가는 중 → 소급해서 부재
        # (분할 판정이 fail_s초 늦게 걸려, 그 사이 앵글이 엉뚱한 사람을 비추는 것 방지)
        for sgi in np.unique(seg_of)[:-1]:
            m = np.where(seg_of == sgi)[0]
            nxt = m[-1] + 1
            if has[m].any() and not has[nxt] and seen_k[m].any():
                has[m[m > m[seen_k[m]][-1]]] = False
        resid = np.full((len(ts), 2), np.nan)
        for k in range(len(ts)):
            if pid in obs[k] and has[k]:
                st = seat_at[k][pid]
                resid[k] = [obs[k][pid][0] - st[0] - drift_s[k, 0], obs[k][pid][1] - st[1] - drift_s[k, 1]]
        found = ~np.isnan(resid[:, 0])
        for c in (0, 1):
            resid[~found, c] = np.interp(ts[~found], ts[found], resid[found, c]) if found.sum() >= 2 else 0.0
        rs = np.clip(np.stack([gsmooth(resid[:, c], sigma_res, None) for c in (0, 1)], 1), -res_max, res_max)
        rows, last = [], None
        for k in range(len(ts)):
            if has[k]:
                st = seat_at[k][pid]
                last = ((st[0] + drift_s[k, 0] + rs[k, 0] + 180) % 360 - 180, st[1] + drift_s[k, 1] + rs[k, 1])
            if last is not None:      # 5번째 값 present: 이 구간에 좌석이 있음(화면에 존재). 없으면 앵글은 마지막 위치에 정지
                rows.append((round(float(ts[k]), 3), round(float(last[0]), 2), round(float(last[1]), 2), bool(found[k]),
                             bool(has[k])))
        paths[pid] = rows
    if return_segments:
        return paths, (_interior_path(ts, seg_of, segments, drift_s, interior_hfov) if with_interior else None), segments
    if with_interior:
        return paths, _interior_path(ts, seg_of, segments, drift_s, interior_hfov)
    return paths


def _interior_path(ts, seg_of, segments, drift_s, hfov):
    """차 실내 고정 광각: 좌석 공통 회전 δ(t)는 '차에 앉은 사람 전체가 안정화 화면에서 돈 양' = 실내 회전.
    방향 = 구간마다 hfov 창 안에 좌석이 가장 많이 들어오는 쪽(그 좌석들의 원형 평균) + δ(t). 개인 잔차는 쓰지 않는다."""
    dirs = {}
    for sgi, sg in enumerate(segments):
        S = np.array([np.asarray(v, float) for v in sg["seat"].values()])
        if not len(S):
            continue
        best = None
        for c in range(-180, 180, 5):
            inside = np.abs((S[:, 0] - c + 180) % 360 - 180) <= hfov / 2 - 10
            if best is None or inside.sum() > best[0]:
                best = (inside.sum(), inside)
        m = S[best[1]] if best[0] else S
        yaw = math.degrees(math.atan2(np.sin(np.radians(m[:, 0])).mean(), np.cos(np.radians(m[:, 0])).mean()))
        dirs[sgi] = (yaw, float(m[:, 1].mean()))
    rows = []
    for k in range(len(ts)):
        if seg_of[k] in dirs:
            y, p = dirs[seg_of[k]]
            rows.append((round(float(ts[k]), 3), round(float((y + drift_s[k, 0] + 180) % 360 - 180), 2),
                         round(float(p + drift_s[k, 1]), 2), True, True))
    return rows


def smooth_path(path, sigma_s=SMOOTH_S):
    """[(t,yaw,pitch,found)] → 평활된 (t, yaw, pitch). yaw는 펼쳐서(unwrap) 평활."""
    a = np.array([(r[0], r[1], r[2]) for r in path], float)
    if len(a) < 3:
        return a
    a[:, 1] = np.degrees(np.unwrap(np.radians(a[:, 1])))
    dt = np.median(np.diff(a[:, 0]))
    k = max(1, int(round(sigma_s / dt)))
    g = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2); g /= g.sum()
    out = a.copy()
    for c in (1, 2):
        pad = np.pad(a[:, c], 3 * k, mode="edge")
        out[:, c] = np.convolve(pad, g, "valid")
    return out


def view_fn(smoothed, pitch_bias=0.0):
    """시간 → 뷰→세계 회전 (경로를 따라가는 카메라)."""
    t, y, p = smoothed[:, 0], smoothed[:, 1], smoothed[:, 2]   # path 행: (t, yaw, pitch, found[, present])
    return lambda x: _view_basis(float(np.interp(x, t, y)), float(np.interp(x, t, p)) + pitch_bias)


# ---------- 3. 렌더 ----------

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("imu"); a1.add_argument("src"); a1.add_argument("out"); a1.add_argument("--meta", required=True)
    a1.add_argument("--start", type=float, default=0.0, help="src 0초가 원본의 몇 초인지 (구간 자르기 요청 시작)")
    a0 = sub.add_parser("pano"); a0.add_argument("src"); a0.add_argument("imu"); a0.add_argument("out")
    a0.add_argument("--meta", required=True); a0.add_argument("--width", type=int, default=3840)
    a2 = sub.add_parser("paths"); a2.add_argument("panorama"); a2.add_argument("people"); a2.add_argument("out")
    a2.add_argument("--fps", type=float, default=TRACK_FPS)
    a2.add_argument("--model", choices=["seats", "free"], default="seats",
                    help="seats = 사람이 제자리(차·테이블): 좌석+공통회전, 사람 안 바뀜 / free = 사람이 돌아다님: 개별 추적")
    a3 = sub.add_parser("render")
    for k in ("src", "imu", "paths", "out_dir"):
        a3.add_argument(k)
    a3.add_argument("--meta", required=True)
    a3.add_argument("--hfov", type=float, default=70); a3.add_argument("--pitch", type=float, default=0)
    a3.add_argument("--w", type=int, default=1920); a3.add_argument("--h", type=int, default=1080)
    a3.add_argument("--proj", choices=["flat", "pannini"], default="flat"); a3.add_argument("--pannini-d", type=float, default=0.7)
    a3.add_argument("--bitrate", default="12M"); a3.add_argument("--only")
    a3.add_argument("--jobs", type=int, default=1, help="인물 단위 병렬 프로세스 수")
    a3.add_argument("--wide-hfov", type=float, default=WIDE_HFOV, help="실내 고정 광각(cam_wide) 화각, 0이면 생략")
    a = ap.parse_args(argv)

    if a.cmd == "imu":
        print(json.dumps(build_imu(a.src, a.meta, a.out, start=a.start)))
        return 0
    if a.cmd == "pano":       # 안정화 파노라마 (인물 검출·경로 추적·라벨링용)
        from .engine import equirect_dirs
        pj = DynamicProjector(load_lenses(a.meta), equirect_dirs(a.width))
        n = render_dynamic(a.src, [(pj, a.out, "20M", lambda t: np.eye(3))], rot_lookup(a.imu))
        print(json.dumps({"frames": n}))
        return 0
    if a.cmd == "paths":
        people = json.load(open(a.people))["people"]
        interior = None
        if a.model == "seats":
            paths, interior = track_seats(a.panorama, people, a.fps, with_interior=True)
        else:
            paths = track_paths(a.panorama, people, a.fps)
        found = {k: round(sum(1 for x in v if x[3]) / max(1, len(v)), 3) for k, v in paths.items()}
        present = {k: round(sum(1 for x in v if (len(x) < 5 or x[4])) / max(1, len(v)), 3) for k, v in paths.items()}
        json.dump({"fps": a.fps, "paths": paths, "interior": interior, "found_ratio": found, "present_ratio": present},
                  open(a.out, "w"))
        print(json.dumps({"found_ratio": found, "present_ratio": present}))
        return 0

    lenses = load_lenses(a.meta)
    rot = rot_lookup(a.imu)
    pj_all = json.load(open(a.paths))
    paths = dict(pj_all["paths"])
    if pj_all.get("interior") and a.wide_hfov > 0:
        paths[WIDE_ID] = pj_all["interior"]                 # 실내 고정 광각 = 사람 앵글과 같은 렌더 경로, 화각만 다름
    os.makedirs(a.out_dir, exist_ok=True)
    if a.jobs > 1 and not a.only:              # 인물 단위로 나눠 하위 프로세스 병렬
        ids = list(paths)
        base, skip = [sys.executable, "-m", "multicam360.track"], False
        for x in (argv or sys.argv[1:]):
            if skip: skip = False; continue
            if x == "--jobs": skip = True; continue
            base.append(x)
        # VideoToolbox 세션 한도: 동시 인코더 MAX_HW_ENC·디코더 MAX_HW_DEC까지
        # (실측: 4K 인코더 6개 동시 → 인코더 열기 실패, 디코더 4개 동시 → 'No frame decoded' + 시스템 시간 폭증)
        for w in range(0, len(ids), MAX_HW_ENC):
            wave = ids[w:w + MAX_HW_ENC]
            nproc = min(a.jobs, MAX_HW_DEC)                       # 프로세스마다 원본 하드웨어 디코더 1개
            chunks = [wave[i::nproc] for i in range(nproc) if wave[i::nproc]]
            procs = [subprocess.Popen(base + ["--only", ",".join(c)]) for c in chunks]
            if any([pr.wait() for pr in procs]):
                print("병렬 렌더 실패", file=sys.stderr); return 1
        from .render import render_json
        render_json(a, a.out_dir, tracked=True)
        return 0
    outs = []
    for pid, path in paths.items():
        if a.only and pid not in a.only.split(","):
            continue
        hf = a.wide_hfov if pid == WIDE_ID else a.hfov
        if pid == WIDE_ID:                                     # 110° 평면 투영은 가장자리가 과하게 늘어나 파니니 고정
            dirs = pannini_dirs(0, 0, hf, a.w, a.h, a.pannini_d)
        else:
            dirs = (pannini_dirs(0, 0, hf, a.w, a.h, a.pannini_d) if a.proj == "pannini"
                    else flat_dirs(0, 0, hf, a.w, a.h))
        vf = view_fn(smooth_path(path), 0.0 if pid == WIDE_ID else a.pitch)
        outs.append((DynamicProjector(lenses, dirs), os.path.join(a.out_dir, f"cam_{pid}.mp4"), a.bitrate, vf))
    for w in range(0, len(outs), MAX_HW_ENC):                 # 한 프로세스 안에서도 인코더 동시 수 제한
        render_dynamic(a.src, outs[w:w + MAX_HW_ENC], rot)
    if not a.only:
        from .render import render_json
        render_json(a, a.out_dir, tracked=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
