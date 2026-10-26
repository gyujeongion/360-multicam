"""자이로 기반 안정화·수평 유지: 프레임별 카메라 자세 R(t) (카메라 좌표 → 수평 맞춘 세계 좌표).

좌표: 엔진과 같음 (x 오른쪽, y 아래, z 렌즈0 정면). 세계 좌표의 y = 중력 방향(아래).
- 자이로 적분(바디 각속도, 우측 곱) + 세계 좌표 가속도 장시간 평균(중력)으로 수평 보정.
- 요(수평 회전) 기준은 시작 시점 카메라 정면 — 방향 고정. 추적 모드는 이 위에서 시선을 따로 움직인다.
X6 축 대응(실측, re/README.md·LEDGER): 카메라 y = 센서 +x, 카메라 x = 센서 +z, 자이로 부호 그대로.
"""
import numpy as np

IMU_TO_CAM = {"Insta360 X6": np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])}


def _skew(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])


def expm(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = _skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def align(a, b):
    """벡터 a를 b로 보내는 최소 회전."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    K = _skew(v)
    return np.eye(3) + K + K @ K / (1 + c)


def gyro_bias(imu, M, win=1.0):
    """가장 조용한 win초 구간의 자이로 중앙값 = 바이어스 (카메라 축, rad/s)."""
    t, g = imu["t"], np.radians(imu["gyr"]) @ M.T
    mag = np.linalg.norm(g, axis=1)
    best, bias = np.inf, np.zeros(3)
    for s in np.arange(t[0], t[-1] - win, win / 2):
        m = (t >= s) & (t < s + win)
        if m.sum() > 10 and mag[m].mean() < best:
            best, bias = mag[m].mean(), np.median(g[m], axis=0)
    return bias


def orientations(imu, times, model="Insta360 X6", level=True, time_offset=0.0, level_window=4.0):
    """times(영상 초)마다 R: 카메라 → 세계(수평 맞춤). 세계 방향 d_w의 카메라 방향 = Rᵀ d_w.

    1) 자이로 적분(바이어스 제거)으로 상대 자세 → 빠른 흔들림 제거
    2) 수평: 세계 좌표로 옮긴 가속도를 level_window초 창으로 길게 평균 — 차량 가감속은 상쇄, 중력만 남음 —
       그 중력이 '아래'가 되도록 천천히 변하는 보정. (매 샘플 상보필터는 가감속에 흔들려 폐기, LEDGER 참조)
    time_offset: 영상 시각 + offset = 자이로 시각 (파일마다 estimate_time_offset으로 추정)
    """
    M = IMU_TO_CAM[model]
    t = imu["t"]
    g = np.radians(imu["gyr"]) @ M.T - gyro_bias(imu, M)
    acc = imu["acc"] @ M.T                         # 비력(정지 시 위쪽), 카메라 좌표
    q = np.asarray(times, float) + time_offset
    lo, hi = q.min() - level_window, q.max() + level_window
    sel = np.where((t >= lo) & (t <= hi))[0]
    # 자이로 적분: 구간 시작 카메라 자세 = 단위
    Rs = np.empty((len(sel), 3, 3)); R = np.eye(3)
    for n, i in enumerate(sel):
        if n:
            dt = t[i] - t[sel[n - 1]]
            if 0 < dt < 0.1:
                R = R @ expm(g[i] * dt)
        Rs[n] = R
    ts = t[sel]
    out = [Rs[min(np.searchsorted(ts, x), len(ts) - 1)].copy() for x in q]
    if not level:
        return out
    # 세계 좌표 중력(위쪽 비력) 장시간 평균 → 수평 보정
    up_w = np.einsum("nij,nj->ni", Rs, acc[sel])
    step = max(1, len(sel) // 4000)
    tw, uw = ts[::step], up_w[::step]
    res = []
    for x, Rx in zip(q, out):
        m = np.abs(tw - x) <= level_window / 2
        u = uw[m].mean(0) if m.any() else uw.mean(0)
        res.append(align(u, np.array([0.0, -1.0, 0.0])) @ Rx)
    # 요 기준: 첫 시각의 카메라 정면 방위를 0으로
    fwd = res[0] @ np.array([0, 0, 1.0])
    y0 = np.arctan2(fwd[0], fwd[2])
    Ry = np.array([[np.cos(-y0), 0, np.sin(-y0)], [0, 1, 0], [-np.sin(-y0), 0, np.cos(-y0)]])
    return [Ry @ R for R in res]


def estimate_time_offset(imu, frame_times, frame_motion, search=0.3, step=0.01):
    """영상 프레임간 움직임(밝기 차)과 자이로 크기의 상관이 최대인 시간차(초). 영상 t + offset = 자이로 t.
    누적합으로 창 평균을 벡터화 — 구간 자르기(키프레임 시작) 오차까지 넓게(search=3) 찾아도 빠르다."""
    t, gm = imu["t"], np.linalg.norm(imu["gyr"], axis=1)
    cs = np.concatenate([[0.0], np.cumsum(gm)])
    ft = np.asarray(frame_times, float)
    dt = np.median(np.diff(ft))
    fm = (frame_motion - frame_motion.mean()) / (frame_motion.std() + 1e-9)
    best = (0.0, -2.0)
    for off in np.arange(-search, search + 1e-9, step):
        i1 = np.searchsorted(t, ft + off); i0 = np.searchsorted(t, ft - dt + off)
        a = (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)
        a = (a - a.mean()) / (a.std() + 1e-9)
        c = float((fm * a).mean())
        if c > best[1]:
            best = (round(float(off), 3), c)
    return best


def tilt_deg(R, up_cam):
    """세계 좌표에서 측정 중력(위)과 진짜 위의 각도 차 — 수평 오차."""
    u = R @ (up_cam / np.linalg.norm(up_cam))
    return float(np.degrees(np.arccos(np.clip(-u[1], -1, 1))))
