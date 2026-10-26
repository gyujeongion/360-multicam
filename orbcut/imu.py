"""insv 트레일러의 IMU(자이로·가속도) 읽기 + 영상 시간축 정렬.

형식 참고: AdrianEddy/telemetry-parser (MIT/Apache-2.0) src/insta360/record.rs — 레코드 id 3(Gyro):
  raw 모드(is_raw_gyro) 항목 20B = u64 시각(µs) + u16×3 가속도 + u16×3 자이로 (각 -32768 오프셋)
  비raw 항목 56B = u64 시각(ms) + f64×3 가속도(g) + f64×3 자이로(rad/s)
시간 정렬(같은 출처): t_s = (시각 - first_frame_timestamp)/1e6 - gyro_timestamp/1000   (raw, µs 기준)
X6 축 방향은 위 프로젝트에 없음 → stabilize.py에서 실측(중력·회전)으로 결정.
"""
import shlex, struct, subprocess

import numpy as np

from .config import nas_ffmpeg, nas_host

from .calib import TRAILER_MAGIC

# ExtraMetadata 필드 (re/README.md E3)
F_FFT, F_GYRO_TS, F_RAW, F_GCFG = 24, 28, 62, 65


def read_trailer(path):
    """원본 insv(로컬 또는 nas:/...)에서 트레일러 전체 바이트."""
    if path.startswith("nas:"):
        q = shlex.quote(path[4:])
        foot = subprocess.run(["ssh", nas_host(), f"tail -c 72 {q}"], capture_output=True, check=True).stdout
        n = struct.unpack("<I", foot[32:36])[0]
        return subprocess.run(["ssh", nas_host(), f"tail -c {n} {q}"], capture_output=True, check=True).stdout
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell()
        f.seek(size - 72); foot = f.read(72)
        assert foot.endswith(TRAILER_MAGIC), "Insta360 트레일러 없음"
        n = min(struct.unpack("<I", foot[32:36])[0], size)
        f.seek(size - n)
        return f.read(n)


def records(raw):
    out, p = {}, len(raw) - 72
    while p > 6:
        rid, size = struct.unpack("<HI", raw[p - 6:p])
        start = p - 6 - size
        if start < 0:
            break
        if size:
            out.setdefault(rid >> 8, raw[start:start + size])   # 상위 바이트 = 레코드 종류, 하위 = 포맷
        p = start
    return out


def _pb_fields(buf, want):
    out, i = {}, 0
    def varint():
        nonlocal i
        r = s = 0
        while True:
            c = buf[i]; i += 1
            r |= (c & 0x7F) << s; s += 7
            if c < 0x80:
                return r
    while i < len(buf):
        tag = varint(); fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v = varint()
        elif wt == 1:
            v = struct.unpack("<d", buf[i:i + 8])[0]; i += 8
        elif wt == 5:
            v = struct.unpack("<f", buf[i:i + 4])[0]; i += 4
        else:
            L = varint(); v = buf[i:i + L]; i += L
        if fn in want:
            out[fn] = v
    return out


def read_imu(path_or_raw):
    """→ dict(t[s, 영상 시간축], acc[g] (N,3), gyr[deg/s] (N,3)) — 센서 원래 축."""
    raw = path_or_raw if isinstance(path_or_raw, (bytes, bytearray)) else read_trailer(path_or_raw)
    rec = records(raw)
    meta = _pb_fields(rec[1], {F_FFT, F_GYRO_TS, F_RAW, F_GCFG})
    fft = meta.get(F_FFT, 0)
    gts = meta.get(F_GYRO_TS, 0.0)
    is_raw = bool(meta.get(F_RAW, 0))
    acc_range, gyro_range = 16.0, 2000.0
    if F_GCFG in meta:
        g = _pb_fields(meta[F_GCFG], {1, 2})
        acc_range, gyro_range = float(g.get(1, acc_range)), float(g.get(2, gyro_range))
    d = rec[3]
    if is_raw:
        n = len(d) // 20
        a = np.frombuffer(d[:n * 20], np.dtype([("t", "<u8"), ("acc", "<u2", 3), ("gyr", "<u2", 3)]))
        t = (a["t"].astype(np.int64) - fft) / 1e6 - gts / 1000
        acc = (a["acc"].astype(np.float64) - 32768) / (32768 / acc_range)
        gyr = (a["gyr"].astype(np.float64) - 32768) / (32768 / gyro_range)
    else:
        n = len(d) // 56
        a = np.frombuffer(d[:n * 56], np.dtype([("t", "<u8"), ("acc", "<f8", 3), ("gyr", "<f8", 3)]))
        t = (a["t"].astype(np.float64) - fft / 1000) / 1000 - gts / 1000
        acc, gyr = a["acc"].copy(), np.degrees(a["gyr"])
    return {"t": t, "acc": acc, "gyr": gyr, "raw_mode": is_raw}


def motion_summary(imu, win=5.0, until=None):
    t, g = imu["t"], np.linalg.norm(imu["gyr"] - np.median(imu["gyr"], axis=0), axis=1)
    end = until or t[-1]
    return [(round(s, 1), round(float(g[(t >= s) & (t < s + win)].mean()), 1))
            for s in np.arange(0, end, win) if ((t >= s) & (t < s + win)).any()]
