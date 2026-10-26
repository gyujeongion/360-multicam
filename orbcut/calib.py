"""Insta360 insv 공장 보정값(offset_v6)으로 듀얼 어안을 정확히 투영한다.

형식(re/README.md 참조):
- 보정값: insv 트레일러 레코드 0x101 = protobuf 메타데이터, 그 안의 offset_v6 문자열
- 투영: Mei omnidir 모델 + radial-tangential·박막 프리즘 왜곡(13계수)
- offset_v6 렌즈 블록 27값: xi fx fy cx cy | rot3(deg) | trans3 | dist13 | W H max_fov(전각, deg)
"""
import struct

import numpy as np

TRAILER_MAGIC = b"8db42d694ccc418790edff439fe026bf"


def read_metadata_bytes(path_or_tail):
    """insv 끝(또는 끝부분을 담은 파일)에서 0x101 레코드 바이트를 찾는다."""
    with open(path_or_tail, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        n = min(size, 16 << 20)
        f.seek(size - n)
        raw = f.read(n)
    assert raw.endswith(TRAILER_MAGIC), "Insta360 트레일러 없음"
    p = len(raw) - 72
    while p > 6:
        rid, rsize = struct.unpack("<HI", raw[p - 6:p])
        start = p - 6 - rsize
        if rid == 0x101:
            return raw[start:start + rsize]
        if start < 0:
            break
        p = start
    raise ValueError("0x101 레코드 없음")


def offset_v6_from_metadata(meta_bytes):
    """protobuf 필드 111(offset_v6) 문자열만 직접 파싱 (스키마 의존성 없이)."""
    i = 0
    def varint():
        nonlocal i
        r = s = 0
        while True:
            c = meta_bytes[i]; i += 1
            r |= (c & 0x7F) << s; s += 7
            if c < 0x80:
                return r
    while i < len(meta_bytes):
        tag = varint()
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            varint()
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
        elif wt == 2:
            L = varint()
            if fn == 111:
                return meta_bytes[i:i + L].decode()
            i += L
        else:
            raise ValueError(f"wire type {wt}")
    raise ValueError("offset_v6 없음")


def parse_offset_v6(s):
    v = s.split("_")
    n = int(v[0])
    lenses = []
    for k in range(n):
        b = [float(x) for x in v[1 + 27 * k: 1 + 27 * (k + 1)]]
        lenses.append({"xi": b[0], "f": np.array(b[1:3]), "c": np.array(b[3:5]),
                       "rot": np.array(b[5:8]), "t": np.array(b[8:11]), "dist": np.array(b[11:24]),
                       "size": (int(b[24]), int(b[25])), "max_fov": np.radians(b[26] / 2)})
    return lenses


def radtan_pro(x, y, k):
    """Studio RadtanDistortPro 그대로."""
    mx2, my2, mxy = x * x, y * y, x * y
    r2 = mx2 + my2
    r4 = r2 * r2
    rad = 1 + k[0] * r2 + k[1] * r4 + k[2] * r4 * r2 + k[3] * r4 * r4 + k[4] * r4 * r4 * r2
    px = x * rad + (k[5] + r2 * k[7]) * (r2 + 2 * mx2) + 2 * (k[6] + r2 * k[8]) * mxy + k[9] * r2 + k[11] * r4
    py = y * rad + (k[6] + r2 * k[8]) * (r2 + 2 * my2) + 2 * (k[5] + r2 * k[7]) * mxy + k[10] * r2 + k[12] * r4
    return px, py


def omni_project(rays, lens):
    """카메라 좌표 광선 (N,3) → 보정 좌표계 픽셀 (N,2), 유효 마스크. Studio OmniProjectionPro."""
    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    valid = z / r >= np.cos(lens["max_fov"]) - 0.01
    tmp = 1.0 / (z + lens["xi"] * r)
    px, py = radtan_pro(x * tmp, y * tmp, lens["dist"])
    return np.stack([px * lens["f"][0] + lens["c"][0], py * lens["f"][1] + lens["c"][1]], 1), valid


def euler(a, b, c, order="xyz"):
    """도 단위 오일러 → 회전행렬. order는 적용 순서(첫 글자가 먼저 적용)."""
    def R(ax, d):
        t = np.radians(d); cs, sn = np.cos(t), np.sin(t)
        return {"x": np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]]),
                "y": np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]]),
                "z": np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])}[ax]
    ang = {"x": a, "y": b, "z": c}
    M = np.eye(3)
    for ax in order:
        M = R(ax, ang[ax]) @ M
    return M
