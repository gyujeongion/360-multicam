"""M15: Insta360 Studio 360 내보내기 vs orbcut 파노라마 화질 비교.

두 equirect 영상을 같은 시각 프레임끼리 3840x1920으로 맞추고,
① 특징점 → 단위벡터 → RANSAC 회전 추정으로 방향을 맞춘 뒤(Studio는 수평·방향 보정을 할 수 있음)
② 이음새 띠(렌즈 축과 직각 ±6°)와 나머지 영역을 나눠 선명도·노이즈·구조 유사도·색을 잰다.
선명도·노이즈는 각자 자기 영상에서 재므로 정렬 오차와 무관. 구조 유사도는 정렬 품질에 의존(참고용).

usage: python re/studio_compare.py <studio.mp4> <ours.mp4> <out_dir> [--times 5,10,...]
"""
import argparse, json, os, subprocess

import cv2
import numpy as np

W, H = 3840, 1920
SEAM_HALF_DEG = 6.0


def grab(path, t):
    """t초 프레임 하나를 W×H BGR로(ffmpeg 1스레드, 메모리 작게)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                          "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()


def px_to_dir(x, y):
    lon = (x + 0.5) / W * 2 * np.pi - np.pi
    lat = (y + 0.5) / H * np.pi - np.pi / 2
    return np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)], -1)   # y 아래+


def remap_rot(img, R):
    """출력 방향 d → 입력에서 R·d를 읽는다(=입력을 R⁻¹만큼 돌림)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = px_to_dir(xx, yy) @ R.T
    lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    mx = ((lon + np.pi) / (2 * np.pi) * W - 0.5).astype(np.float32)
    my = ((lat + np.pi / 2) / np.pi * H - 0.5).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def kabsch(a, b):
    """a_i ≈ R b_i 인 R."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def estimate_rotation(ref, mov, iters=400, thr_deg=0.4):
    """ref 방향 = R · mov 방향. 위도 ±60° 안 ORB 특징점, RANSAC."""
    orb = cv2.ORB_create(8000)
    g1, g2 = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (ref, mov))
    mask = np.zeros((H, W), np.uint8); mask[H // 6: H * 5 // 6] = 255
    k1, d1 = orb.detectAndCompute(g1, mask); k2, d2 = orb.detectAndCompute(g2, mask)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.array([k1[x.queryIdx].pt for x in m]); p2 = np.array([k2[x.trainIdx].pt for x in m])
    a, b = px_to_dir(p1[:, 0], p1[:, 1]), px_to_dir(p2[:, 0], p2[:, 1])
    rng = np.random.default_rng(0); best, thr = None, np.cos(np.radians(thr_deg))
    for _ in range(iters):
        idx = rng.choice(len(a), 3, replace=False)
        R = kabsch(a[idx], b[idx])
        inl = np.einsum("ij,ij->i", a, b @ R.T) > thr
        if best is None or inl.sum() > best.sum():
            best = inl
    R = kabsch(a[best], b[best])
    return R, int(best.sum()), len(a)


def seam_mask():
    """orbcut 파노라마(카메라 좌표, 렌즈0 = +z) 기준 이음새 띠: 렌즈 축(z)과 거의 직각인 방향."""
    yy, xx = np.mgrid[0:H, 0:W]
    d = px_to_dir(xx, yy)
    ang = np.degrees(np.arccos(np.clip(np.abs(d[..., 2]), 0, 1)))       # 축과의 각, 90°가 경계
    lat = np.abs(np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1))))
    return (np.abs(90 - ang) < SEAM_HALF_DEG) & (lat < 60)


def sharpness(gray, m):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap[m].var())


def noise(gray, m):
    """평탄 영역(국소 기울기 하위 20%)의 고주파 잔차 표준편차 — 클수록 노이즈 많음."""
    g = gray.astype(np.float32)
    resid = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    grad = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 3)
    flat = m & (grad < np.percentile(grad[m], 20))
    return float(resid[flat].std())


def ssim_map(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    mu_a, mu_b = cv2.GaussianBlur(a, (0, 0), 1.5), cv2.GaussianBlur(b, (0, 0), 1.5)
    va = cv2.GaussianBlur(a * a, (0, 0), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (0, 0), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (0, 0), 1.5) - mu_a * mu_b
    c1, c2 = 6.5025, 58.5225
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("studio"); ap.add_argument("ours"); ap.add_argument("out")
    ap.add_argument("--times", default="3,10,17,24,31,38,45")
    ap.add_argument("--offset", type=float, default=0.0, help="ours 영상이 원본 몇 초부터인지(구간 렌더)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seam_ours = seam_mask()
    rows, crops = [], []
    for t in [float(x) for x in a.times.split(",")]:
        st, ou = grab(a.studio, t), grab(a.ours, t - a.offset)
        R, inl, n = estimate_rotation(st, ou)                       # studio 방향 = R · ours 방향
        ou_al = remap_rot(ou, R.T)                                  # ours를 studio 방향으로
        seam = remap_rot(seam_ours.astype(np.uint8) * 255, R.T) > 127
        lat = np.abs((np.mgrid[0:H, 0:W][0] + 0.5) / H * 180 - 90) < 60
        body = lat & ~seam
        gs, go = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (st, ou_al))
        ss = ssim_map(gs, go)
        ls, lo = (cv2.cvtColor(i, cv2.COLOR_BGR2LAB).astype(np.float32) for i in (st, ou_al))
        r = {"t": t, "align_inliers": f"{inl}/{n}",
             "sharp_seam": [sharpness(gs, seam), sharpness(go, seam)],
             "sharp_body": [sharpness(gs, body), sharpness(go, body)],
             "noise_body": [noise(gs, body), noise(go, body)],
             "ssim_seam": float(ss[seam].mean()), "ssim_body": float(ss[body].mean()),
             "lab_mean_studio": ls[body].mean(0).round(1).tolist(), "lab_mean_ours": lo[body].mean(0).round(1).tolist()}
        rows.append(r)
        # 이음새에서 두 영상이 가장 다른 곳 확대(위 Studio / 아래 orbcut)
        sm = cv2.blur(np.where(seam, 1 - ss, 0).astype(np.float32), (101, 101))
        y, x = np.unravel_index(np.argmax(sm), sm.shape)
        y0, x0 = int(np.clip(y - 150, 0, H - 300)), int(np.clip(x - 200, 0, W - 400))
        crops.append(np.vstack([st[y0:y0 + 300, x0:x0 + 400], ou_al[y0:y0 + 300, x0:x0 + 400]]))
        print(json.dumps(r, ensure_ascii=False))
    cv2.imwrite(os.path.join(a.out, "seam_crops.jpg"), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 90])
    agg = lambda k, i: float(np.median([r[k][i] for r in rows]))
    summary = {
        "sharp_seam_ratio": agg("sharp_seam", 1) / agg("sharp_seam", 0),
        "sharp_body_ratio": agg("sharp_body", 1) / agg("sharp_body", 0),
        "noise_ratio": agg("noise_body", 1) / agg("noise_body", 0),          # >1 = orbcut이 더 노이즈
        "ssim_seam": float(np.median([r["ssim_seam"] for r in rows])),
        "ssim_body": float(np.median([r["ssim_body"] for r in rows])),
        "lab_delta": (np.median([np.array(r["lab_mean_ours"]) - np.array(r["lab_mean_studio"]) for r in rows], 0)).round(1).tolist(),
    }
    json.dump({"summary": summary, "frames": rows}, open(os.path.join(a.out, "studio_compare.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
