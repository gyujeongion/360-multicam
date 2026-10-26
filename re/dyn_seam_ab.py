"""동적 렌더 이음새 A/B: feather(페더만) vs band2(카메라 고정 띠 플로우 + 2밴드) — 같은 swap 계획 클립을 순차 디코딩해
같은 프레임에서 두 방식을 같이 돌리고 이음새 지표를 잰다(띠 플로우 EMA가 실제 렌더처럼 이어지게 순차).

지표(이음새 띠 0.2<alpha<0.8, 정렬 오차만 0.3<alpha<0.7):
- align  두 렌즈 샘플 밝기 차 평균(0~255). 실제로 섞이는 두 장의 어긋남 = 이중상의 원인. 낮을수록 좋음
- ret    고주파 보존율 = Σ|∇합성| / max(Σ|∇렌즈0|, Σ|∇렌즈1|) (seam_ab와 같음). 이중상·블러면 내려감
- fake   가짜 에지(seam_ab와 같음). 어긋난 채 자르면 올라감
- flick  연속 프레임 차이 띠/본문 비. 1 = 본문과 같음
- wob_t  (band2만) 보이는 띠 열의 프레임 간 플로우 변화 평균(띠 px) — 출렁임
usage: python re/dyn_seam_ab.py <swap_plan.json> <out_dir> [--every 15] [--dur 초]"""
import argparse, json, os, subprocess, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multicam360.engine import DEC_COLOR, DynamicProjector, flat_dirs, load_lenses  # noqa: E402
from multicam360.swap import view_fn  # noqa: E402
from multicam360.track import rot_lookup  # noqa: E402


def grad(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def gray(pair, x, y):
    return cv2.cvtColor(cv2.remap(pair, x, y, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan"); ap.add_argument("out")
    ap.add_argument("--every", type=int, default=15); ap.add_argument("--dur", type=float)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    it = json.load(open(a.plan))[0]
    lenses = load_lenses(it["meta"])
    view = flat_dirs(0, 0, it["hfov"], it["w"], it["h"])
    pjs = {"feather": DynamicProjector(lenses, view, seam="feather"), "band2": DynamicProjector(lenses, view, seam="band2")}
    rot, V, s0 = rot_lookup(it["imu"]), view_fn(it), it["in_s"]
    dur = a.dur or it["dur"]
    S = 3840
    dec = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", str(s0), "-t", str(dur), "-i", it["src"],
           "-filter_complex", f"[0:0][0:1]hstack,{DEC_COLOR},format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"]
    pd = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=10 ** 8)
    fsz, n = S * 2 * S * 3, 0
    m = {k: {"align": [], "ret": [], "fake": [], "flick": [], "sec": []} for k in pjs}
    wob, prevF, prev_out, crops = [], None, {}, []
    while True:
        buf = pd.stdout.read(fsz)
        if len(buf) < fsz:
            break
        pair = np.frombuffer(buf, np.uint8).reshape(S, 2 * S, 3)
        t = s0 + n / (30000 / 1001)
        Q = rot(t).T @ V(t)
        outs = {}
        pjs["band2"].strip.last_cols = None              # 이번 프레임에 띠를 계산했을 때만 출렁임을 잰다
        for k, pj in pjs.items():
            t0 = time.time(); outs[k] = pj(pair, Q); m[k]["sec"].append(time.time() - t0)
        st = pjs["band2"].strip
        if getattr(st, "last_cols", None) is not None:
            F = st.F[:, st.last_cols].copy()
            if prevF is not None and np.array_equal(prevF[0], st.last_cols):
                wob.append(float(np.hypot(*(F - prevF[1]).transpose(2, 0, 1)).mean()))
            prevF = (st.last_cols.copy(), F)
        if n % a.every == 0 and n > 0:
            xf, yf, alpha, _ = pjs["feather"].last
            xb, yb, _, _ = pjs["band2"].last
            band, mid = (alpha > 0.2) & (alpha < 0.8), (alpha > 0.3) & (alpha < 0.7)
            body = ~((alpha > 0.001) & (alpha < 0.999))
            if band.sum() > 2000:
                w0 = cv2.remap(pair, xf[0], yf[0], cv2.INTER_CUBIC); w1 = cv2.remap(pair, xf[1], yf[1], cv2.INTER_CUBIC)
                g0, g1 = grad(w0), grad(w1); gmax = np.maximum(g0, g1)
                for k, (x, y) in {"feather": (xf, yf), "band2": (xb, yb)}.items():
                    m[k]["align"].append(float(np.abs(gray(pair, x[0], y[0]) - gray(pair, x[1], y[1]))[mid].mean()))
                    gb = grad(outs[k])
                    m[k]["ret"].append(float(gb[band].sum() / (max(g0[band].sum(), g1[band].sum()) + 1e-6)))
                    m[k]["fake"].append(float(np.maximum(0, gb - gmax)[band].mean() / (gmax[band].mean() + 1e-6)))
                    if k in prev_out:
                        d = cv2.absdiff(outs[k], prev_out[k]).mean(-1)
                        m[k]["flick"].append(float(d[band].mean() / (d[body].mean() + 1e-6)))
                ys, xs = np.where(mid)
                cy, cx = int(np.median(ys)), int(np.median(xs))
                y0, x0 = min(max(cy - 180, 0), it["h"] - 360), min(max(cx - 240, 0), it["w"] - 480)
                crops.append((round(t, 2), np.hstack([outs["feather"][y0:y0 + 360, x0:x0 + 480], outs["band2"][y0:y0 + 360, x0:x0 + 480]])))
        if (n + 1) % a.every == 0:
            prev_out = {k: v.copy() for k, v in outs.items()}
        else:
            prev_out = {}
        n += 1
    pd.wait()
    res = {k: {kk: round(float(np.median(v)), 3) for kk, v in d.items() if v} for k, d in m.items()}
    res["band2"]["wob_t"] = round(float(np.median(wob)), 3) if wob else None
    res["frames"], res["samples"] = n, len(m["feather"]["align"])
    json.dump(res, open(os.path.join(a.out, "dyn_seam_ab.json"), "w"), indent=1)
    for t, c in crops:
        cv2.imwrite(os.path.join(a.out, f"crop_{t:06.2f}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
