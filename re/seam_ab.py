"""M15-2: 이음새 합성 변형 A/B — 샘플 프레임을 원본에서 직접 투영해(렌더 없이) 이음새 지표를 잰다.

지표(모두 우리 좌표, 블렌딩 띠 0.2<alpha<0.8 & |lat|<60°):
- ret   고주파 보존율 = Σ|∇G₁(합성)| / max(Σ|∇G₁(렌즈0 투영)|, Σ|∇G₁(렌즈1 투영)|). 1.0 = 블렌딩으로 잃은 선명도 없음
- fake  가짜 에지 = mean max(0, |∇합성| − max(|∇렌즈0|, |∇렌즈1|)) / mean max(...). 어긋난 채 자르면 올라감
- flick 플리커 = 연속 2프레임 평균 차이(띠) / (본문). 1.0 = 본문과 같음
- lap   Laplacian 분산 Studio 대비(±6° 띠, 회전 정렬) — 기존 지표, 참고용
usage: python re/seam_ab.py <insv> <studio.mp4> <out_dir> [--times 3,10,...] [--only name,name]"""
import argparse, json, os, subprocess, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbcut import engine
from orbcut.engine import DEC_COLOR, Projector, equirect_dirs, load_lenses
from studio_compare import H, W, estimate_rotation, grab, remap_rot, seam_mask, sharpness

VARIANTS = {
    "base":      {},
    "noflow":    {"_flow": False},
    "cur":       {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0},
    "sm8":       {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0, "FLOW_SMOOTH": 8},
    "sm16":      {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0, "FLOW_SMOOTH": 16},
    "sm32":      {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0, "FLOW_SMOOTH": 32},
    "vc3":       {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0, "FLOW_VCLAMP": 3},
    "sm16_vc3":  {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0, "FLOW_SMOOTH": 16, "FLOW_VCLAMP": 3},
    "sm16_vc3_ema": {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 0, "FLOW_EMA": 0.8, "FLOW_SMOOTH": 16, "FLOW_VCLAMP": 3},
    "sm32_vc3_ema": {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 0, "FLOW_EMA": 0.8, "FLOW_SMOOTH": 32, "FLOW_VCLAMP": 3},
    "sm16_vc3_ema_ad12": {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 12, "FLOW_EMA": 0.8, "FLOW_SMOOTH": 16, "FLOW_VCLAMP": 3},
    "sm16_vc3_ema9": {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 0, "FLOW_EMA": 0.9, "FLOW_SMOOTH": 16, "FLOW_VCLAMP": 3},
    "sm16_vc3_max40": {"SEAM_MODE": "band2", "FLOW_TILE": 0, "EMA_ADAPT": 2.0, "FLOW_SMOOTH": 16, "FLOW_VCLAMP": 3, "FLOW_MAX": 40},
}
DEFAULTS = {k: getattr(engine, k) for k in ("FEATHER", "FLOW_EMA", "FLOW_MAX_ELEV", "FLOW_TILE", "SEAM_MODE",
                                             "EMA_ADAPT", "BAND2_K", "BAND2_SIGMA", "CONF_FB", "CONF_RES",
                                             "FLOW_SMOOTH", "FLOW_VCLAMP", "FLOW_MAX")}


def pair_at(insv, t, S=3840):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", f"{t:.3f}", "-i", insv, "-frames:v", "1",
                          "-filter_complex", f"[0:0][0:1]hstack,{DEC_COLOR},format=bgr24[v]", "-map", "[v]",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(S, 2 * S, 3).copy()


def grad(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("insv"); ap.add_argument("studio"); ap.add_argument("out")
    ap.add_argument("--times", default="3,10,17,24,31,38,45")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    times = [float(x) for x in a.times.split(",")]
    names = a.only.split(",") if a.only else list(VARIANTS)
    lenses = load_lenses(a.insv)
    dirs = equirect_dirs(W)
    lat_ok = np.abs((np.mgrid[0:H, 0:W][0] + 0.5) / H * 180 - 90) < 60
    seam6 = seam_mask()
    frames = [(t, grab(a.studio, t), pair_at(a.insv, t), pair_at(a.insv, t + 1 / 29.97)) for t in times]
    print(f"프레임 {len(frames)}쌍 준비", flush=True)
    results, R_cache = {}, {}
    for name in names:
        cfg = {**DEFAULTS, **VARIANTS[name]}
        flow_on = cfg.pop("_flow", True)
        for k, v in cfg.items():
            setattr(engine, k, v)
        t0 = time.time()
        pj = Projector(lenses, dirs, stream_w=3840, flow=flow_on)
        band = (pj.alpha > 0.2) & (pj.alpha < 0.8) & lat_ok
        body = lat_ok & ~((pj.alpha > 0.001) & (pj.alpha < 0.999))
        m = {"ret": [], "fake": [], "flick": [], "lap": [], "lap_alpha": [], "wob_t": [], "wob_s": []}
        crops = []
        for t, st, p0, p1 in frames:
            pj.prev = {}
            o0 = pj(p0)
            F0 = {k: v.copy() for k, v in pj.prev.items() if isinstance(k, int)}
            o1 = pj(p1)
            F1 = {k: v for k, v in pj.prev.items() if isinstance(k, int)}
            if F1:   # 출렁임: 프레임 간 플로우 변화 평균(px) · 플로우 공간 고주파(px, σ8 잔차)
                m["wob_t"].append(float(np.mean([np.hypot(*(F1[k] - F0[k]).transpose(2, 0, 1)).mean() for k in F1 if k in F0])))
                m["wob_s"].append(float(np.mean([np.hypot(*(F1[k] - cv2.GaussianBlur(F1[k], (0, 0), 8)).transpose(2, 0, 1)).mean()
                                                 for k in F1])))
            w0 = cv2.remap(p0, pj.mx[0], pj.my[0], cv2.INTER_CUBIC)
            w1 = cv2.remap(p0, pj.mx[1], pj.my[1], cv2.INTER_CUBIC)
            gb, g0, g1 = grad(o0), grad(w0), grad(w1)
            gmax = np.maximum(g0, g1)
            # 분모는 픽셀별 max가 아니라 렌즈별 합의 max — 픽셀별 max는 노이즈를 골라 담아 단일 렌즈도 1을 못 넘긴다
            m["ret"].append(float(gb[band].sum() / (max(g0[band].sum(), g1[band].sum()) + 1e-6)))
            m["fake"].append(float(np.maximum(0, gb - gmax)[band].mean() / (gmax[band].mean() + 1e-6)))
            d = cv2.absdiff(o0, o1).mean(-1)
            m["flick"].append(float(d[band].mean() / (d[body].mean() + 1e-6)))
            gob = cv2.cvtColor(o0, cv2.COLOR_BGR2GRAY)
            m["lap_alpha"].append(sharpness(gob, band))
            if t not in R_cache:
                R_cache[t] = estimate_rotation(st, o0)[0]
            ou2 = remap_rot(o0, R_cache[t].T)
            seam = remap_rot(seam6.astype(np.uint8) * 255, R_cache[t].T) > 127
            m["lap"].append(sharpness(cv2.cvtColor(ou2, cv2.COLOR_BGR2GRAY), seam) /
                            sharpness(cv2.cvtColor(st, cv2.COLOR_BGR2GRAY), seam))
            ys, xs = np.where(band[:, W // 4: W // 2])          # 왼쪽 이음새 가운데 부근 확대 저장
            cy, cx = int(np.median(ys)), int(np.median(xs)) + W // 4
            crops.append(o0[cy - 150: cy + 150, cx - 200: cx + 200])
        r = {k: round(float(np.median(v)), 3) for k, v in m.items()}
        r["sec_per_frame"] = round((time.time() - t0) / (2 * len(frames)), 2)
        results[name] = r
        cv2.imwrite(os.path.join(a.out, f"seam_{name}.jpg"), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(name, json.dumps(r), flush=True)
        json.dump(results, open(os.path.join(a.out, "seam_ab.json"), "w"), indent=1)
    print("== 끝")


if __name__ == "__main__":
    main()
