"""렌즈1 회전 잔차 추정: 카메라 고정 이음새 띠(SeamStrip)에서 두 렌즈 사이 원시 플로우를 고리 전체로 재서
- 이음새를 따라가는 성분(θ) = 렌즈1의 광축 둘레 회전 오차(롤). 시차는 이음새를 가로지르는 방향(φ)이라 섞이지 않는다.
- 가로지르는 성분(φ)의 θ 사인 성분 = 광축에 수직인 두 축 기울기 오차(시차가 섞여 참고용).
usage: python re/lens_roll_est.py <out.json> <src> ... [--times 2,8,14] [--corr "roll,tx,ty"(도)]"""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbcut import engine  # noqa: E402
from orbcut.engine import SEAM_PPD, SeamStrip, load_lenses  # noqa: E402
from readme_seam_shots import pair_at  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("srcs", nargs="+")
    ap.add_argument("--times", default="2,8,14,20,26")
    ap.add_argument("--corr", default="")
    a = ap.parse_args()
    engine.FLOW_SMOOTH = engine.FLOW_VCLAMP = 0           # 원시 플로우(평활·세로 제한 없이)
    if a.corr:
        os.environ["ORBCUT_LENS1_CORR"] = a.corr
    res = {"per_frame": [], "per_src": {}}
    for src in a.srcs:
        meta = src + ".trailer" if os.path.exists(src + ".trailer") else src
        lenses = load_lenses(meta)
        rolls, tilts = [], []
        for t in [float(x) for x in a.times.split(",")]:
            try:
                pair = pair_at(src, t)
            except Exception:
                continue
            engine._STRIP_MAPS.clear()
            st = SeamStrip(lenses)
            F, C = st.update(pair, np.arange(st.W))
            th = (np.arange(st.W) + 0.5) / SEAM_PPD - 180
            # 열마다: 신뢰도 높은 행의 중앙값(겹침 띠 가운데 ±4°만)
            rows = slice(int((12 - 4) * SEAM_PPD), int((12 + 4) * SEAM_PPD))
            Fc, Cc = F[rows], C[rows]
            good = Cc > 0.6
            ok = good.sum(0) > 10
            if ok.sum() < 200:
                continue
            fx = np.array([np.median(Fc[:, j, 0][good[:, j]]) if ok[j] else np.nan for j in range(st.W)]) / SEAM_PPD
            fy = np.array([np.median(Fc[:, j, 1][good[:, j]]) if ok[j] else np.nan for j in range(st.W)]) / SEAM_PPD
            m = ~np.isnan(fx)
            roll = float(np.median(fx[m]))
            A = np.stack([np.ones(m.sum()), np.cos(np.radians(th[m])), np.sin(np.radians(th[m]))], 1)
            coef = np.linalg.lstsq(A, fy[m], rcond=None)[0]
            rolls.append(roll); tilts.append(coef[1:].tolist())
            res["per_frame"].append({"src": src, "t": t, "roll_deg": round(roll, 3), "cols": int(m.sum()),
                                     "phi_offset": round(float(coef[0]), 3), "tilt_cos": round(float(coef[1]), 3), "tilt_sin": round(float(coef[2]), 3)})
            print(src, t, res["per_frame"][-1], flush=True)
        if rolls:
            res["per_src"][src] = {"rot": [l["rot"].tolist() for l in lenses], "roll_med": round(float(np.median(rolls)), 3),
                                   "roll_iqr": round(float(np.subtract(*np.percentile(rolls, [75, 25]))), 3),
                                   "tilt_med": np.round(np.median(np.array(tilts), 0), 3).tolist(), "n": len(rolls)}
    allr = [f["roll_deg"] for f in res["per_frame"]]
    res["roll_all_med"] = round(float(np.median(allr)), 3) if allr else None
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=1))


if __name__ == "__main__":
    main()
