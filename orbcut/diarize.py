"""화자 분리(소리) + 인물 앵글(화면)을 합친 화자→인물 배정과 화자 컷.

소리 쪽(소니옥스 등 화자 분리 JSON)이 "언제 누가 말했나"(경계)를, 화면 쪽이 "그 화자가 어느 얼굴인가"(대응)를 맡는다.
대응은 구간 전체 통계로 한 번에 정한다. 단서 세 가지:
  1. ASD 대비 — 화자 s가 혼자 말할 때 인물 c의 Light-ASD 점수가 다른 화자가 말할 때보다 얼마나 높은가
  2. 입 움직임 대비 — 같은 대비를 얼굴 아래쪽(입) 프레임 차이로. 옆얼굴·작은 얼굴에서 ASD보다 덜 무너진다
  3. 거리 일관성 — 360 카메라 마이크는 방향을 못 듣지만 거리는 듣는다. 가까운 사람일수록 목소리가 크고 얼굴도 크다
     (음압·화면 크기 모두 거리에 반비례). 배정마다 (음량 dB − 20·log10 얼굴 크기)의 화자 간 흩어짐을 벌점으로 준다.
     카메라를 든 사람(가장 가깝고 옆·아래 얼굴)과 뒷줄 작은 얼굴을 가르는 데 결정적이었다(차 안 실측).
배정은 화자→인물 단사 함수 전수 탐색(화자·인물 수가 작다)으로 총점 최대.

usage: python -m orbcut diarize <orbcut out_dir> <diar.json> [--audio SRC] [--out cut_diar.json] [--map 1=D] [--offset 1.616]
--offset: diar.json 시각이 원본 구간 시작 기준인데 렌더 결과 앞에 키프레임 리드가 붙어 있을 때 그 초만큼 민다.
diar.json: {"tokens": [{"speaker": "1", "text": "..", "start_ms": .., "end_ms": ..}, ...]} (소니옥스 형식)
"""
import glob, itertools, json, math, os, pickle, subprocess

import cv2
import numpy as np

from .cut import ASD_FPS, person_scores, smooth

W_PROX = 0.25          # 거리 일관성 벌점(점/dB). 차량 클립 D구간에서 정답 유지 범위 0.05~1.6의 기하 중간
ID_LINK = 0.8          # 같은 사람으로 묶는 얼굴 신원(SFace) 유사도. 실측 같은 사람 0.84~0.99, 다른 사람 ≤0.73
MIN_SPAN = 0.2         # 화자가 말한 시간 중 그 인물 ID의 경로 기록(span)이 덮는 비율이 이보다 낮으면 배정 후보에서 뺀다 — 잠깐
# 생긴 ID(차량 클립 E 마지막 3초)가 배정을 가로채지 않게. 기각한 대안(차량 9구간 실측): 화면 존재 비율 기준(A 못 고치고 H 회귀),
# 비례 축소(카메라 든 사람 깎임, H 회귀), 절대 프레임 수 축소(짧은 구간 회귀 B·H·I)
MIN_SOLO = 10          # 대비 계산에 필요한 최소 프레임
FRAME_W = 1920         # 인물 앵글(ASD 입력) 폭


def n_frames(cam_dir):
    k = len(glob.glob(os.path.join(cam_dir, "pyframes", "*.jpg")))
    if k:
        return k
    return int(cv2.VideoCapture(os.path.join(cam_dir, "pyavi", "video.avi")).get(cv2.CAP_PROP_FRAME_COUNT))


def speaker_activity(tokens, n, gap=0.35):
    """화자별 발화 프레임(ASD_FPS). 단어 사이 gap초 이하 틈은 같은 발화로 메움."""
    spk = sorted({t["speaker"] for t in tokens if t.get("speaker") and t["text"].strip()})
    V = np.zeros((len(spk), n), bool)
    for t in tokens:
        if not t.get("speaker") or not t["text"].strip():
            continue
        k = spk.index(t["speaker"])
        a, b = int(t["start_ms"] / 1000 * ASD_FPS), int(np.ceil(t["end_ms"] / 1000 * ASD_FPS))
        V[k, max(a, 0):min(b, n)] = True
    g = int(gap * ASD_FPS)
    for k in range(len(spk)):
        idx = np.flatnonzero(V[k])
        for i, j in zip(idx[:-1], idx[1:]):
            if 1 < j - i <= g:
                V[k, i:j] = True
    return spk, V


def face_boxes(cam_dir, n):
    """프레임별 가운데 인물 얼굴 (x, y, s) — 앵글이 따라가는 그 사람. 없으면 nan."""
    tracks = pickle.load(open(os.path.join(cam_dir, "pywork", "tracks.pckl"), "rb"))
    B = np.full((n, 3), np.nan)
    for t in tracks:
        x, y, s = (np.array(t["proc_track"][k], float) for k in "xys")
        if abs(x.mean() - FRAME_W / 2) > FRAME_W * 0.2:
            continue
        for i, f in enumerate(t["track"]["frame"].tolist()):
            if f < n and (np.isnan(B[f, 2]) or s[i] > B[f, 2]):
                B[f] = (x[i], y[i], s[i])
    return B


def _roi(g, cx, cy, w, h, size=(48, 24)):
    x0, x1, y0, y1 = int(cx - w), int(cx + w), int(cy - h), int(cy + h)
    if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0] or x1 - x0 < 8 or y1 - y0 < 8:
        return None
    r = cv2.resize(g[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (r - r.mean()) / (r.std() + 1e-3)


def mouth_motion(cam_dir, n):
    """프레임별 입 움직임 = log(입 영역 프레임 차 / 이마·눈 영역 프레임 차). 머리·카메라 움직임은 분모가 상쇄.
    pywork/mouth.npy에 캐시."""
    cache = os.path.join(cam_dir, "pywork", "mouth.npy")
    video = os.path.join(cam_dir, "pyavi", "video.avi")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(video):   # 앵글을 다시 렌더했으면 무효
        m = np.load(cache)
        if len(m) >= n:
            return m[:n]
    B = face_boxes(cam_dir, n)
    cap = cv2.VideoCapture(os.path.join(cam_dir, "pyavi", "video.avi"))
    out, prev = np.full(n, np.nan), None
    for f in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if np.isnan(B[f, 2]):
            prev = None
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        x, y, s = B[f]
        lo, up = _roi(g, x, y + 0.45 * s, 0.45 * s, 0.25 * s), _roi(g, x, y - 0.35 * s, 0.45 * s, 0.25 * s)
        if lo is None or up is None:
            prev = None
            continue
        if prev is not None:
            out[f] = np.log((np.abs(lo - prev[0]).mean() + 1e-3) / (np.abs(up - prev[1]).mean() + 1e-3))
        prev = (lo, up)
    np.save(cache, out)
    return out


def speech_db(src, n, sr=16000):
    """ASD 프레임마다 음성 대역(200~4000Hz) 에너지 dB."""
    from scipy.signal import butter, sosfilt
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", src, "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = sosfilt(butter(4, [200, 4000], "bandpass", fs=sr, output="sos"), np.frombuffer(raw, np.float32))
    hop = sr // ASD_FPS
    m = min(n, len(x) // hop)
    db = 10 * np.log10((x[:m * hop].reshape(m, hop) ** 2).mean(1) + 1e-12)
    return np.pad(db, (0, n - m), constant_values=np.nan)


def contrast(X, V, solo):
    """M[s, c] = 화자 s 단독 발화 중 신호 평균 − 다른 화자 발화 중 평균 (인물별 z 정규화, nan 무시).
    양쪽 프레임이 MIN_SOLO 이하면 0."""
    Z = (X - np.nanmean(X, 1, keepdims=True)) / (np.nanstd(X, 1, keepdims=True) + 1e-6)
    M = np.zeros((V.shape[0], X.shape[0]))
    for s in range(V.shape[0]):
        on, off = solo[s], V.any(0) & ~V[s]
        for c in range(X.shape[0]):
            a, b = Z[c][on], Z[c][off]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            if len(a) > MIN_SOLO and len(b) > MIN_SOLO:
                M[s, c] = a.mean() - b.mean()
    return M


def identity(cam_dir, n, kit=None, samples=40):
    """인물 앵글의 얼굴 신원(SFace 평균, 정면에 가까운 얼굴만). 3장 미만이면 None. pywork/ident.npy에 캐시."""
    cache = os.path.join(cam_dir, "pywork", "ident.npy")
    video = os.path.join(cam_dir, "pyavi", "video.avi")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(video):
        e = np.load(cache)
        return e if e.size else None
    from .faces import FaceKit
    kit = kit or FaceKit()
    B = face_boxes(cam_dir, n)
    idx = np.flatnonzero(~np.isnan(B[:, 2]))
    cap, embs = cv2.VideoCapture(video), []
    for f in (idx[np.linspace(0, len(idx) - 1, min(samples, len(idx))).astype(int)] if len(idx) else []):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        x, y, _ = B[f]
        faces = kit.detect(fr)
        if not faces:
            continue
        row = min(faces, key=lambda r: math.hypot((r[0] + r[2]) / 2 - x, (r[1] + r[3]) / 2 - y))[5]
        if math.hypot(row[4] - row[6], row[5] - row[7]) / max(row[2], 1) < 0.3:     # 옆얼굴 제외
            continue
        embs.append(kit.embed(fr, row))
    e = np.mean(embs, 0) if len(embs) >= 3 else np.zeros(0)
    np.save(cache, e)
    return e if e.size else None


def link_people(ids, embs):
    """얼굴 신원이 ID_LINK 이상인 인물 ID끼리 한 사람으로 묶는다(건네기 뒤 새 ID·같은 사람 중복 추적). → [[ids], ...]"""
    from .faces import FaceKit
    parent = list(range(len(ids)))
    find = lambda i: i if parent[i] == i else find(parent[i])
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if embs[i] is not None and embs[j] is not None and FaceKit.sim(embs[i], embs[j]) >= ID_LINK:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def assign(evidence, loud_db, face_db, w_prox=W_PROX, group=None, fixed=None):
    """화자→인물 단사 배정 전수 탐색. 점수 = Σ 화면 증거 − w·std(음량 dB − 얼굴 크기 dB). 반환 (배정, 점수표).
    group[c]: 인물 ID c가 속한 사람(얼굴 신원으로 묶음) — 한 사람에게 화자 둘을 주지 않는다(차량 클립 A: 같은 사람 F·I에 화자 1·3)."""
    ns, nc = evidence.shape
    ranked = []
    for perm in itertools.permutations(range(nc), min(ns, nc)):
        if group is not None and len({group[c] for c in perm}) < len(perm):
            continue
        if fixed and any(perm[s] != c for s, c in fixed.items() if s < len(perm)):
            continue
        vis = float(sum(evidence[s, perm[s]] for s in range(len(perm))))
        resid = [loud_db[s] - face_db[perm[s]] for s in range(len(perm)) if np.isfinite(loud_db[s])]
        prox = float(np.std(resid)) if len(resid) > 1 else 0.0
        ranked.append((vis - w_prox * prox, vis, prox, perm))
    ranked.sort(key=lambda r: -r[0])
    return ranked[0][3], ranked


def load_tokens(diar_json, offset=0.0):
    """화자 분리 토큰. offset초만큼 뒤로 민다 — 구간 렌더 앞에 붙은 키프레임 리드(무손실 자르기)를 맞출 때."""
    toks = json.load(open(diar_json))["tokens"]
    if offset:
        d = offset * 1000.0
        toks = [dict(t, start_ms=t["start_ms"] + d, end_ms=t["end_ms"] + d) for t in toks]
    return toks


def diarize(out_dir, diar_json, audio=None, min_shot=1.5, gap=0.35, overrides=None, w_prox=W_PROX, offset=0.0, link=True):
    cams = sorted(d for d in glob.glob(os.path.join(out_dir, "cams", "cam_*")) if os.path.isdir(d))
    ids = [os.path.basename(d)[4:] for d in cams]
    n = min(n_frames(d) for d in cams)
    if audio is None:
        audio = json.load(open(os.path.join(out_dir, "manifest.json")))["src"]

    present = np.ones((len(cams), n), bool)
    span = np.ones((len(cams), n), bool)              # 그 ID의 경로 기록이 있는 시간(건네기 뒤 새로 생긴 ID는 그 뒤부터)
    pj = os.path.join(out_dir, "paths.json")
    if os.path.exists(pj):                          # track 모드: 그 사람이 화면에서 사라진 구간은 증거에서 뺀다
        P = json.load(open(pj))["paths"]
        tt = np.arange(n) / ASD_FPS
        for k, pid in enumerate(ids):
            rows = P.get(pid)
            if rows and len(rows[0]) >= 5:
                # 증거 마스크는 M16 그대로(경로 기록 앞은 첫 값으로 채움) — 기록 밖을 0으로 바꾸면 걷는 장면(차량 클립 H)의
                # 배정이 흔들렸다. 기록 밖 ID가 배정을 가로채는 문제는 아래 MIN_SPAN이 맡는다
                present[k] = np.interp(tt, [r[0] for r in rows], [1.0 if r[4] else 0.0 for r in rows]) > 0.5
                span[k] = (tt >= rows[0][0] - 0.5) & (tt <= rows[-1][0] + 0.5)
    asd = np.stack([smooth(person_scores(d, n), int(0.4 * ASD_FPS)) for d in cams])
    asd[~present] = np.nan
    mm = np.stack([mouth_motion(d, n) for d in cams])
    k = int(0.4 * ASD_FPS)
    mm = np.stack([np.where(np.isnan(x), np.nan, np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), np.ones(k) / k, "same"))
                   for x in mm])
    mm[~present] = np.nan

    spk, V = speaker_activity(load_tokens(diar_json, offset), n, gap)
    solo = V & (V.sum(0) == 1)
    M_asd, M_mouth = contrast(asd, V, solo), contrast(mm, V, solo)
    ev = M_asd + M_mouth
    for s in range(len(spk)):                        # 그 화자가 말하는 동안 기록이 거의 없던 ID는 후보에서 뺀다
        cover = span[:, solo[s]].mean(1) if solo[s].any() else np.ones(len(ids))
        ev[s, cover < MIN_SPAN] = -1e3
    db = speech_db(audio, n)
    loud = np.array([np.nanmedian(db[solo[s]]) if solo[s].sum() > MIN_SOLO else np.nan for s in range(len(spk))])
    face = np.array([20 * np.log10(np.nanmedian(face_boxes(d, n)[:, 2])) for d in cams])
    groups = link_people(ids, [identity(d, n) for d in cams]) if link else [[i] for i in range(len(ids))]
    group = {i: gi for gi, g in enumerate(groups) for i in g}
    auto_perm, ranked0 = assign(ev, loud, face, w_prox, group)
    auto = {spk[s]: ids[auto_perm[s]] for s in range(len(auto_perm)) if ev[s, auto_perm[s]] > -1e2}
    # 덮어쓰기(육안 확정)는 고정 제약 — 나머지 화자는 그 아래에서 다시 배정(같은 사람·같은 ID 중복 없이)
    fixed = {}
    for kv in filter(None, (overrides or "").split(",")):
        a, b = kv.split("=")
        if a in spk and b in ids:
            fixed[spk.index(a)] = ids.index(b)
            ev[spk.index(a), ids.index(b)] = max(ev[spk.index(a), ids.index(b)], 0.0)
    perm, ranked = assign(ev, loud, face, w_prox, group, fixed) if fixed else (auto_perm, ranked0)
    mapping = {spk[s]: ids[perm[s]] for s in range(len(perm)) if ev[s, perm[s]] > -1e2}
    for s in spk:
        mapping.setdefault(s, "W")                  # 화자가 인물보다 많으면 남는 화자는 와이드
        auto.setdefault(s, "W")
    margin = round(ranked0[0][0] - ranked0[1][0], 3) if len(ranked0) > 1 else None
    # 사람 잇기(렌더): 배정된 ID의 경로 기록 밖(건네기 전·뒤)에는 얼굴 신원이 같은 다른 ID로 넘긴다
    same = {ids[i]: sorted(g, key=lambda j: -span[j].sum()) for g in groups for i in g}
    gnames = ["+".join(ids[i] for i in g) for g in groups if len(g) > 1]

    def cam_for(name, f):
        if name == "W" or name not in same:
            return name if name in ids else "W"
        if span[ids.index(name), f]:
            return name
        for j in same[name]:
            if span[j, f]:
                return ids[j]
        return "W"                                  # 그 사람이 지금 기록에 없음(카메라 건네기 중·시야 밖) → 와이드

    # 프레임별 카메라: 혼자 말하면 그 사람, 겹치면 현재 샷 유지(없으면 와이드), 침묵이면 유지
    cur, lab = None, []
    for f in range(n):
        act = [spk[s] for s in range(len(spk)) if V[s, f]]
        if len(act) == 1:
            cur = cam_for(mapping[act[0]], f)
        elif len(act) > 1:
            now = {cam_for(mapping[x], f) for x in act}
            cur = cur if cur in now else "W"
        lab.append(cur or "W")
    segs = []
    for f, x in enumerate(lab):
        if segs and segs[-1][2] == x:
            segs[-1][1] = f + 1
        else:
            segs.append([f, f + 1, x])
    mf, merged = int(min_shot * ASD_FPS), []
    for s in segs:                                   # 짧은 샷은 앞 샷에 흡수
        if merged and (s[1] - s[0] < mf or merged[-1][2] == s[2]):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    r3 = lambda A: [[round(float(v), 3) for v in row] for row in A]
    return {"mapping": mapping, "auto_mapping": auto, "margin": margin, "speakers": spk, "cams": ids, "linked": gnames,
            "evidence": {"asd": r3(M_asd), "mouth": r3(M_mouth),
                         "loud_db": [round(float(v), 1) for v in loud], "face_db": [round(float(v), 1) for v in face]},
            "runner_up": {spk[s]: ids[ranked0[1][3][s]] for s in range(len(ranked0[1][3]))} if len(ranked0) > 1 else None,
            "talk_seconds": {spk[s]: round(float(V[s].sum()) / ASD_FPS, 1) for s in range(len(spk))},
            "segments": [{"start": round(s / ASD_FPS, 2), "end": round(e / ASD_FPS, 2), "cam": x} for s, e, x in merged]}


def to_edit(r, wide_id="wide"):
    """diarize 결과 → cut 단계와 같은 edit.json 형식. 와이드(W)는 실내 광각 마스터 id로."""
    segs = [{"start": x["start"], "end": x["end"], "cam": wide_id if x["cam"] == "W" else x["cam"]} for x in r["segments"]]
    return {"talk_seconds": r["talk_seconds"], "segments": segs, "source": "diarize",
            "mapping": r["mapping"], "auto_mapping": r["auto_mapping"], "margin": r["margin"], "evidence": r["evidence"]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="화자 분리 + 화면 증거 화자 컷 → edit.json (파이프라인 cut 단계 대체)")
    ap.add_argument("out_dir"); ap.add_argument("diar")
    ap.add_argument("--edit", required=True); ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--map", default=""); ap.add_argument("--min-shot", type=float, default=1.5)
    a = ap.parse_args(argv)
    r = diarize(a.out_dir, a.diar, min_shot=a.min_shot, overrides=a.map, offset=a.offset)
    json.dump(to_edit(r), open(a.edit, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: r[k] for k in ("mapping", "auto_mapping", "margin")}, ensure_ascii=False), len(r["segments"]), "shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
