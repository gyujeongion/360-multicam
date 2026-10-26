"""M14: 추적 모드 — 차 안 테스트 클립 40~70초(50초대 손으로 크게 흔듦).
인물별 앵글(1080p, 경로 기준 pitch 0)에서, 얼굴이 보이는 프레임 기준
  ① 얼굴 중심이 화면 가운데 1/3 영역 ② 처음 잡힌 사람과 같은 신원(SFace 얼굴 임베딩, 인물 바뀜 없음) 비율. 뒤통수만 보이는 프레임은 판정 제외.
  판정은 그 인물이 '존재'하는 구간만(카메라를 건네받아 시야에서 사라진 구간은 앵글이 정지하고 컷 대상에서도 빠짐).
주요 인물(카메라에 가장 가까운 2명 = 인물 검출 단계 얼굴 크기 상위 2명 = 나·친구) 모두 ≥ 90%.
(결과 비율로 고르면 스스로 유리하게 고르는 셈이라, 렌더 전 정보인 얼굴 크기로 미리 정한다)
원격 호스트(config remote_host)가 있으면 그쪽에 위임, 결과 work/trk016/out (없으면 생성)."""
import json, os, subprocess, sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "work", "trk016", "out")
SRC = os.path.join(REPO, "work", "trk016", "source.mkv")


def _remote():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _remote import delegate
    return delegate(".venv/bin/python -m pytest tests/test_track.py -q -s")


def _check(video, kit, present=None, fps=2.0, ref_n=3):
    """얼굴이 보이는 프레임에서: 가운데 1/3 비율, 처음 잡힌 사람과 같은 신원(SFace) 비율. 뒤통수 프레임은 제외."""
    import cv2
    import numpy as np
    from orbcut.faces import SAME_PERSON
    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(vfps / fps)))
    n = seen = centered = same = frontal = i = 0
    refs = []
    while True:
        if not cap.grab():
            break
        if i % step == 0 and (present is None or present(i / vfps)):
            _, fr = cap.retrieve()
            # 추적기와 같은 960 폭에서 검출: YuNet은 화면을 가득 채운 큰 얼굴(1080p 근접 인물)에서 박스를 잘못 잡는다
            # (실측: 같은 B 프레임이 1920폭 78% / 960폭 95% — 렌더는 파노라마와 픽셀 동일 확인, re/diag_scale_metric.py)
            full = fr
            sc = fr.shape[1] / 960
            fr = cv2.resize(fr, (960, 960 * fr.shape[0] // fr.shape[1]))
            h, w = fr.shape[:2]
            n += 1
            faces = kit.detect(fr)
            if faces:
                x1, y1, x2, y2, _, row = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                seen += 1
                centered += (w / 3 <= cx <= 2 * w / 3) and (h / 3 <= cy <= 2 * h / 3)
                # 신원 판정은 정면에 가까운 얼굴만 (두 눈 간격 / 얼굴 폭 ≥ 0.35) — 옆얼굴은 SFace 점수가 원래 낮음
                eye = float(np.hypot(row[4] - row[6], row[5] - row[7]))
                if eye / max(row[2], 1) >= 0.35:
                    frontal += 1
                    full_row = row.copy(); full_row[:14] *= sc          # 박스·랜드마크를 원본 해상도로 → 신원 임베딩은 원본 화질로
                    emb = kit.embed(full, full_row)
                    if len(refs) < ref_n:
                        refs.append(emb); same += 1
                    else:
                        same += kit.sim(np.mean(refs, axis=0), emb) >= SAME_PERSON
        i += 1
    return {"frames": n, "seen": seen, "frontal": frontal, "centered": centered / max(seen, 1),
            "same_person": (same / frontal) if frontal >= 5 else None}       # 정면 얼굴이 거의 없으면(운전자) 판정 불가


def test_tracked_faces_stay_centered():
    if not os.path.exists(SRC):
        rc = _remote()
        if rc is None:
            pytest.skip("픽스처 없음(work/trk016/source.mkv)·원격 미설정")
        assert rc == 0
        return
    m_path = os.path.join(OUT, "manifest.json")
    if not (os.path.exists(m_path) and json.load(open(m_path)).get("status") == "ok"):
        r = subprocess.run([sys.executable, "-m", "orbcut", "--json", "all", SRC, "--out", OUT, "--meta", SRC + ".trailer",
                            "--mode", "track", "--src-start", "40"], cwd=REPO, capture_output=True, text=True)
        assert json.load(open(m_path))["status"] == "ok", r.stdout[-500:] + r.stderr[-800:]
    from orbcut.faces import FaceKit
    kit = FaceKit()
    people = json.load(open(os.path.join(OUT, "people.json")))["people"]
    paths = json.load(open(os.path.join(OUT, "paths.json")))["paths"]

    def present_fn(pid):
        rows = paths.get(pid, [])
        def f(t):
            if not rows:
                return True
            j = min(range(len(rows)), key=lambda k: abs(rows[k][0] - t))
            return len(rows[j]) < 5 or rows[j][4]
        return f
    res = {p["id"]: _check(os.path.join(OUT, "cams", f"cam_{p['id']}.mp4"), kit, present_fn(p["id"])) for p in people}
    for k, r in res.items():
        sp = "판정불가" if r["same_person"] is None else f"{r['same_person']:.0%}"
        print(f"{k}: 존재 프레임 {r['frames']} · 얼굴 보임 {r['seen']} (정면 {r['frontal']}) · 가운데 {r['centered']:.0%} · 같은 사람 {sp}")
    main = [p["id"] for p in sorted(people, key=lambda p: -p["face_h"])[:2]]
    for k in main:
        r = res[k]
        assert r["seen"] >= 10, f"{k}: 얼굴 보이는 프레임 부족 {r}"
        assert r["centered"] >= 0.9 and (r["same_person"] is None or r["same_person"] >= 0.9), f"{k}: {r}"


def _shake(pano, path_rows, fixed, seg_end, fps=10.0):
    """안정화 파노라마에서 경로(또는 고정 방향)로 100° 뷰를 잘라, 프레임 간 특징점 이동량 중앙값의 평균(픽셀).
    seg_end: 첫 카메라 이동 전까지만(실내 기준이 바뀌기 전)."""
    import cv2
    import numpy as np
    from orbcut.engine import flat_dirs
    t_ = np.array([r[0] for r in path_rows]); y_ = np.degrees(np.unwrap(np.radians([r[1] for r in path_rows])))
    p_ = np.array([r[2] for r in path_rows])
    cap = cv2.VideoCapture(pano); vfps = cap.get(cv2.CAP_PROP_FPS); step = max(1, int(round(vfps / fps)))
    prev, mags, i = None, [], 0
    while True:
        if not cap.grab():
            break
        t = i / vfps
        if t > seg_end:
            break
        if i % step == 0:
            _, eq = cap.retrieve(); H, W = eq.shape[:2]
            yaw, pit = (y_[0], p_[0]) if fixed else (float(np.interp(t, t_, y_)), float(np.interp(t, t_, p_)))
            d = flat_dirs(yaw, pit, 100, 640, 360).reshape(-1, 3)
            lon = np.arctan2(d[:, 0], d[:, 2]); lat = np.arcsin(np.clip(d[:, 1], -1, 1))
            mx = ((lon + np.pi) / (2 * np.pi) * W - .5).astype(np.float32).reshape(360, 640)
            my = ((lat + np.pi / 2) / np.pi * H - .5).astype(np.float32).reshape(360, 640)
            g = cv2.cvtColor(cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
                if pts is not None:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
                    ok = st.ravel() == 1
                    if ok.sum() >= 20:
                        mags.append(float(np.median(np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1))))
            prev = g
        i += 1
    return float(np.mean(mags)), len(mags)


def test_interior_wide_steadier():
    """실내 고정 광각: 좌석 공통 회전 δ(t)를 더한 경로가 자이로 단독(고정 방향)보다 실내 흔들림을 ≥10% 줄인다.
    (사전 고정 기준. 창밖 풍경 특징점도 섞이지만 중앙값이라 실내 다수가 지배)"""
    if not os.path.exists(SRC):
        pytest.skip("픽스처 없음(원격에서 test_tracked_faces_stay_centered가 함께 실행)")
    pj = json.load(open(os.path.join(OUT, "paths.json")))
    rows = pj.get("interior")
    assert rows, "paths.json에 interior 없음"
    assert os.path.exists(os.path.join(OUT, "masters", "cam_wide.mp4")), "실내 광각 마스터 없음"
    # 첫 카메라 이동(어떤 인물이든 present가 꺼지는 첫 시각) 전까지
    ends = [r[0] for v in pj["paths"].values() for r in v if len(r) >= 5 and not r[4]]
    seg_end = min(ends) if ends else rows[-1][0]
    pano = os.path.join(OUT, "panorama.mp4")
    gyro, n1 = _shake(pano, rows, True, seg_end)
    track, n2 = _shake(pano, rows, False, seg_end)
    print(f"실내 흔들림(특징점 이동 중앙값 평균, 0~{seg_end:.1f}s): 자이로 단독 {gyro:.2f}px · 실내 경로 {track:.2f}px "
          f"({(1 - track / gyro):+.0%} 감소, n={n1}/{n2})")
    assert track <= 0.9 * gyro


def test_handoff_boundary_from_imu():
    """M19: 카메라 건네기 경계 — 화면 육안 정답 19.5~20.0초(trk016, 원본 59.5초 부근). IMU 급회전 묶음 시작으로 당겨
    지연 ≤1초(종전: 배정 실패 2초 누적 후라 22.9초, 3.1초 지연). 캐시된 얼굴 검출 재사용이라 가볍다."""
    pano = os.path.join(OUT, "panorama.mp4")
    if not os.path.exists(pano + ".dets_4.pkl"):
        pytest.skip("검출 캐시 없음(원격에서 test_tracked_faces_stay_centered가 함께 실행)")
    from orbcut.track import TRACK_FPS, track_seats
    people = json.load(open(os.path.join(OUT, "people.json")))["people"]
    new = [s["start"] for s in track_seats(pano, people, TRACK_FPS, return_segments=True)[2][1:]]
    old = [s["start"] for s in track_seats(pano, people, TRACK_FPS, bursts=[], return_segments=True)[2][1:]]
    truth = 19.75
    lag = lambda starts: min((s - truth for s in starts if s >= truth - 1.0), default=99)
    print(f"건네기 경계: IMU {new} (지연 {lag(new):+.2f}s) · 종전 {old} (지연 {lag(old):+.2f}s)")
    assert abs(lag(new)) <= 1.0 and lag(old) >= 2.0
