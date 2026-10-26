"""단계 오케스트레이션 + manifest.json (AI·사람 모두 이 파일 하나로 상태 판독).

각 단계는 별도 프로세스(`python -m multicam360.<모듈>`)로 실행 — 작업 디렉토리·메모리 격리.
"""
import hashlib, json, os, subprocess, sys, time

from .config import REPO, asd_dir

PY = sys.executable
STAGES = ["probe", "slice", "panorama", "people", "angles", "asd", "cut", "masters", "xml"]


class StageError(RuntimeError):
    def __init__(self, stage, msg):
        super().__init__(f"[{stage}] {msg}")
        self.stage, self.msg = stage, msg


class Manifest:
    def __init__(self, out):
        self.path = os.path.join(out, "manifest.json")
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {"stages": {}}

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.data, open(tmp, "w"), indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def stage(self, name, **kw):
        self.data["stages"].setdefault(name, {}).update(kw)
        self.save()


def run(stage, args, cwd=None, log=None):
    with open(log, "w") if log else open(os.devnull, "w") as lf:
        r = subprocess.run(args, cwd=cwd or REPO, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = open(log).read()[-1500:] if log else ""
        raise StageError(stage, f"exit {r.returncode}. 로그 끝:\n{tail}")


def mod(name, *a):
    return [PY, "-m", f"multicam360.{name}", *map(str, a)]


def parse_range(r):
    if not r:
        return None, None
    a, b = r.split(":")
    t = lambda x: sum(float(p) * 60 ** i for i, p in enumerate(reversed(x.split("-"))))  # 4-00 → 240
    return t(a), t(b) - t(a)


DEFAULTS = {
    "panorama": {"width": 3840},
    "angles": {"hfov": 70, "pitch": 0},
    "cut": {"min_shot": 1.0, "hold": 1.0, "smooth": 0.8, "thr": -0.5},   # M7: 사람 정답 3분 기준 튜닝(앞절반 선택·뒤절반 검증 90.0%), re/tune_cut.py
    "masters": {"hfov": 90, "w": 3840, "h": 2160, "pitch": 8, "bitrate": "25M", "proj": "pannini", "pannini_d": 0.7, "wide_hfov": 110},
    "xml": {"face_ratio": 0.32, "eye_line": 0.38, "hfov_min": 55, "hfov_max": 85},
}


def media_duration(path):
    """실제 재생 길이 = 소리 마지막 패킷 끝. mkv 무손실 자르기는 머리말 길이가 요청값(-t)으로 적혀 실제보다 짧다
    (실측: 머리말 180.0초, 영상 5448프레임·소리 끝 181.76초 — 영상·소리는 서로 맞음)."""
    head = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", path]).decode())
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-read_intervals", f"{max(0, head - 3)}%",
                                   "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", path]).decode().split()
    ends = [float(a) + float(b) for a, b in (r.split(",")[:2] for r in out if r.count(",") >= 1 and "N/A" not in r)]
    return max(ends + [head])


def load_params(out, overrides=None):
    """기본값 ← <out>/multicam360.json ← 호출 인자. 단계별 해시에 들어가 바뀐 단계만 다시 돈다."""
    params = json.loads(json.dumps(DEFAULTS))
    cfg = os.path.join(out, "multicam360.json")
    for src in ([json.load(open(cfg))] if os.path.exists(cfg) else []) + [overrides or {}]:
        for stage, kv in src.items():
            params.setdefault(stage, {}).update(kv)
    return params


def fingerprint(paths):
    """입력 파일 지문: (경로, 크기, 수정시각). 폴더면 결과물 파일(.mp4/.json/.pckl)만 2단계 깊이까지."""
    items = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                if root[len(p):].count(os.sep) >= 3:
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if d not in ("pyframes", "pycrop", "pyavi")]
                items += [os.path.join(root, f) for f in files if f.endswith((".mp4", ".json", ".pckl"))]
        elif os.path.exists(p):
            items.append(p)
    # 상대경로로 바꾼 뒤 정렬: 같은 파일을 절대·상대 경로로 불러도 키가 같아야 한다
    # (실측: 정렬 후 변환이라 순서가 뒤바뀌어 캐시가 엇갈림 → 앵글·화자감지 불필요 재실행)
    return sorted((os.path.relpath(f), os.path.getsize(f), int(os.path.getmtime(f))) for f in items)


_LOCKS = []


def _lock_out(out):
    """한 결과 폴더엔 한 실행만. 두 실행이 같은 앵글·화자감지 파일을 동시에 쓰면 디스크 대기로 멈춘다
    (실측 2026-10-22: 검증 위임 + 수동 재실행이 겹쳐 화자감지 4개가 2시간 반 정지)."""
    import fcntl
    f = open(os.path.join(out, ".multicam360.lock"), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"다른 multicam360 실행이 이 폴더를 쓰는 중: {out} — 끝난 뒤 다시 실행")
    _LOCKS.append(f)                       # 프로세스가 끝날 때까지 잠금 유지


def _guard_local_length(src, rng):
    """긴 구간을 이 기기에서 돌리지 않게 막는다(설정 local_max_s, 기본 60초). 원격은 --host로."""
    from .config import local_max_s
    limit = local_max_s()
    if not limit or not os.path.exists(src):
        return
    start, dur = parse_range(rng)
    if dur is None:
        try:
            dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                 "-of", "csv=p=0", src]).decode())
        except (subprocess.CalledProcessError, ValueError):
            return
    if dur > limit:
        raise RuntimeError(f"구간 {dur:.0f}초 > 이 기기 한도 {limit:.0f}초 — 메모리 고갈 위험. "
                           f"--host <원격 호스트>로 실행하거나, 렌더 전용 기기면 config local_max_s=0")


def run_all(src, out, meta=None, rng=None, eq_width=3840, jobs=None, overrides=None, force=None, mode="table",
            src_start=0.0, until=None):
    """mode: table = 카메라 고정(테이블 가운데) / track = 손에 든·차 안 — 자이로 안정화 + 인물별 얼굴 경로 추적."""
    os.makedirs(out, exist_ok=True)
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    _lock_out(out)
    P = lambda *p: os.path.join(out, *p)
    m = Manifest(out)
    params = load_params(out, overrides)
    params["panorama"]["width"] = eq_width
    # 렌더 프로세스마다 원본 8K를 따로 디코딩한다 → 동시 프로세스 = 동시 디코더. 상한 MULTICAM360_MAX_DEC(기본 2)
    # (실측: 4개 동시 → 원격 부하 42/20, 맥북에선 ffmpeg 8개 약 50GB로 메모리 고갈)
    max_dec = int(os.environ.get("MULTICAM360_MAX_DEC", 2))
    jobs = min(jobs or max(1, min(5, (os.cpu_count() or 4) // 4)), max_dec)
    meta_path = meta or src
    _guard_local_length(src, rng)
    m.data.update({"mode": mode, "src": os.path.abspath(src) if os.path.exists(src) else src, "range": rng,
                   "status": "running", "error": None, "params": params})
    m.save()
    force = set(force or [])
    state = {"current": None}

    def step(name, inputs, outputs, fn, extra=None):
        """입력 지문+파라미터가 같고 산출물이 있으면 건너뜀(cached). 아니면 실행 후 해시 기록."""
        if state.get("stop"):
            return {}
        key = hashlib.sha1(json.dumps([params.get(name), extra, fingerprint(inputs)], default=str).encode()).hexdigest()
        st = m.data["stages"].get(name, {})
        if name not in force and st.get("status") == "ok" and st.get("key") == key and all(map(os.path.exists, outputs)):
            m.stage(name, cached=True)
            if until == name:                       # 캐시로 건너뛴 단계에서도 --until은 멈춰야 한다
                state["stop"] = True                # (실측: 컷 캐시 시 마스터 4K 렌더까지 진행)
            return st
        state["current"] = name
        m.stage(name, status="running", t0=time.time(), cached=False, error=None)
        info = fn() or {}
        m.stage(name, status="ok", t1=time.time(), key=key, outputs=outputs, **info)
        if until == name:
            state["stop"] = True
        return m.data["stages"][name]

    try:
        step("probe", [meta_path], [P("probe.json")],
             lambda: run("probe", mod("probe", meta_path, P("probe.json")), log=P("logs", "probe.log")))

        start, dur = parse_range(rng)
        seg = P("source.mkv") if start is not None else src

        def do_slice():
            if start is not None:
                run("slice", ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", src, "-t", str(dur), "-map", "0:0",
                              "-map", "0:1", "-map", "0:a:0", "-c", "copy", seg], log=P("logs", "slice.log"))
            # 무손실 자르기는 직전 키프레임부터 시작 → 실제 길이 기록 (컷 시각은 이 파일 기준)
            return {"requested_start_s": start, "requested_duration_s": dur, "actual_duration_s": round(media_duration(seg), 3)}
        step("slice", [src], [seg], do_slice, extra=[start, dur])

        if mode == "track":
            # 프레임별 카메라 자세(자이로+수평) + 영상-자이로 시간차(구간 자르기 오차 포함)
            step("imu", [seg, meta_path], [P("imu.npz")],
                 lambda: run("imu", mod("track", "imu", seg, P("imu.npz"), "--meta", meta_path,
                                        "--start", (start if start is not None else 0) + (src_start or 0)),
                                 log=P("logs", "imu.log")))
            step("panorama", [seg, meta_path, P("imu.npz")], [P("panorama.mp4")],
                 lambda: run("panorama", mod("track", "pano", seg, P("imu.npz"), P("panorama.mp4"), "--meta", meta_path,
                                             "--width", params["panorama"]["width"]), log=P("logs", "panorama.log")))
        else:
            # 파노라마: 인물 검출 + 라벨링용 (엔진, 원본에서 직접, 이음새 플로우)
            step("panorama", [seg, meta_path], [P("panorama.mp4")],
                 lambda: run("panorama", mod("render", "panorama", seg, P("panorama.mp4"), "--meta", meta_path,
                                             "--width", params["panorama"]["width"]), log=P("logs", "panorama.log")))

        def do_people():
            dur_s = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                   "-of", "csv=p=0", P("panorama.mp4")]).decode())
            times = ",".join(str(round(dur_s * (i + 0.5) / 8, 2)) for i in range(8))
            run("people", mod("people", P("panorama.mp4"), P("people.json"), "--equirect", "--times", times),
                log=P("logs", "people.log"))
            n = len(json.load(open(P("people.json")))["people"])
            if not n:
                raise StageError("people", "인물을 찾지 못함 (얼굴이 너무 작거나 어두움)")
            return {"people": n}
        step("people", [P("panorama.mp4")], [P("people.json")], do_people)
        people = json.load(open(P("people.json")))["people"]

        people_for_xml = P("people.json")
        a = params["angles"]
        if mode == "track":
            def do_paths():
                run("paths", mod("track", "paths", P("panorama.mp4"), P("people.json"), P("paths.json")),
                    log=P("logs", "paths.log"))
                # 각 앵글은 자기 경로 기준 좌표로 렌더 → XML 구도 계산엔 모든 인물 yaw 0
                pv = json.load(open(P("people.json")))
                known = {p["id"]: p for p in pv["people"]}
                ids = list(json.load(open(P("paths.json")))["paths"])      # 카메라 이동 후 새로 나타난 인물 포함
                pv["people"] = [{**known.get(i, {"id": i, "face_h": 0.0, "seen": 0}), "yaw": 0.0} for i in ids]
                json.dump(pv, open(P("people_view.json"), "w"), indent=2)
                return {"found_ratio": json.load(open(P("paths.json")))["found_ratio"]}
            step("paths", [P("panorama.mp4"), P("people.json")], [P("paths.json"), P("people_view.json")], do_paths)
            people_for_xml = P("people_view.json")
            step("angles", [seg, P("paths.json"), P("imu.npz")], [P("cams", "render.json")],
                 lambda: run("angles", mod("track", "render", seg, P("imu.npz"), P("paths.json"), P("cams"),
                                           "--meta", meta_path, "--hfov", a["hfov"], "--pitch", 0, "--jobs", jobs,
                                           "--wide-hfov", 0),          # 화자 감지용엔 실내 광각 불필요
                             log=P("logs", "angles.log")))
        else:
            # 인물 앵글(화자 감지용): 플로우 생략(속도)
            step("angles", [seg, P("people.json")], [P("cams", "render.json")],
                 lambda: run("angles", mod("render", "views", seg, P("people.json"), P("cams"), "--meta", meta_path,
                                           "--hfov", a["hfov"], "--pitch", a["pitch"], "--jobs", jobs),
                             log=P("logs", "angles.log")))

        if mode == "track" and os.path.exists(P("people_view.json")):
            people = json.load(open(P("people_view.json")))["people"]

        def do_asd():
            env = {**os.environ, "ASD_NOVIS": "1"}
            bad = []
            for w in range(0, len(people), max_dec):                     # 동시 max_dec명씩(각자 ffmpeg 추출)
                procs = [(p["id"], subprocess.Popen(
                    [PY, "-W", "ignore", "Columbia_test.py", "--videoName", f"cam_{p['id']}", "--videoFolder", P("cams")],
                    cwd=asd_dir(), env=env, stdout=open(P("logs", f"asd_{p['id']}.log"), "w"), stderr=subprocess.STDOUT))
                    for p in people[w:w + max_dec]]
                bad += [pid for pid, pr in procs if pr.wait() != 0]
            if bad:
                raise StageError("asd", f"Light-ASD 실패: {bad} (logs/asd_*.log)")
        step("asd", [P("cams", f"cam_{p['id']}.mp4") for p in people],
             [P("cams", f"cam_{p['id']}", "pywork", "scores.pckl") for p in people], do_asd)

        c = params["cut"]
        cut_extra = ["--paths", P("paths.json")] if mode == "track" else []
        if c.get("diar"):       # 화자 분리 JSON이 있으면 소리 경계 + 화면 증거 배정(diarize)으로 컷
            step("cut", [P("cams"), c["diar"]] + ([P("paths.json")] if mode == "track" else []), [P("edit.json")],
                 lambda: run("cut", mod("diarize", out, c["diar"], "--edit", P("edit.json"),
                                        "--offset", c.get("diar_offset", 0.0), "--map", c.get("diar_map", ""),
                                        "--min-shot", c.get("diar_min_shot", 1.5)), log=P("logs", "cut.log")))
        else:
            step("cut", [P("cams")] + ([P("paths.json")] if mode == "track" else []), [P("edit.json")],
                 lambda: run("cut", mod("cut", P("cams"), P("edit"), "--min-shot", c["min_shot"], "--hold", c["hold"],
                                        "--smooth", c["smooth"], "--thr", c["thr"], *cut_extra), log=P("logs", "cut.log")))

        # 4K 광각 마스터 — 프리미어에서 확대해도 추가 손실 없음
        ms = params["masters"]
        if mode == "track":
            step("masters", [seg, P("paths.json"), P("imu.npz")], [P("masters", "render.json")],
                 lambda: run("masters", mod("track", "render", seg, P("imu.npz"), P("paths.json"), P("masters"),
                                            "--meta", meta_path, "--hfov", ms["hfov"], "--w", ms["w"], "--h", ms["h"],
                                            "--pitch", ms.get("track_pitch", -4), "--bitrate", ms["bitrate"],
                                            "--proj", ms["proj"], "--pannini-d", ms["pannini_d"], "--jobs", jobs,
                                            "--wide-hfov", ms.get("wide_hfov", 110)),   # 0이면 실내 광각 생략
                             log=P("logs", "masters.log")))
        else:
            step("masters", [seg, P("people.json")], [P("masters", "render.json")],
                 lambda: run("masters", mod("render", "views", seg, P("people.json"), P("masters"), "--meta", meta_path,
                                            "--hfov", ms["hfov"], "--w", ms["w"], "--h", ms["h"], "--pitch", ms["pitch"],
                                            "--bitrate", ms["bitrate"], "--flow", "--jobs", jobs,
                                            "--proj", ms["proj"], "--pannini-d", ms["pannini_d"]),
                             log=P("logs", "masters.log")))

        x = params["xml"]
        step("xml", [people_for_xml, P("edit.json"), P("masters", "render.json"), P("cams")],
             [P("premiere.xml"), P("premiere_shots.json"), P("preview.mp4")],
             lambda: run("xml", mod("masters_xml", people_for_xml, P("cams"), P("edit.json"), P("masters"),
                                    P("premiere.xml"), "--preview", P("preview.mp4"), "--face-ratio", x["face_ratio"],
                                    "--eye-line", x["eye_line"], "--hfov-min", x["hfov_min"],
                                    "--hfov-max", x["hfov_max"]), log=P("logs", "xml.log")))

        if state.get("stop"):
            m.data.update(status="partial", stopped_after=until); m.save()
            return m.data
        m.data.update(status="ok", outputs={"premiere_xml": P("premiere.xml"), "shots": P("premiere_shots.json"),
                                            "preview": P("preview.mp4"), "masters": P("masters"),
                                            "panorama": P("panorama.mp4")})
        m.save()
        return m.data
    except StageError as e:
        m.stage(e.stage, status="failed", error=e.msg)
        m.data.update(status="failed", error=str(e)); m.save()
        raise
    except Exception as e:
        m.stage(state["current"] or "?", status="failed", error=repr(e))
        m.data.update(status="failed", error=repr(e)); m.save()
        raise StageError(state["current"] or "?", repr(e))


def doctor_local():
    import shutil
    probs = [f"{t} 없음" for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    for f in ("weight/pretrain_AVA_CVPR.model", "model/faceDetector/s3fd/sfd_face.pth"):
        if not os.path.exists(os.path.join(asd_dir(), f)):
            probs.append(f"Light-ASD 파일 없음: {f} (MULTICAM360_ASD_DIR={asd_dir()})")
    try:
        import torch, cv2  # noqa
    except ImportError as e:
        probs.append(f"파이썬 패키지 없음: {e.name}")
    load = os.getloadavg()[0]; ncpu = os.cpu_count()
    if load > ncpu * 0.7:
        probs.append(f"부하 높음 {load:.0f}/{ncpu} — --host <원격 호스트> 권장")
    return {"ok": not probs, "host": "local", "load": round(load, 1), "ncpu": ncpu, "problems": probs}
