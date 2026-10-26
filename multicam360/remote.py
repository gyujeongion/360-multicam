"""원격 호스트(작업용 맥 등)에서 multicam360 실행: 코드 동기화 → 환경 준비 → 실행 → 결과 회수.

원격 경로: ~/Development/261020_360multicam (같은 상대 구조). 대용량 산출물은 원격에 두고
결과(manifest·xml·shots·preview·logs)와 4K 마스터를 회수하고 XML 경로를 로컬로 바꾼다. --no-fetch-masters면 마스터 제외.
"""
import json, os, shlex, subprocess, sys

from .config import REPO, nas_ffmpeg, nas_host

from .config import remote_dir

REMOTE_DIR = remote_dir()
DEPS_FILE = "requirements.txt"   # 원격도 같은 목록 사용


def sh(host, cmd, check=True, capture=False):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], text=True,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    if check and r.returncode != 0:
        raise RuntimeError(f"원격 명령 실패({r.returncode}): {cmd}\n{r.stdout or ''}")
    return r.stdout if capture else r.returncode


def sync(host):
    """코드·벤더(Light-ASD)·보정 트레일러만 동기화. work/cache/영상은 제외."""
    sh(host, f"mkdir -p {REMOTE_DIR}")
    subprocess.run(["rsync", "-az", "--delete", "--exclude", "work/", "--exclude", "cache/", "--exclude", ".venv/",
                    "--exclude", "__pycache__/", "--exclude", ".goalset/", "--exclude", "*.mp4", "--exclude", "*.mkv",
                    # 역분석·테스트 픽스처(수백 MB~GB) 제외 — 원격 스모크용 트레일러(re/tail8m.bin)만 보냄
                    "--include", "re/tail8m.bin", "--exclude", "re/*.bin", "--exclude", "re/*.trailer",
                    "--exclude", "re/*/", "--exclude", "re/*.png", "--exclude", "re/*.npy", "--exclude", "re/*.npz",
                    REPO + "/", f"{host}:{REMOTE_DIR}/"], check=True)


def setup(host):
    sh(host, f"cd {REMOTE_DIR} && (test -x .venv/bin/python || uv venv -q -p 3.11 .venv) && "
             f"uv pip install -q -p .venv -r {DEPS_FILE}")


def doctor(host):
    """원격 준비 상태 점검. 문제 목록이 비면 OK."""
    probs, info = [], {}
    try:
        out = sh(host, "uptime; sysctl -n hw.ncpu; df -k ~ | tail -1; which ffmpeg ffprobe uv; "
                       f"cd {REMOTE_DIR} 2>/dev/null && .venv/bin/python -c 'import torch,cv2,scenedetect,google.protobuf;print(\"py_ok\")'; "
                       "test -f Light-ASD/weight/pretrain_AVA_CVPR.model && echo asd_ok; "
                       "test -f Light-ASD/model/faceDetector/s3fd/sfd_face.pth && echo s3fd_ok",
                 check=False, capture=True)
    except FileNotFoundError:
        return {"ok": False, "problems": ["ssh 없음"]}
    lines = out.strip().splitlines()
    if not lines or "load averages" not in lines[0]:
        return {"ok": False, "problems": [f"ssh {host} 접속 실패: {out.strip()[:200]}"]}
    load1 = float(lines[0].split("load averages:")[1].split()[0])
    ncpu = int(lines[1])
    free_gb = int(lines[2].split()[3]) / 1024 / 1024
    info.update(load=load1, ncpu=ncpu, free_gb=round(free_gb))
    for tool in ("ffmpeg", "ffprobe", "uv"):
        if not any(l.endswith("/" + tool) for l in lines):
            probs.append(f"{tool} 없음")
    if "py_ok" not in out: probs.append("venv/패키지 미준비 → multicam360 remote setup")
    if "asd_ok" not in out: probs.append("Light-ASD 가중치 없음 → sync")
    if "s3fd_ok" not in out: probs.append("S3FD 가중치 없음 → sync")
    if load1 > ncpu * 0.7: probs.append(f"부하 높음 {load1}/{ncpu} — 다른 무거운 작업 대기 권장")
    if free_gb < 100: probs.append(f"디스크 부족 {free_gb:.0f}GB")
    return {"ok": not probs, "host": host, **info, "problems": probs}


def fetch_source(host, nas_path, start, dur, dest_rel):
    """NAS 원본에서 구간만 무손실로 원격 호스트에 직접 복사(맥북 경유 없음). 트레일러 전체도 함께."""
    q = shlex.quote(nas_path)
    sh(host, f"mkdir -p {REMOTE_DIR}/$(dirname {dest_rel}) && "
             f"ssh {nas_host()} \"{nas_ffmpeg()} -v error -ss {start} -i {q} -t {dur} -map 0:0 -map 0:1 -map 0:a:0 "
             f"-c copy -f matroska -\" > {REMOTE_DIR}/{dest_rel} && "
             # 트레일러 전체(보정값 + 자이로) — 길이는 끝 72B 푸터의 uint32
             f"ssh {nas_host()} \"tail -c \$(tail -c 72 {q} | od -An -tu4 -j32 -N4 | tr -d ' ') {q}\" "
             f"> {REMOTE_DIR}/{dest_rel}.trailer")


def relocate(out, remote_base, local_base):
    """회수한 XML·샷 목록·manifest 안의 원격 절대경로를 로컬 경로로 (프리미어에서 미디어 오프라인 방지)."""
    import urllib.parse
    pairs = [(remote_base, local_base), (urllib.parse.quote(remote_base), urllib.parse.quote(local_base))]
    for fn in ("premiere.xml", "premiere_shots.json", "manifest.json", "edit.xml"):
        p = os.path.join(out, fn)
        if os.path.exists(p):
            s = open(p, encoding="utf-8").read()
            for a, b in pairs:
                s = s.replace(a, b)
            open(p, "w", encoding="utf-8").write(s)


def run(host, mc_args, out_rel, fetch_to=None, fetch_masters=True):
    """원격에서 `python -m multicam360 --json <mc_args>` 실행 후 결과 회수."""
    rc = sh(host, f"cd {REMOTE_DIR} && .venv/bin/python -m multicam360 --json {' '.join(shlex.quote(a) for a in mc_args)}",
            check=False)
    if fetch_to:
        os.makedirs(fetch_to, exist_ok=True)
        inc = ["--include", "*/", "--include", "manifest.json", "--include", "*.xml", "--include", "*shots.json",
               "--include", "edit.json", "--include", "people.json", "--include", "preview.mp4", "--include", "logs/*"]
        if fetch_masters:
            inc += ["--include", "masters/*.mp4", "--include", "masters/render.json"]
        subprocess.run(["rsync", "-az", *inc, "--exclude", "*", f"{host}:{REMOTE_DIR}/{out_rel}/", fetch_to + "/"],
                       check=True)
    return rc


def fetch_nas_local(nas_path, start, dur, dest):
    """NAS 원본 → 로컬 파일 (구간 무손실 + 트레일러 전체 → dest.trailer)."""
    q = shlex.quote(nas_path)
    seek = f"-ss {start} " if start is not None else ""
    tdur = f"-t {dur} " if dur is not None else ""
    with open(dest, "wb") as f:
        subprocess.run(["ssh", nas_host(), f"{nas_ffmpeg()} -v error {seek}-i {q} {tdur}-map 0:0 -map 0:1 -map 0:a:0 "
                        f"-c copy -f matroska -"], stdout=f, check=True)
    from .imu import read_trailer
    with open(dest + ".trailer", "wb") as f:
        f.write(read_trailer("nas:" + nas_path))     # 트레일러 전체(보정값 + 자이로)


def run_all_remote(host, src, out, rng=None, meta=None, eq_width=3840, fetch_masters=True, mode="table",
                   sets=None, force=None, diar=None):
    """원격 전체 실행. src: nas:/... 이면 NAS→원격 직접 복사, 로컬 경로면 rsync로 올림.
    sets/force는 그대로 넘기고, diar(화자 분리 JSON)는 원격 작업 폴더로 올려 cut.diar를 그 경로로 바꾼다."""
    from .pipeline import parse_range
    doc = doctor(host)
    if any("venv" in p or "가중치" in p for p in doc["problems"]):
        sync(host); setup(host); doc = doctor(host)
    else:
        sync(host)
    blocking = [p for p in doc["problems"] if "부하" not in p]
    if blocking:
        return {"status": "failed", "stage": "doctor", "error": blocking}
    name = os.path.basename(os.path.abspath(out))
    rsrc = f"work/{name}/source.mkv"
    sh(host, f"mkdir -p {REMOTE_DIR}/work/{name}")
    if src.startswith("nas:"):
        start, dur = parse_range(rng)
        fetch_source(host, src[4:], start if start is not None else 0, dur if dur is not None else 999999, rsrc)
        rmeta, rrange = rsrc + ".trailer", None
        src_start = start if start is not None else 0
    else:
        subprocess.run(["rsync", "-a", src, f"{host}:{REMOTE_DIR}/{rsrc}"], check=True)
        rmeta = None
        if meta:
            subprocess.run(["rsync", "-a", meta, f"{host}:{REMOTE_DIR}/{rsrc}.trailer"], check=True)
            rmeta = rsrc + ".trailer"
        rrange = rng
        src_start = 0
    args = ["all", rsrc, "--src-start", str(src_start), "--out", f"work/{name}/out", "--eq-width", str(eq_width), "--mode", mode]
    if rmeta: args += ["--meta", rmeta]
    if rrange: args += ["--range", rrange]
    sets = list(sets or [])
    if diar:
        rdiar = f"work/{name}/diar.json"
        subprocess.run(["rsync", "-a", diar, f"{host}:{REMOTE_DIR}/{rdiar}"], check=True)
        sets = [x for x in sets if not x.startswith("cut.diar=")] + [f"cut.diar={rdiar}"]
    for x in sets:
        args += ["--set", x]
    for x in (force or []):
        args += ["--force", x]
    run(host, args, f"work/{name}/out", fetch_to=out, fetch_masters=fetch_masters)
    m = json.load(open(os.path.join(out, "manifest.json")))
    outs = m.get("outputs") or {}
    rbase = os.path.dirname(outs["premiere_xml"]) if "premiere_xml" in outs else None
    if rbase:
        relocate(out, rbase, os.path.abspath(out))
    return {"status": m.get("status"), "error": m.get("error"), "host": host,
            "remote_dir": f"{host}:~/{REMOTE_DIR}/work/{name}/out", "local_dir": os.path.abspath(out),
            "outputs": {k: os.path.join(os.path.abspath(out), os.path.relpath(v, rbase)) for k, v in outs.items()}}
