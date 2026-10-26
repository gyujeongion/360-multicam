"""orbcut — Insta360 360 영상 → 화자 기반 멀티캠 컷 + 프리미어 XML.

명령:
  all <src> --out DIR [--range 4-00:7-00] [--meta TRAILER] [--host HOST]
                                     전체 파이프라인 (manifest.json 갱신). src = 로컬 경로 또는 nas:/<NAS 절대경로>
  probe <insv|trailer|nas:/...> [out.json]
                                     기종·보정값 버전·길이 확인. exit 0=지원 / 3=미지원 형식 / 4=읽기 실패
  status DIR                         manifest.json 요약 (JSON)
  doctor [--host HOST]               실행 환경 점검 (원격이면 ssh·ffmpeg·venv·가중치·부하·디스크)
  remote-setup --host HOST           원격에 코드 동기화 + 파이썬 환경 설치
  eval EDIT.json LABELS.json         화자 컷 정확도 (사람 정답 대비, labels는 라벨링 도구 출력)
  diarize DIR DIAR.json [--out F]    화자 분리(소니옥스 JSON) + 화면 증거로 화자→인물 배정·화자 컷 (DIR = all 결과)
  vrkeys DIR PERSON [--hfov 70]      track 결과의 인물 경로 → 프리미어 'VR 투영' 팬·기울기·롤 키프레임 ExtendScript
                                     (4K panorama.mp4 클립에 걸어 손으로 고칠 출발점. 출력 = .jsx 경로)

출력은 기본 사람용, --json이면 마지막 줄 한 줄 JSON (AI 판독용).
"""
import argparse, json, os, sys

from . import pipeline


def parse_sets(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        stage, key = k.split(".", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        out.setdefault(stage, {})[key] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orbcut", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="결과를 한 줄 JSON으로 (AI용)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("all", help="전체 파이프라인")
    a.add_argument("src", help="insv, 구간 mkv, 또는 NAS 경로")
    a.add_argument("--out", required=True)
    a.add_argument("--range", help="구간 시작:끝, 분-초 표기 (예: 4-00:7-00) 또는 초 (240:420)")
    a.add_argument("--meta", help="보정값 트레일러가 든 파일(src가 잘린 구간일 때 원본 insv 또는 트레일러 끝부분)")
    a.add_argument("--eq-width", type=int, default=3840, help="파노라마(인물 검출·라벨링용) 폭")
    a.add_argument("--jobs", type=int, help="렌더 병렬 프로세스 수 (기본: 코어/4, 최대 5)")
    a.add_argument("--set", action="append", default=[], metavar="단계.키=값",
                   help="파라미터 덮어쓰기, 예: --set xml.face_ratio=0.28 --set cut.min_shot=2 (바뀐 단계부터만 재실행)")
    a.add_argument("--force", action="append", default=[], metavar="단계", help="이 단계 강제 재실행")
    a.add_argument("--until", metavar="단계", help="이 단계까지만 실행 (튜닝 반복용)")
    a.add_argument("--mode", choices=["table", "track"], default="table",
                   help="table = 카메라 고정(테이블 가운데) / track = 손에 든·차 안: 자이로 안정화 + 인물별 얼굴 추적 앵글")
    a.add_argument("--src-start", type=float, default=0.0, help="src 0초가 원본의 몇 초인지 (이미 잘린 구간 파일일 때, track 자이로 동기용)")
    a.add_argument("--host", help="원격 실행 호스트 (ssh 호스트 이름). 결과는 --out으로 회수")
    a.add_argument("--diar", help="화자 분리 JSON(소니옥스 토큰) — 주면 컷을 소리 경계 + 화면 증거 배정(diarize)으로")
    a.add_argument("--diar-offset", type=float, default=0.0, help="화자 분리 시각을 이 초만큼 뒤로(구간 렌더 키프레임 리드)")
    a.add_argument("--diar-map", default="", help='화자→인물 배정 덮어쓰기, 예: "1=D,2=H"')
    a.add_argument("--no-fetch-masters", action="store_true",
                   help="원격 실행 시 4K 마스터를 회수하지 않음 (수 GB 절약, 대신 XML 미디어가 오프라인)")

    d = sub.add_parser("doctor", help="실행 환경 점검")
    d.add_argument("--host")

    r = sub.add_parser("remote-setup", help="원격 코드 동기화 + 환경 설치")
    r.add_argument("--host", required=True)

    p = sub.add_parser("probe", help="기종·보정값 확인")
    p.add_argument("src"); p.add_argument("out", nargs="?")

    e = sub.add_parser("eval", help="화자 컷 정확도 평가")
    e.add_argument("edit"); e.add_argument("labels")
    e.add_argument("--tol", type=float, default=0.75, help="컷 시점 허용 오차(초)")

    dz = sub.add_parser("diarize", help="화자 분리 JSON + 화면 증거로 화자→인물 배정")
    dz.add_argument("dir"); dz.add_argument("diar")
    dz.add_argument("--audio", help="음량 계산용 원본(기본: manifest src)")
    dz.add_argument("--out", help="결과 JSON (기본: DIR/cut_diar.json)")
    dz.add_argument("--map", default="", help='배정 덮어쓰기, 예: "1=D,2=H"')
    dz.add_argument("--min-shot", type=float, default=1.5)
    dz.add_argument("--offset", type=float, default=0.0, help="화자 분리 시각을 이 초만큼 뒤로(구간 렌더 앞 키프레임 리드)")

    vk = sub.add_parser("vrkeys", help="인물 경로 → 프리미어 VR 투영 키프레임 ExtendScript")
    vk.add_argument("dir"); vk.add_argument("person")
    vk.add_argument("--hfov", type=float, default=70); vk.add_argument("--step", type=float, default=0.5)
    vk.add_argument("--track", type=int, default=0, help="비디오 트랙 번호(0=V1)"); vk.add_argument("--clip", type=int, default=0)
    vk.add_argument("--start", type=float, default=0.0, help="클립 첫 프레임이 경로의 몇 초인지")
    vk.add_argument("--seq-w", type=int, default=1920); vk.add_argument("--seq-h", type=int, default=1080)
    vk.add_argument("--sequence", help="이 이름의 시퀀스가 활성일 때만 실행")
    vk.add_argument("--out", help="출력 .jsx (기본: DIR/vrkeys_<인물>.jsx)")

    s = sub.add_parser("status", help="manifest 요약")
    s.add_argument("dir")

    args = ap.parse_args(argv)
    if args.cmd == "all" and args.diar:
        args.set += [f"cut.diar={os.path.abspath(args.diar) if not args.host else args.diar}",
                     f"cut.diar_offset={args.diar_offset}", f"cut.diar_map={json.dumps(args.diar_map)}"]
    if args.cmd == "all" and args.host:
        from . import remote
        res = remote.run_all_remote(args.host, args.src, args.out, rng=args.range, meta=args.meta,
                                    eq_width=args.eq_width, fetch_masters=not args.no_fetch_masters, mode=args.mode,
                                    sets=args.set, force=args.force, diar=args.diar)
        print(json.dumps(res, ensure_ascii=False) if args.json else json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("status") == "ok" else 2
    if args.cmd == "all" and args.src.startswith("nas:"):
        from . import remote
        start, dur = pipeline.parse_range(args.range)
        local = os.path.join(os.path.abspath(args.out), "nas_source.mkv")
        os.makedirs(args.out, exist_ok=True)
        remote.fetch_nas_local(args.src[4:], start, dur, local)
        args.src, args.meta, args.range, args.src_start = local, local + ".trailer", None, start or 0.0
    if args.cmd == "all":
        try:
            res = pipeline.run_all(args.src, args.out, meta=args.meta, rng=args.range, eq_width=args.eq_width,
                                   jobs=args.jobs, overrides=parse_sets(args.set), force=args.force, mode=args.mode,
                                   src_start=args.src_start, until=args.until)
        except pipeline.StageError as e:
            print(json.dumps({"status": "failed", "stage": e.stage, "error": e.msg}, ensure_ascii=False) if args.json
                  else f"실패: {e}", file=sys.stdout)
            return 2
        if res.get("status") == "partial":
            print(json.dumps({"status": "partial", "stopped_after": res.get("stopped_after")}, ensure_ascii=False))
            return 0
        print(json.dumps({"status": "ok", "outputs": res["outputs"]}, ensure_ascii=False) if args.json
              else "완료\n" + "\n".join(f"  {k}: {v}" for k, v in res["outputs"].items()))
        return 0
    if args.cmd == "probe":
        from .probe import ProbeError, probe
        try:
            info = probe(args.src)
        except ProbeError as e:
            print(json.dumps({"supported": False, "error_code": e.code, "error": str(e)}, ensure_ascii=False))
            return 4
        if args.out:
            json.dump(info, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(json.dumps(info, ensure_ascii=False))
        return 0 if info["supported"] else 3
    if args.cmd == "doctor":
        if args.host:
            from . import remote
            res = remote.doctor(args.host)
        else:
            res = pipeline.doctor_local()
        print(json.dumps(res, ensure_ascii=False, indent=None if args.json else 2))
        return 0 if res["ok"] else 1
    if args.cmd == "remote-setup":
        from . import remote
        remote.sync(args.host); remote.setup(args.host)
        res = remote.doctor(args.host)
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res["ok"] else 1
    if args.cmd == "eval":
        from .eval import evaluate
        r = evaluate(json.load(open(args.edit)), json.load(open(args.labels)), args.tol)
        print(json.dumps(r, ensure_ascii=False, indent=None if args.json else 2))
        return 0
    if args.cmd == "diarize":
        from .diarize import diarize
        r = diarize(args.dir, args.diar, audio=args.audio, min_shot=args.min_shot, overrides=args.map, offset=args.offset)
        out = args.out or os.path.join(args.dir, "cut_diar.json")
        json.dump(r, open(out, "w"), ensure_ascii=False, indent=1)
        brief = {k: r[k] for k in ("mapping", "auto_mapping", "margin", "runner_up")}
        brief.update(shots=len(r["segments"]), out=out)
        print(json.dumps(brief, ensure_ascii=False, indent=None if args.json else 2))
        return 0
    if args.cmd == "vrkeys":
        from .vrproj import extendscript, path_keys
        paths = json.load(open(os.path.join(args.dir, "paths.json")))["paths"]
        if args.person not in paths:
            print(json.dumps({"status": "failed", "error": f"인물 {args.person} 없음", "people": sorted(paths)}, ensure_ascii=False))
            return 2
        keys = [(round(t - args.start, 3),) + tuple(k) for t, *k in [list(x) for x in path_keys(paths[args.person], step=args.step)]
                if t >= args.start - 1e-6]
        out = args.out or os.path.join(args.dir, f"vrkeys_{args.person}.jsx")
        open(out, "w").write(extendscript(args.track, args.clip, keys, args.hfov, args.seq_w, args.seq_h, args.sequence))
        print(json.dumps({"status": "ok", "keys": len(keys), "out": out}, ensure_ascii=False))
        return 0
    if args.cmd == "status":
        m = pipeline.Manifest(args.dir).data
        brief = {"status": m.get("status"), "error": m.get("error"),
                 "stages": {k: v.get("status") for k, v in m.get("stages", {}).items()}, "outputs": m.get("outputs")}
        print(json.dumps(brief, ensure_ascii=False, indent=None if args.json else 2))
        return 0 if m.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
