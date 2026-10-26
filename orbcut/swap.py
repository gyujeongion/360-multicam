"""프리미어에서 4K 파노라마 + 'VR 투영'으로 잡은 각도를 원본 고화질로 바꿔 끼우기.

흐름(맥북은 가볍게, 무거운 렌더는 원격):
  1. dump  — 프리미어에서 실행할 ExtendScript: 시퀀스의 panorama.mp4 클립마다 위치·길이·VR 투영 팬/기울기/롤 키·모션 비율.
  2. plan  — 덤프(JSON) → 클립별 렌더 계획: 원본(source.mkv)·IMU·보정값, 클립 구간, 시각별 각도, 화각.
  3. render — (원격) 원본에서 클립마다 프리미어와 같은 방향·화각의 직선 원근 영상을 렌더(DynamicProjector).
  4. place — 프리미어에서 실행할 ExtendScript: 렌더 결과를 맨 위 새 비디오 트랙의 같은 자리에 올리고 파노라마 클립은 끈다.

좌표: 프리미어 VR 투영 회전 Rz(롤)·Rx(기울기)·Ry(팬)(vrproj, 실측)이 곧 파노라마 좌표의 뷰→세계 회전이고, 파노라마는
track 모드의 안정화 좌표라 render_dynamic의 rot_fn(IMU)과 그대로 맞물린다. 화각 = 36000 / 모션 폭 비율(%).
결과는 직선 원근이라 프리미어 미리보기(등장방형 자르기)보다 직선이 곧다 — 중심 방향·가로 화각은 같다.
"""
import argparse, json, os, sys

import numpy as np

from .vrproj import MOTION_SCALE_H, MOTION_SCALE_W, MOTION_UNIFORM, PROP_PAN, PROP_ROLL, PROP_TILT, VR_EFFECT, premiere_matrix

BIN = "360 원본 렌더"


def dump_script(sequence=None):
    """활성(또는 이름 지정) 시퀀스의 panorama*.mp4 클립 덤프 ExtendScript. 결과는 toSource() 문자열(JSON 호환)."""
    sel = f'var want = {json.dumps(sequence)};' if sequence else 'var want = null;'
    return sel + f'''
var p = app.project, seq = p.activeSequence;
if (want) {{ for (var i = 0; i < p.sequences.numSequences; i++) if (p.sequences[i].name == want) seq = p.sequences[i]; }}
if (!seq || (want && seq.name != want)) return "NO_SEQUENCE";
function ser(v) {{               // ExtendScript엔 JSON이 없다 — toSource는 JSON이 아니라 직접 직렬화
  if (v === null || v === undefined) return "null";
  if (typeof v == "number") return isFinite(v) ? String(v) : "null";
  if (typeof v == "boolean") return v ? "true" : "false";
  if (typeof v == "string") return '"' + v.replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"') + '"';
  if (v instanceof Array) {{ var a = []; for (var i = 0; i < v.length; i++) a.push(ser(v[i])); return "[" + a.join(",") + "]"; }}
  var o = []; for (var k in v) o.push(ser(String(k)) + ":" + ser(v[k])); return "{{" + o.join(",") + "}}";
}}
function keysOf(pr) {{
  if (!pr.isTimeVarying()) return [[null, pr.getValue()]];
  var ks = pr.getKeys(), out = [];
  for (var i = 0; i < ks.length; i++) out.push([ks[i].seconds, pr.getValueAtKey(ks[i])]);
  return out;
}}
var res = {{sequence: seq.name, id: seq.sequenceID, clips: []}};
for (var t = 0; t < seq.videoTracks.numTracks; t++) {{
  var tr = seq.videoTracks[t];
  for (var c = 0; c < tr.clips.numItems; c++) {{
    var cl = tr.clips[c], pi = cl.projectItem;
    if (!pi) continue;
    var media = pi.getMediaPath();
    if (!media || media.indexOf("panorama") < 0 || media.toLowerCase().indexOf(".mp4") < 0) continue;
    var vr = null, mo = null;
    for (var k = 0; k < cl.components.numItems; k++) {{
      var cp = cl.components[k];
      if (cp.displayName == {json.dumps(VR_EFFECT)}) vr = cp;
      if (cp.matchName == "AE.ADBE Motion") mo = cp;
    }}
    if (!vr) continue;
    res.clips.push({{track: t, index: c, name: cl.name, media: media, start: cl.start.seconds, end: cl.end.seconds,
      in_s: cl.inPoint.seconds, out_s: cl.outPoint.seconds, ticks_start: String(cl.start.ticks), ticks_end: String(cl.end.ticks),
      disabled: cl.disabled ? true : false,
      pan: keysOf(vr.properties[{PROP_PAN}]), tilt: keysOf(vr.properties[{PROP_TILT}]), roll: keysOf(vr.properties[{PROP_ROLL}]),
      scale_h: mo.properties[{MOTION_SCALE_H}].getValue(), scale_w: mo.properties[{MOTION_SCALE_W}].getValue(),
      uniform: mo.properties[{MOTION_UNIFORM}].getValue(), position: mo.properties[0].getValue()}});
  }}
}}
return ser(res);
'''


def parse_dump(text):
    """premiere_run_extendscript 결과 → dict (스크립트가 JSON 문자열을 돌려주고, MCP가 그걸 다시 문자열로 감쌀 수 있다)."""
    v = json.loads(text) if isinstance(text, str) else text
    return json.loads(v) if isinstance(v, str) else v


def _keyfn(keys):
    """[[t|None, v], ...] → t(매체 시각) → v. 키 사이 선형(프리미어 효과 기본 보간)."""
    if keys[0][0] is None:
        v = float(keys[0][1])
        return lambda t: v
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([float(k[1]) for k in keys], float)
    return lambda t: float(np.interp(t, ts, vs))


def plan(dump, seq_w=1920, seq_h=1080):
    """덤프 → 클립별 렌더 계획. 원본 위치는 파노라마 옆 manifest.json의 src(원격 경로)."""
    out = []
    for c in dump["clips"]:
        mdir = os.path.dirname(c["media"])
        man = json.load(open(os.path.join(mdir, "manifest.json")))
        src = man["src"]
        rdir = os.path.join(os.path.dirname(src), "out")
        sw = float(c["scale_h"] if c["uniform"] else c["scale_w"])
        hfov = 36000.0 / sw
        out.append({"name": c["name"], "track": c["track"], "index": c["index"], "start": c["start"], "end": c["end"],
                    "ticks_start": c["ticks_start"], "media": c["media"], "src": src, "meta": src + ".trailer",
                    "imu": os.path.join(rdir, "imu.npz"), "in_s": c["in_s"], "dur": round(c["end"] - c["start"], 4),
                    "hfov": round(hfov, 3), "w": seq_w, "h": seq_h, "pan": c["pan"], "tilt": c["tilt"], "roll": c["roll"],
                    "position": c["position"], "key": f"t{c['track']}_c{c['index']}_{c['start']:.2f}"})
    return out


def view_fn(item):
    """매체 시각 t → 뷰→세계 회전(파노라마 좌표)."""
    P, T, R = _keyfn(item["pan"]), _keyfn(item["tilt"]), _keyfn(item["roll"])
    return lambda t: premiere_matrix(P(t), T(t), R(t))


def render(plan_items, out_dir, bitrate="40M"):
    """(원격) 계획대로 원본에서 렌더. 클립마다 <key>.mp4."""
    from .engine import DynamicProjector, flat_dirs, load_lenses, render_dynamic
    from .track import rot_lookup
    os.makedirs(out_dir, exist_ok=True)
    done = []
    for it in plan_items:
        path = os.path.join(out_dir, it["key"] + ".mp4")
        pj = DynamicProjector(load_lenses(it["meta"]), flat_dirs(0, 0, it["hfov"], it["w"], it["h"]))
        rot, V, s0 = rot_lookup(it["imu"]), view_fn(it), it["in_s"]
        n = render_dynamic(it["src"], [(pj, path, bitrate, lambda t: V(s0 + t))], lambda t: rot(s0 + t),
                           start=s0, dur=it["dur"])
        done.append({**{k: it[k] for k in ("key", "track", "index", "start", "end", "ticks_start", "name")}, "file": path, "frames": n})
        print(json.dumps(done[-1], ensure_ascii=False), flush=True)
    return done


def place_script(results, sequence):
    """렌더 결과를 맨 위 새 비디오 트랙의 같은 자리에 올리고 원래 파노라마 클립은 끄는 ExtendScript. 소리는 넣지 않는다."""
    items = [{"file": r["file"], "ticks": str(r["ticks_start"]), "track": r["track"], "index": r["index"], "name": r["name"]}
             for r in results]
    return f'''
var p = app.project, seq = null, want = {json.dumps(sequence)};
for (var i = 0; i < p.sequences.numSequences; i++) if (p.sequences[i].name == want) seq = p.sequences[i];
if (!seq) return "NO_SEQUENCE";
var items = {json.dumps(items, ensure_ascii=False)};
var root = p.rootItem, bin = null;
for (var i = 0; i < root.children.numItems; i++) if (root.children[i].name == {json.dumps(BIN)}) bin = root.children[i];
if (!bin) bin = root.createBin({json.dumps(BIN)});
var files = []; for (var i = 0; i < items.length; i++) files.push(items[i].file);
p.importFiles(files, true, bin, false);
function find(path) {{ for (var i = 0; i < bin.children.numItems; i++) if (bin.children[i].getMediaPath() == path) return bin.children[i]; return null; }}
// 맨 위에 빈 비디오 트랙 하나(이미 이 이름의 클립만 있는 맨 위 트랙이면 재사용)
var top = seq.videoTracks[seq.videoTracks.numTracks - 1], reuse = top.clips.numItems == 0;
if (!reuse) {{ var allOurs = true; for (var c = 0; c < top.clips.numItems; c++) if (top.clips[c].projectItem && top.clips[c].projectItem.getMediaPath().indexOf("/swap/") < 0) allOurs = false; reuse = allOurs; }}
if (!reuse) {{ app.enableQE(); var qs = null; for (var i = 0; i < qe.project.numSequences; i++) if (qe.project.getSequenceAt(i).name == want) qs = qe.project.getSequenceAt(i);
  qs.addTracks(1, seq.videoTracks.numTracks, 0); top = seq.videoTracks[seq.videoTracks.numTracks - 1]; }}
var placed = 0, missing = [];
for (var i = 0; i < items.length; i++) {{
  var it = find(items[i].file); if (!it) {{ missing.push(items[i].file); continue; }}
  var sub = it.createSubClip(items[i].name + " 원본", "0", it.getOutPoint().ticks, 1, 1, 0);
  top.overwriteClip(sub || it, items[i].ticks);
  var src = seq.videoTracks[items[i].track].clips[items[i].index];
  try {{ src.disabled = true; }} catch (e) {{}}
  placed++;
}}
return "placed=" + placed + " track=V" + seq.videoTracks.numTracks + (missing.length ? " missing=" + missing.join("|") : "");
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description="파노라마 VR 투영 각도 → 원본 고화질로 바꿔 끼우기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump-script"); a.add_argument("--sequence")
    b = sub.add_parser("plan"); b.add_argument("dump"); b.add_argument("out")
    c = sub.add_parser("render"); c.add_argument("plan"); c.add_argument("out_dir"); c.add_argument("--bitrate", default="40M")
    d = sub.add_parser("place-script"); d.add_argument("results"); d.add_argument("sequence")
    a_ = ap.parse_args(argv)
    if a_.cmd == "dump-script":
        print(dump_script(a_.sequence))
    elif a_.cmd == "plan":
        dump = parse_dump(open(a_.dump).read())
        json.dump(plan(dump), open(a_.out, "w"), ensure_ascii=False, indent=1)
        print(a_.out)
    elif a_.cmd == "render":
        render(json.load(open(a_.plan)), a_.out_dir, a_.bitrate)
    elif a_.cmd == "place-script":
        print(place_script(json.load(open(a_.results)), a_.sequence))
    return 0


if __name__ == "__main__":
    sys.exit(main())
