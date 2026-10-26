"""엔진 방향(yaw·pitch·hfov) → 프리미어 내장 'VR 투영' 효과 팬·기울기 + 모션 비율.

4K 등장방형 파노라마 클립에 VR 투영을 걸면 구가 돌아갈 뿐 화면은 360×180 전체가 시퀀스 프레임에 늘어난다
(출력 FOV 속성은 자르지 않는다 — 실측). 그래서 리프레임 = VR 투영으로 대상을 가운데로 돌리고 모션 비율로 확대.
결과는 등장방형을 잘라 낸 것이라 직선이 휜다 — 자동 구도의 화질은 원격에서 렌더한 인물 앵글이 맡고, 이 경로는 손으로 고칠 때용.

실측(2026-10-24, 프리미어 26.5, 3840×1920 클립·1920×1080 시퀀스, 출력 프레임을 원본에서 재현해 피팅, NCC ≥0.998):
  회전 = Rz(롤)·Rx(기울기)·Ry(팬) (원본 방향 = R · 출력 방향, 세 값 모두 + 부호). 기울기는 고정 X축 기준이라
  옆을 볼 때 그대로 넣으면 롤처럼 돈다 → 엔진 시선 Ry(yaw)·Rx(pitch)를 이 순서의 오일러각으로 역산해 롤로 보정.
  'NewRotationOrder' 속성은 출력에 영향 없음(프레임 바이트 동일). 팬은 ±180에서 잘린다(200 → 180).
  모션 폭 비율 = 36000/hfov %, 높이 비율 = 18000/vfov %, vfov = hfov × 시퀀스 높이/폭 (정사각 픽셀). 화각 오차 ≤0.1°.
"""
import json, math

import numpy as np

VR_EFFECT = "VR 투영"   # 한국어 UI 효과 이름(QE getVideoEffectByName)
PROP_PAN, PROP_TILT, PROP_ROLL = 13, 14, 15     # VR 투영 컴포넌트 속성 번호(26.5 실측)
MOTION_SCALE_H, MOTION_SCALE_W, MOTION_UNIFORM = 1, 2, 3


def motion_scale(hfov, seq_w=1920, seq_h=1080):
    """(높이 비율 %, 폭 비율 %) — 등장방형 hfov°만큼을 시퀀스 폭에, 같은 각 해상도로 세로를 채운다."""
    vfov = hfov * seq_h / seq_w
    return 18000.0 / vfov, 36000.0 / hfov


def _rx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def premiere_matrix(pan, tilt, roll):
    """VR 투영 값 → 회전 행렬 (원본 방향 = R · 출력 방향)."""
    return _rz(math.radians(roll)) @ _rx(math.radians(tilt)) @ _ry(math.radians(pan))


def engine_matrix(yaw, pitch):
    """엔진 시선(engine.flat_dirs와 같은 규약: pitch 위 +, 그 뒤 yaw 오른쪽 +, 롤 없음)."""
    return _ry(math.radians(yaw)) @ _rx(math.radians(pitch))


def to_vr(yaw, pitch):
    """엔진 방향 → (팬, 기울기, 롤): Rz(r)·Rx(t)·Ry(p) = Ry(yaw)·Rx(pitch) 를 푼다."""
    M = engine_matrix(yaw, pitch)
    t = math.asin(max(-1.0, min(1.0, M[2, 1])))
    p = math.atan2(-M[2, 0], M[2, 2])
    r = math.atan2(-M[0, 1], M[1, 1])
    return math.degrees(p), math.degrees(t), math.degrees(r)


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def path_keys(rows, step=0.5, smooth_s=0.6, t0=0.0, t1=None):
    """paths.json 한 사람의 [(t, yaw, pitch, found, present), ...] → step초 간격 (t, 팬, 기울기, 롤).
    못 찾은 구간은 앞뒤로 보간, yaw는 언랩 후 가우시안 평활. 팬이 ±180을 넘어가는 순간은
    프리미어가 반대로 돌며 보간하므로 경계 직전·직후(1프레임 간격) 키를 넣어 한 프레임 안에 넘긴다."""
    r = [row for row in rows if len(row) < 4 or row[3]] or rows
    t = np.array([x[0] for x in r]); yaw = np.unwrap(np.radians([x[1] for x in r])); pit = np.array([x[2] for x in r])
    t1 = t[-1] if t1 is None else t1
    grid = np.arange(t0, t1 + 1e-9, 1 / 25.0)
    y, p = np.interp(grid, t, yaw), np.interp(grid, t, pit)
    k = max(1, int(smooth_s * 25))
    w = np.exp(-0.5 * (np.arange(-2 * k, 2 * k + 1) / k) ** 2); w /= w.sum()
    pad = lambda a: np.convolve(np.pad(a, 2 * k, mode="edge"), w, "valid")
    y, p = np.degrees(pad(y)), pad(p)
    keys, prev = [], None
    for tk in np.arange(t0, t1 + 1e-9, step):
        i = min(int(round((tk - t0) * 25)), len(grid) - 1)
        pan, tilt, roll = to_vr(_wrap(y[i]), p[i])
        if prev is not None and abs(pan - prev[1]) > 180:          # ±180 경계를 넘음 → 1프레임 전환 키 두 개
            tc = round(float(tk) - 1 / 29.97, 3)
            edge = 180.0 if prev[1] > 0 else -180.0
            keys.append((tc, edge, prev[2], prev[3]))
            keys.append((round(tc + 1 / 29.97 * 0.5, 3), -edge, tilt, roll))
        prev = (round(float(tk), 3), round(pan, 2), round(tilt, 2), round(roll, 2))
        keys.append(prev)
    return keys


def extendscript(clip_track, clip_index, keys, hfov, seq_w=1920, seq_h=1080, seq_name=None):
    """활성(또는 이름 지정) 시퀀스의 V{clip_track} 트랙 clip_index번 클립에 VR 투영을 걸고 키를 쓰는 ExtendScript.
    클립이 이미 VR 투영을 갖고 있으면 그 효과의 키를 갈아 쓴다."""
    sh, sw = motion_scale(hfov, seq_w, seq_h)
    return f"""
var seq = app.project.activeSequence;
{'if (seq.name != ' + json.dumps(seq_name) + ') return "wrong sequence " + seq.name;' if seq_name else ''}
var cl = seq.videoTracks[{clip_track}].clips[{clip_index}];
function findVR() {{ for (var k = 0; k < cl.components.numItems; k++) if (cl.components[k].displayName == {json.dumps(VR_EFFECT)}) return cl.components[k]; return null; }}
var vr = findVR();
if (!vr) {{ app.enableQE(); var qi = qe.project.getActiveSequence().getVideoTrackAt({clip_track}).getItemAt({clip_index});
  qi.addVideoEffect(qe.project.getVideoEffectByName({json.dumps(VR_EFFECT)})); vr = findVR(); }}
if (!vr) return "VR 효과 적용 실패";
var mo = cl.components[1];
mo.properties[{MOTION_UNIFORM}].setValue(false, true);
mo.properties[{MOTION_SCALE_H}].setValue({sh:.3f}, true); mo.properties[{MOTION_SCALE_W}].setValue({sw:.3f}, true);
var keys = {json.dumps(keys)};
var props = [vr.properties[{PROP_PAN}], vr.properties[{PROP_TILT}], vr.properties[{PROP_ROLL}]];
for (var p = 0; p < 3; p++) {{ props[p].setTimeVarying(false); props[p].setTimeVarying(true); }}
var base = cl.inPoint.seconds;
for (var i = 0; i < keys.length; i++) {{ var tm = new Time(); tm.seconds = base + keys[i][0];
  for (var p = 0; p < 3; p++) {{ props[p].addKey(tm); props[p].setValueAtKey(tm, keys[i][p + 1], i == keys.length - 1); }} }}
return "keys=" + props[0].getKeys().length + "," + props[1].getKeys().length + "," + props[2].getKeys().length;
"""
