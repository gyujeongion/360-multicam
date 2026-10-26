// orbcut — orbcut 결과(premiere_shots.json)를 프리미어에서 다루는 패널.
// 처리 엔진은 orbcut CLI. 이 패널은 불러오기·샷 이동·Motion(Scale/Position) 적용·미세조정만 한다.
// API: Premiere UXP 25.6+ (require("premierepro")). Motion 위치 = 0~1 정규화 PointF (프리미어 26.5 실기 확인, 2026-10-21).
const ppro = require("premierepro");
const { storage } = require("uxp");
const fs = storage.localFileSystem;

const state = { shots: null, dir: null };
const $ = (s) => document.querySelector(s);
const status = (msg, warn) => { const el = $("#status"); el.textContent = msg; el.classList.toggle("warn", !!warn); };

async function project() { return await ppro.Project.getActiveProject(); }
async function sequence() { return await (await project()).getActiveSequence(); }

async function motionParams(trackItem) {
  // Motion 컴포넌트를 matchName으로 찾고, Position/Scale 파라미터를 표시 이름으로 찾는다 (인덱스 가정 금지)
  const chain = await trackItem.getComponentChain();
  for (let i = 0; i < chain.getComponentCount(); i++) {
    const comp = chain.getComponentAtIndex(i);
    if ((await comp.getMatchName()) !== "AE.ADBE Motion") continue;
    const out = {};
    for (let p = 0; p < comp.getParamCount(); p++) {
      const prm = comp.getParam(p);
      const name = (prm.displayName || "").toLowerCase();
      if (name === "position" || name === "위치") out.position = prm;
      if (name === "scale" || name === "비율 조정" || name === "크기 조정") out.scale = prm;
    }
    return out;
  }
  throw new Error("Motion 컴포넌트 없음");
}

function setParamActions(prm, value) {
  // 시간 변화 없는 값으로 고정 후 설정 (키프레임 X)
  return [prm.createSetTimeVaryingAction(false), prm.createSetValueAction(prm.createKeyframe(value), true)];
}

async function applyMotion(trackItem, scale, horiz, vert) {
  const { position, scale: sc } = await motionParams(trackItem);
  // orbcut center(시퀀스 중앙 0, ±0.5) → Premiere Position(0~1, 왼쪽 위 원점)
  const pt = new ppro.PointF(0.5 + horiz, 0.5 + vert);
  transact(await project(), () => [...setParamActions(position, pt), ...setParamActions(sc, scale)], "orbcut 구도");
}

function transact(proj, makeActions, label) {
  // 액션·키프레임은 잠금 안에서 만들어야 한다 — 밖에서 만든 객체는 "script object is no longer valid" (26.5 실측)
  let ok = false;
  proj.lockedAccess(() => { ok = proj.executeTransaction((ca) => makeActions().forEach((a) => ca.addAction(a)), label); });
  if (!ok) throw new Error("트랜잭션 거부: " + label);
}

async function v1Items() {
  const seq = await sequence();
  const track = await seq.getVideoTrack(0);
  return track.getTrackItems(1 /* Clip */, false);
}

async function applyAll() {
  if (!state.shots) return;
  const items = await v1Items();
  status(`적용 중… V1 클립 ${items.length}개`);
  let done = 0, skipped = 0;
  for (const it of items) {
    const t = (await it.getStartTime()).seconds;
    const shot = state.shots.find((s) => Math.abs(s.start - t) < 0.05);
    if (!shot) { skipped++; continue; }
    status(`적용 중… ${done + 1}/${items.length} (${shot.cam})`);
    await applyMotion(it, shot.scale, shot.horiz, shot.vert);
    done++;
  }
  status(`자동 구도 적용 ${done}개` + (skipped ? ` · 매칭 안 된 클립 ${skipped}개` : ""), skipped > 0);
}

async function nudge(kind, a, b) {
  const sel = await (await sequence()).getSelection();
  const items = await sel.getTrackItems();
  if (!items.length) { status("타임라인에서 클립을 먼저 선택하세요", true); return; }
  for (const it of items) {
    const { position, scale } = await motionParams(it);
    const zero = ppro.TickTime.createWithSeconds(0);
    if (kind === "scale") {
      const cur = await scale.getValueAtTime(zero);
      transact(await project(), () => setParamActions(scale, cur + a), "orbcut 확대");
    } else {
      const cur = await position.getValueAtTime(zero);
      const pt = new ppro.PointF(cur.x + a, cur.y + b);
      transact(await project(), () => setParamActions(position, pt), "orbcut 위치");
    }
  }
  status(`선택 클립 ${items.length}개 조정`);
}

function renderShots() {
  const ul = $("#shots"); ul.innerHTML = "";
  state.shots.forEach((s) => {
    const li = document.createElement("li");
    const t = document.createElement("span"); t.className = "t";
    t.textContent = `${s.start.toFixed(1)}–${s.end.toFixed(1)}s`;
    const c = document.createElement("span"); c.textContent = `cam_${s.cam} · ${s.scale.toFixed(0)}%`;
    li.append(t, c);
    li.addEventListener("click", async () => {
      await (await sequence()).setPlayerPosition(ppro.TickTime.createWithSeconds(s.start + 0.01));
    });
    ul.appendChild(li);
  });
}

function loaded(shots, where) {
  state.shots = shots;
  renderShots();
  $("#applyAll").disabled = false;
  status(`샷 ${shots.length}개 불러옴 (${where}) — '자동 구도 전체 적용'을 누르세요`);
}

async function autoShotsPath() {
  // 활성 시퀀스 V1 첫 클립의 미디어 = <out>/masters/cam_X.mp4 → <out>/premiere_shots.json
  const items = await v1Items();
  if (!items.length) return null;
  const clip = ppro.ClipProjectItem.cast(await items[0].getProjectItem());
  const media = await clip.getMediaFilePath();
  const masters = media.substring(0, media.lastIndexOf("/"));
  return masters.substring(0, masters.lastIndexOf("/")) + "/premiere_shots.json";
}

$("#open").addEventListener("click", async () => {
  // 1순위: 시퀀스 미디어 위치에서 자동으로 찾는다(파일 고를 필요 없음). 실패하면 파일 선택.
  try {
    const p = await autoShotsPath();
    if (p) {
      const entry = await fs.getEntryWithUrl("file:" + p);
      loaded(JSON.parse(await entry.read()), "자동");
      return;
    }
  } catch (e) { /* 다른 폴더 구조 → 파일 선택으로 */ }
  try {
    const f = await fs.getFileForOpening({ types: ["json"] });
    if (!f) return;
    loaded(JSON.parse(await f.read()), "선택");
  } catch (e) { status("불러오기 실패: " + e.message, true); }
});

$("#importXml").addEventListener("click", async () => {
  try {
    const f = await fs.getFileForOpening({ types: ["xml"] });
    if (!f) return;
    const ok = await (await project()).importFiles([f.nativePath], true, null, false);
    status(ok ? "XML 가져옴 — 생긴 시퀀스를 열고 '자동 구도 전체 적용'" : "XML 가져오기 실패 — 파일 > 가져오기로 직접 불러오세요", !ok);
  } catch (e) { status("XML 가져오기 실패: " + e.message, true); }
});

$("#applyAll").addEventListener("click", () => applyAll().catch((e) => status("적용 실패: " + e.message, true)));
document.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
  const d = b.dataset;
  (d.act === "scale" ? nudge("scale", +d.v) : nudge("pos", +d.dx, +d.dy)).catch((e) => status("조정 실패: " + e.message, true));
}));
