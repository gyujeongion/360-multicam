// M9 오프라인 검증: 매니페스트 v5 필수 항목 / HTML 참조 / JS 구문 / 프리미어 UXP API 호출이 공식 문서 목록에 있는지.
// 실기(프리미어 로드)는 D-02 — 프리미어가 비었을 때.
import { readFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const fails = [];
const m = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
if (m.manifestVersion !== 5) fails.push("manifestVersion은 5");
if (m.host?.app !== "premierepro") fails.push("host.app은 premierepro");
if (!/^2[5-9]\.\d+/.test(m.host?.minVersion || "")) fails.push("host.minVersion 25.6 이상 (UXP for Premiere)");
if (!m.entrypoints?.some((e) => e.type === "panel" && e.id)) fails.push("panel entrypoint 없음");
for (const k of ["id", "name", "version", "main"]) if (!m[k]) fails.push(`manifest.${k} 없음`);
if (!existsSync(join(root, m.main))) fails.push(`main 파일 없음: ${m.main}`);

const html = readFileSync(join(root, m.main), "utf8");
for (const ref of [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((x) => x[1]))
  if (!/^https?:/.test(ref) && !existsSync(join(root, ref))) fails.push(`HTML 참조 파일 없음: ${ref}`);
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((x) => x[1]));

const js = readFileSync(join(root, "index.js"), "utf8");
try { execFileSync(process.execPath, ["--check", join(root, "index.js")]); } catch (e) { fails.push("JS 구문 오류: " + e.stderr); }
for (const id of [...js.matchAll(/\$\("#([\w-]+)"\)/g)].map((x) => x[1])) if (!ids.has(id)) fails.push(`JS가 찾는 #${id}가 HTML에 없음`);

// 2026-10-21 developer.adobe.com/premiere-pro/uxp/ppro-reference 에서 확인한 이름만 허용
const DOCUMENTED = new Set([
  "getActiveProject", "getActiveSequence", "executeTransaction", "importFiles", "openSequence", "setActiveSequence",
  "getVideoTrack", "getVideoTrackCount", "getPlayerPosition", "setPlayerPosition", "getSelection", "clearSelection",
  "getFrameSize", "getTrackItems", "getComponentChain", "getStartTime", "getEndTime", "getInPoint", "getName",
  "getComponentAtIndex", "getComponentCount", "getParam", "getParamCount", "getMatchName", "getDisplayName",
  "createKeyframe", "createSetValueAction", "createSetTimeVaryingAction", "getValueAtTime", "getStartValue",
  "addAction", "createWithSeconds", "createWithTicks", "PointF", "getProjectItem", "cast", "getMediaFilePath", "lockedAccess",
]);
const UXP_FS = new Set(["getFileForOpening", "read", "getParent", "getEntryWithUrl"]);
const JS_BUILTIN = new Set(["querySelector", "querySelectorAll", "addEventListener", "forEach", "find", "toFixed", "append",
  "appendChild", "createElement", "toggle", "parse", "abs", "map", "catch", "push", "then", "toLowerCase", "substring", "lastIndexOf"]);
const LOCAL = new Set([...js.matchAll(/function\s+(\w+)\s*\(/g)].map((x) => x[1]));   // 파일 안 정의 함수
const called = new Set([...js.matchAll(/\.([a-zA-Z_]\w*)\s*\(/g)].map((x) => x[1]));
for (const c of called) if (!DOCUMENTED.has(c) && !UXP_FS.has(c) && !JS_BUILTIN.has(c) && !LOCAL.has(c)) fails.push(`문서에 없는 API 호출: .${c}()`);
if (!/new ppro\.PointF\(/.test(js)) fails.push("Position은 ppro.PointF로 설정해야 함");

console.log(fails.length ? fails.join("\n") : `PLUGIN OK — API ${[...called].filter((c) => DOCUMENTED.has(c)).length}종 문서 확인`);
process.exit(fails.length ? 1 : 0);
