"""스킬 문서 ↔ 실제 CLI 정합성 검사. 스킬이 부르는 orbcut 하위 명령·옵션이 전부 실재해야 통과.

usage: python3 tests/check_skill.py [--full]
--full: 완성판 요구 항목(평가·UXP 패널 설치·원격) 문서화까지 검사 (M11)
"""
import os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.environ.get("ORBCUT_SKILL", os.path.expanduser("~/.claude/skills/multicam360/SKILL.md"))
PY = os.path.join(REPO, ".venv", "bin", "python")
fails = []

if not os.path.exists(SKILL):
    print("SKILL.md 없음:", SKILL); sys.exit(1)
text = open(SKILL).read()
fm = re.match(r"---\n(.*?)\n---\n", text, re.S)
if not fm or "name: multicam360" not in fm.group(1) or "description:" not in fm.group(1):
    fails.append("frontmatter(name/description) 누락")

helptext = subprocess.run([PY, "-m", "orbcut", "--help"], cwd=REPO, capture_output=True, text=True).stdout
subs = set(re.findall(r"\{([a-z,\-]+)\}", helptext)[0].split(","))
used = set(re.findall(r"\$MC (?:--json )?([a-z\-]+)", text))
for cmd in sorted(used):
    if cmd not in subs:
        fails.append(f"스킬이 부르는 명령 없음: {cmd}")
    else:
        sub_help = subprocess.run([PY, "-m", "orbcut", cmd, "--help"], cwd=REPO, capture_output=True, text=True).stdout
        for opt in set(re.findall(r"\$MC (?:--json )?" + re.escape(cmd) + r"[^\n`]*", text)):
            for flag in re.findall(r"(--[a-z\-]+)", opt):
                if flag not in sub_help and flag != "--json":
                    fails.append(f"{cmd}: 옵션 {flag} 없음")

for path in re.findall(r"`\$REPO/([^`\s]+)`", text) + re.findall(r"`(orbcut/[a-z_]+\.py)`", text):
    if not os.path.exists(os.path.join(REPO, path)):
        fails.append(f"문서가 가리키는 파일 없음: {path}")

if "--full" in sys.argv:
    for need, why in [("$MC eval", "정확도 평가 명령"), ("plugin", "UXP 패널 설치 안내"), ("--host ", "원격 실행"),
                      ("labels", "정답 라벨 도구 안내")]:
        if need not in text:
            fails.append(f"완성판 누락: {why} ({need})")

print("\n".join(fails) if fails else f"SKILL OK — 명령 {sorted(used)}")
sys.exit(1 if fails else 0)
