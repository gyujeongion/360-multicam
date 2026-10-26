#!/bin/bash
# multicam360 설치: 파이썬 환경 + Light-ASD(업스트림 고정 커밋 + 패치) — macOS(Apple Silicon) 기준
set -eo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ffmpeg 필요 (brew install ffmpeg)"; exit 1; }
command -v uv >/dev/null || { echo "uv 필요 (brew install uv)"; exit 1; }
[ -x .venv/bin/python ] || uv venv -q -p 3.11 .venv
uv pip install -q -p .venv -r requirements.txt
if [ ! -d Light-ASD ]; then
  git clone -q https://github.com/Junhua-Liao/Light-ASD.git Light-ASD
  git -C Light-ASD checkout -q "$(cat vendor/light_asd.commit)"
  git -C Light-ASD apply ../vendor/light_asd.patch
fi
.venv/bin/python -m multicam360 doctor
