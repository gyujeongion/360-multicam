"""원격 위임 공통: config의 remote_host가 있으면 코드 동기화 후 그 호스트에서 같은 테스트를 실행한다."""
import os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from orbcut.config import remote_dir, remote_host  # noqa: E402

ON_REMOTE = os.environ.get("ORBCUT_ON_REMOTE") == "1"


def delegate(cmd):
    """원격에서 `cmd` 실행(작업 폴더 = 원격 코드 위치) → 종료 코드. 원격 미설정이면 None."""
    host = remote_host()
    if not host or ON_REMOTE:
        return None
    subprocess.run([sys.executable, "-c", f"from orbcut import remote; remote.sync({host!r})"], cwd=REPO, check=True)
    return subprocess.run(["ssh", host, f"cd {remote_dir()} && ORBCUT_ON_REMOTE=1 {cmd}"]).returncode
