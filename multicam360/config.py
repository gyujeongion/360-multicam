"""경로·외부 의존 설정. 환경변수로 덮어쓸 수 있다."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def asd_dir():
    """패치된 Light-ASD 체크아웃 위치 (MULTICAM360_ASD_DIR, 기본 <repo>/Light-ASD)."""
    return os.environ.get("MULTICAM360_ASD_DIR", os.path.join(REPO, "Light-ASD"))


def _user_config():
    import json
    p = os.path.expanduser("~/.config/multicam360/config.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def nas_host():
    """`nas:/...` 경로를 읽을 ssh 호스트 (MULTICAM360_NAS_HOST 또는 ~/.config/multicam360/config.json의 nas_host)."""
    h = os.environ.get("MULTICAM360_NAS_HOST") or _user_config().get("nas_host")
    if not h:
        raise RuntimeError("NAS 호스트 미설정: MULTICAM360_NAS_HOST 환경변수 또는 ~/.config/multicam360/config.json {\"nas_host\": \"...\"}")
    return h


def nas_ffmpeg():
    """NAS 쪽 ffmpeg 경로 (Synology 기본 /usr/bin/ffmpeg)."""
    return os.environ.get("MULTICAM360_NAS_FFMPEG") or _user_config().get("nas_ffmpeg", "/usr/bin/ffmpeg")


def remote_host():
    """원격 실행 기본 호스트 (MULTICAM360_REMOTE_HOST 또는 config remote_host). 없으면 None — 원격 테스트는 건너뜀."""
    return os.environ.get("MULTICAM360_REMOTE_HOST") or _user_config().get("remote_host")


def remote_dir():
    """원격 호스트의 코드 위치(홈 기준 상대경로)."""
    return os.environ.get("MULTICAM360_REMOTE_DIR") or _user_config().get("remote_dir", "multicam360")


def test_nas_src():
    """NAS 연동 테스트에 쓸 원본 경로 (MULTICAM360_TEST_NAS_SRC 또는 config test_nas_src). 없으면 그 테스트 건너뜀."""
    return os.environ.get("MULTICAM360_TEST_NAS_SRC") or _user_config().get("test_nas_src")


def local_max_s():
    """이 기기에서 돌릴 수 있는 최대 구간 길이(초). 0 = 제한 없음(렌더 전용 기기에 설정).
    기본 60초 — 8K 원본을 ffmpeg 파이프로 여러 개 동시에 디코딩하면 노트북 메모리가 고갈된다
    (실측 2026-10-21: 36GB 맥북에서 ffmpeg 8개가 약 50GB 점유 → 커널 패닉 재부팅)."""
    v = os.environ.get("MULTICAM360_LOCAL_MAX_S") or _user_config().get("local_max_s", 60)
    return float(v)
