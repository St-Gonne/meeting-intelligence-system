"""Scoped discovery of the existing local audio tools for background workers."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil


class AudioToolUnavailable(RuntimeError):
    def __init__(self, tool):
        self.tool = tool
        super().__init__(f"{tool} is unavailable")


def _fallback_directories():
    return (Path.home() / "homebrew/bin",
            Path(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")) / "bin", Path("/usr/local/bin"))


@contextmanager
def audio_tool_environment(tools=("ffmpeg", "ffprobe")):
    """Preserve PATH precedence; supply installed Homebrew tools when missing.

    This changes only the calling worker scope and always restores its prior
    environment. It never installs tools or changes a service or shell profile.
    """
    if any(tool not in {"ffmpeg", "ffprobe"} for tool in tools):
        raise ValueError("unsupported audio tool")
    missing = [tool for tool in tools if not shutil.which(tool)]
    candidates = _fallback_directories()
    fallback = next((p for p in candidates if all((p / t).is_file() and os.access(p / t, os.X_OK) for t in missing)), candidates[0])
    for tool in missing:
        binary = fallback / tool
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise AudioToolUnavailable(tool)
    previous = os.environ.get("PATH")
    if missing:
        os.environ["PATH"] = (previous + os.pathsep if previous else "") + str(fallback)
    try:
        yield
    finally:
        if missing:
            if previous is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous
