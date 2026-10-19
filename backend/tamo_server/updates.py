"""Updating the app from GitHub: pull the latest code, then restart through studio.sh,
which installs new packages and rebuilds the interface before serving again.

The Windows app is only a window around what this server serves, so this updates
everything except that window. Its version is in desktop/src-tauri/tauri.conf.json; when
that changes, the interface asks for a reinstall.
"""

import json
import os
import subprocess
import threading
import time

from .config import REPO_ROOT

GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}   # never wait for a password nobody can type


class UpdateError(RuntimeError):
    pass


def git(*args: str, timeout: float = 60) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, env=GIT_ENV, capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UpdateError(f"git {args[0]} didn't finish: {error}") from error
    if result.returncode != 0:
        raise UpdateError((result.stderr or result.stdout).strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def shell_version() -> str | None:
    try:
        return json.loads((REPO_ROOT / "desktop" / "src-tauri" / "tauri.conf.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        return None


def current() -> dict:
    try:
        commit = git("rev-parse", "--short", "HEAD", timeout=10)
        date = git("log", "-1", "--format=%cI", timeout=10)
    except UpdateError:
        commit = date = None
    return {"commit": commit, "date": date, "shell_version": shell_version()}


def check() -> dict:
    """Fetches from GitHub and says how many changes are waiting."""
    git("fetch", "--quiet")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", timeout=10)
    behind = int(git("rev-list", "--count", f"HEAD..{upstream}", timeout=10))
    changes = git("log", "--format=%s", f"HEAD..{upstream}", timeout=10).splitlines() if behind else []
    return {**current(), "behind": behind, "changes": changes[:20]}


def pull() -> dict:
    """Brings in the latest code. Refuses rather than overwrite changes made on this PC."""
    before = current()["commit"]
    git("pull", "--ff-only", "--quiet", timeout=300)
    return {"from": before, **current()}


def restart_soon(stop_queue) -> None:
    """Replaces this server with a fresh ./studio.sh once the response has gone out. Same
    process, so whatever started the server (the Windows app, a terminal) keeps it."""
    def restart():
        time.sleep(0.5)
        stop_queue()
        os.execv("/bin/bash", ["bash", str(REPO_ROOT / "studio.sh")])

    threading.Thread(target=restart, name="restart", daemon=True).start()
