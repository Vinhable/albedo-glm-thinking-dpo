#!/usr/bin/env python3
"""Run production's repo-context service on Windows.

`RepoContextService._ensure_snapshot` downloads each tarball into a
`tempfile.NamedTemporaryFile(...)` and then reopens it by name. Linux allows that; Windows refuses
to reopen a file that is still open (`PermissionError`), so every snapshot fails there. This shim
gives the service's `tempfile` module a `NamedTemporaryFile` that yields a closed file path and
removes it afterwards, and points the cache at its extended-length (`\\\\?\\`) path so deep
repositories extract. Everything else is the unmodified upstream service.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import repo_context_service.core as core  # noqa: E402
from repo_context_service import api  # noqa: E402


class _WindowsTempfile:
    def __getattr__(self, name):
        return getattr(tempfile, name)

    @staticmethod
    @contextlib.contextmanager
    def NamedTemporaryFile(*, dir=None, suffix="", prefix="tmp", **_):  # noqa: N802 - mirrors tempfile
        fd, name = tempfile.mkstemp(dir=dir, suffix=suffix, prefix=prefix)
        os.close(fd)
        try:
            yield SimpleNamespace(name=name)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(name)


if os.name == "nt":
    core.tempfile = _WindowsTempfile()
    # deep repositories exceed Windows' 260-character path limit when extracted; the extended-length
    # form of the cache directory lifts it without a registry change
    cache = os.environ.get("ALBEDO_REPO_CONTEXT_CACHE_DIR", "")
    if cache and not cache.startswith("\\\\?\\"):
        os.environ["ALBEDO_REPO_CONTEXT_CACHE_DIR"] = "\\\\?\\" + str(Path(cache).resolve())

if __name__ == "__main__":
    api.main()
