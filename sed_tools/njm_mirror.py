"""NJM, the SED_Tools data mirror: its address, and what this process knows about it.

The filter and spectra clients share this state, so an outage is reported once and every client
leaves NJM alone for ``NJM_PAUSE`` seconds.
"""

import logging
import os
import threading
from pathlib import Path
from time import monotonic
from typing import Dict, Optional, Union

import requests
import urllib3

# Outage warnings are logged under the filter client's name.
logger = logging.getLogger("sed_tools.njm_filter_grabber")

BASE_URL = "https://nillmill.ddns.net/sed_tools"
USER_AGENT = "SED_Tools/1.0 (NJM Mirror)"
#: Seconds to wait for NJM to accept a connection: a mirror that is down fails fast.
NJM_CONNECT_TIMEOUT = 5.0
#: Seconds NJM is left alone after it stops answering.
NJM_PAUSE = 300
#: What FilterLibrary.facilities() and instruments() put in "errors" when NJM is down; api leaves it out of the failures.
NJM_NOT_LISTED = "NJM unavailable"


def cache_root(filter_root: "Union[str, Path]") -> "Path":  # quoted: filter_catalogue re-exports it, signature as text
    """Where NJM catalogue files are cached for a filter root (beside ``atmospheres/``)."""
    return Path(filter_root).parent / "cache" / "njm"


def _status(exc: Exception) -> Optional[int]:
    """The HTTP status of a failed request, or None when there was no answer."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def _body(response) -> bytes:
    """The response bytes as served (JSON is UTF-8 whatever the Content-Type says)."""
    content = getattr(response, "content", None)
    return content if isinstance(content, bytes) else response.text.encode("utf-8")


def _write_atomic(path: Path, data) -> None:
    """Write ``data`` (bytes, or text as UTF-8) whole, through a ``.<name>.part`` file: never a partial file."""
    tmp = path.with_name(f".{path.name}.part")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def plain_reason(reason) -> str:
    """Why NJM could not be reached, in a few plain words rather than a raw exception's text."""
    status = _status(reason)
    if status is not None:  # NJM answered, but refused (a burst block, say)
        return f"HTTP {status}"
    text = str(reason)
    if len(text) <= 60 and "(" not in text:
        return text
    lowered = text.lower()
    if isinstance(reason, requests.exceptions.Timeout) or "timed out" in lowered:
        return "it did not answer in time"
    if isinstance(reason, requests.exceptions.SSLError):
        return "its certificate or TLS handshake failed"
    for words, plain in (("refused", "connection refused"), ("reset by peer", "connection reset"),
                         ("aborted", "connection reset"), ("no route", "no route to host"),
                         ("name or service not known", "host name not found"),
                         ("nodename nor servname", "host name not found"),
                         ("name resolution", "host name not found")):
        if words in lowered:
            return plain
    return "could not connect"


class _MirrorState:
    """What this process knows about NJM, shared by every NJM client."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.available = False  # a check or a request found NJM answering
        self.unverified_ssl = False  # NJM's certificate could not be verified; requests skip the check
        self.unreachable: Optional[str] = None  # why NJM is not asked, until `until` (monotonic())
        self.until = 0.0
        self.reported = False  # the current outage has been reported
        self.index: Optional[Dict] = None  # NJM's index.json, read at `index_read` (monotonic())
        self.index_read = 0.0


_STATE = _MirrorState()


def pause(reason: str, message: str, log: logging.Logger = logger) -> None:
    """Leave NJM alone for ``NJM_PAUSE`` seconds in this whole process; ``message`` is warned once per outage."""
    with _STATE.lock:
        _STATE.available = False
        _STATE.unreachable, _STATE.until = reason, monotonic() + NJM_PAUSE
        report, _STATE.reported = not _STATE.reported, True
    if report:
        log.warning(message)


def paused() -> bool:
    """Whether NJM stopped answering, in this process, less than ``NJM_PAUSE`` seconds ago."""
    with _STATE.lock:
        if _STATE.unreachable is not None and monotonic() >= _STATE.until:
            _STATE.unreachable = None
        return _STATE.unreachable is not None


def skip_ssl_verification(session: requests.Session) -> None:
    """Switch ``session`` to skip certificate checks, without urllib3's warning on every request."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session.verify = False
