"""HTTP helpers the SVO and MSG grid sources share: asking a grid index once, and sessions that retry."""

import requests
from requests.adapters import HTTPAdapter

#: Connect and read timeouts for a source's grid index.  Discovery asks every source before
#: anything is downloaded, so a source that is down or hanging must cost seconds, not minutes:
#: the index is asked once, without the retries a download gets (see ``mount_discovery``).
DISCOVERY_TIMEOUT = (5, 10)


def mount_discovery(session, url):
    """Make ``session`` ask ``url`` (a grid index) once: no retries, unlike the rest of the source."""
    session.mount(url, HTTPAdapter(max_retries=0))


def discovery_failure(exc):
    """Why a grid index could not be read, in a few plain words."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"answered HTTP {status}"
    if isinstance(exc, requests.Timeout):
        return f"did not answer within {DISCOVERY_TIMEOUT[0]:g}+{DISCOVERY_TIMEOUT[1]:g} s"
    if isinstance(exc, requests.ConnectionError):
        return "could not be reached"
    return str(exc)


def retrying_session(user_agent, retry, separate_adapters=False):
    """A requests session that sends ``user_agent`` and retries http and https requests as ``retry`` says.

    http and https share one adapter, or with ``separate_adapters`` (MSG's session) get one each.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", HTTPAdapter(max_retries=retry) if separate_adapters else adapter)
    return session
