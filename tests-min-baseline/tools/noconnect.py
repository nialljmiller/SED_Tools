"""pytest plugin (-p noconnect): block the network at the socket and name each test that tried.

Every connect or name lookup outside AF_UNIX raises OSError, as a machine without a network does.
The tests that tried are written to $NOCONNECT_LOG, one id per line.
"""
import os
import socket

import pytest

_current = {"id": ""}
_log = os.environ.get("NOCONNECT_LOG")


def _refuse(*args, **kwargs):
    if _log:
        with open(_log, "a") as fh:
            fh.write(_current["id"] + "\n")
    raise OSError("network blocked (noconnect)")


_connect = socket.socket.connect


def _guarded_connect(self, address):
    if self.family == getattr(socket, "AF_UNIX", None):
        return _connect(self, address)
    _refuse()


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = lambda self, address: _guarded_connect(self, address)
socket.getaddrinfo = _refuse
socket.create_connection = _refuse


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _current["id"] = item.nodeid
    yield
