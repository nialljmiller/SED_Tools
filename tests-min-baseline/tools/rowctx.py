"""pytest plugin (-p rowctx, with -p ctxplugin): label coverage with the subtest row that runs.

Inside ``subtests.test(name)`` the context is ``<nodeid>|row=<name>``; after the row it goes back to
``<nodeid>|run``. Subprocesses a row starts inherit the label, as with ctxplugin.
"""
from _pytest import subtests

from ctxplugin import _label

_enter, _exit = subtests._SubTestContextManager.__enter__, subtests._SubTestContextManager.__exit__


def __enter__(self):
    _label(f"{self.request.node.nodeid}|row={self.msg}")
    return _enter(self)


def __exit__(self, *exc):
    try:
        return _exit(self, *exc)
    finally:
        _label(f"{self.request.node.nodeid}|run")


subtests._SubTestContextManager.__enter__ = __enter__
subtests._SubTestContextManager.__exit__ = __exit__
