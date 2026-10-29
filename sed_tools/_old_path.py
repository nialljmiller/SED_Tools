"""The old module paths (sed_tools/filter_library.py and the rest) forward to the modules that now hold the code."""

import sys
import types


class _OldPath(types.ModuleType):
    """A module at an old import path: reading, setting or deleting a name acts on the moved module.

    It keeps its own spec, so python -m and runpy run the old path's file, which calls the moved main.
    """

    def __getattr__(self, name):
        return getattr(self.__dict__["__moved_to__"], name)

    def __setattr__(self, name, value):
        setattr(self.__dict__["__moved_to__"], name, value)

    def __delattr__(self, name):
        delattr(self.__dict__["__moved_to__"], name)

    def __dir__(self):
        return dir(self.__dict__["__moved_to__"])


def forward(name, moved):
    """Make the module at import path ``name`` forward to ``moved``; only its dunder names stay its own."""
    module = sys.modules[name]
    for key in [key for key in module.__dict__ if not key.startswith("__")]:
        del module.__dict__[key]
    module.__dict__["__moved_to__"] = moved
    module.__class__ = _OldPath
