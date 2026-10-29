"""The old module paths (sed_tools/filter_library.py and the rest) forward to the modules that now hold the code."""

import importlib
import sys
import types

_MISSING = object()


class _OldPath(types.ModuleType):
    """A module at an old import path: reading, setting or deleting a name acts on the moved module.

    It keeps its own spec, so python -m and runpy run the old path's file, which calls the moved main.
    Its __all__ is what `from <old path> import *` gave in 0.5.0.
    """

    def __getattr__(self, name):
        restored = self.__dict__["__restored__"]
        if name in restored:  # a name the old module had that the moved one no longer imports
            module, _, attribute = restored[name].partition(":")
            value = importlib.import_module(module)
            return getattr(value, attribute) if attribute else value
        return getattr(self.__dict__["__moved_to__"], name)

    # A test tool that patches a name through the old path undoes it either by setting the old value back
    # (pytest before 9.2, unittest.mock) or, finding nothing in this module's own __dict__, by deleting it
    # (pytest 9.2). Each name keeps a stack of the values its sets replaced, so a delete puts back the
    # previous value instead of deleting the moved module's name.
    def __setattr__(self, name, value):
        moved = self.__dict__["__moved_to__"]
        replaced = self.__dict__["__replaced__"].setdefault(name, [])
        if replaced and replaced[-1] is value:  # setting back what an earlier set replaced
            replaced.pop()
        else:
            replaced.append(getattr(moved, name, _MISSING))
        setattr(moved, name, value)

    def __delattr__(self, name):
        moved = self.__dict__["__moved_to__"]
        replaced = self.__dict__["__replaced__"].get(name)
        previous = replaced.pop() if replaced else _MISSING
        if previous is _MISSING:
            delattr(moved, name)
        else:
            setattr(moved, name, previous)

    def __dir__(self):
        return sorted(set(dir(self.__dict__["__moved_to__"])) | set(self.__dict__["__restored__"]))


def forward(name, moved, **restored):
    """Make the module at import path ``name`` forward to ``moved``; only its dunder names stay its own.

    ``restored`` maps each name the old module had, and ``moved`` no longer has, to where it comes from
    ("module" or "module:attribute"); it is imported only when used.
    """
    module = sys.modules[name]
    for key in [key for key in module.__dict__ if not key.startswith("__")]:
        del module.__dict__[key]
    module.__dict__["__moved_to__"] = moved
    module.__dict__["__restored__"] = restored
    module.__dict__["__replaced__"] = {}
    module.__class__ = _OldPath
