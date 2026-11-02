"""Old module paths (sed_tools.filter_library and the rest) forward to the moved modules, also for test tools."""

import importlib
import types
from unittest import mock

import sed_tools.cube.tool as tool
import sed_tools.filter_library as old_library
import sed_tools.filters.library as library
import sed_tools.flux_cube_tool as old_tool


def _sentinel():
    return "patched"


def test_monkeypatch_through_an_old_path_is_undone_on_the_moved_module(monkeypatch):
    original = library.get_builder
    with monkeypatch.context() as m:
        m.setattr(old_library, "get_builder", _sentinel)
        assert library.get_builder is _sentinel
        m.setattr(old_library, "get_builder", len)  # nested
        assert library.get_builder is len
    assert library.get_builder is original


def test_monkeypatch_of_a_new_name_through_an_old_path_is_removed_again(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(old_library, "not_there_before", 1, raising=False)
        assert library.not_there_before == 1
    assert not hasattr(library, "not_there_before")


def test_mock_patch_through_an_old_path_is_undone_on_the_moved_module():
    original = library.get_builder
    with mock.patch.object(old_library, "get_builder", _sentinel):
        assert library.get_builder is _sentinel
    assert library.get_builder is original
    with mock.patch("sed_tools.filter_library.get_builder", _sentinel):
        assert library.get_builder is _sentinel
    assert library.get_builder is original


def test_reloading_an_old_path_leaves_the_moved_module_as_it_was(monkeypatch):
    importlib.reload(old_tool)
    assert old_tool.__name__ == "sed_tools.flux_cube_tool"
    assert type(tool) is types.ModuleType
    assert tool.__name__ == tool.__spec__.name == "sed_tools.cube.tool"
    assert tool.__package__ == "sed_tools.cube"
    monkeypatch.setattr(tool.Path, "rglob", lambda self, pattern: iter(()))  # search no folders
    old_tool.discover_flux_cube_files([])  # its `from .model import` needs the moved module's own package
