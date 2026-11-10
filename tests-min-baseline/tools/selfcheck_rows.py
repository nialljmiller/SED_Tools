import logging, os, random
import numpy as np
import pytest
import requests
from support import NetworkUsed

SEEN = {}

def _record(t, label):
    from sed_tools import config, njm_filter_grabber
    SEEN.setdefault("tmp", []).append(t.tmp)
    SEEN.setdefault("config", []).append(config._CONFIG_DIR)
    import torch
    mps = torch.rand(1, device="mps").item() if torch.backends.mps.is_available() else None
    SEEN.setdefault("draw", []).append((random.random(), float(np.random.random()), torch.rand(1).item(), mps))
    assert t.capsys.readouterr().out == "", "output leaked from an earlier row"
    assert not t.caplog.records, "log leaked from an earlier row"
    assert os.environ.get("ROW_LEAK") is None and os.getcwd() == SEEN.setdefault("cwd", os.getcwd())
    assert not getattr(requests, "LEAKED", False)
    print("printed by", label)
    logging.getLogger("sed_tools").warning("logged by %s", label)
    t.mp.setenv("ROW_LEAK", "1"); t.mp.setattr(requests, "LEAKED", True, raising=False); t.mp.chdir(t.tmp)
    njm_filter_grabber._STATE.reset()

def _network(t):
    with pytest.raises(NetworkUsed):
        requests.get("https://example.invalid/x")

def test_selfcheck(rows):
    """Copy into tests/ and run: one row (b_fails) fails on purpose and one skips; every other check passes."""
    pytest.importorskip("torch")
    rows({"a": "first", "b_fails": "second", "c": "third"},
         lambda t, label: (_record(t, label), label != "second" or (_ for _ in ()).throw(AssertionError("b broke on purpose"))))
    rows({"d_tuple": (1, 2), "e_skips": (0, 0)}, lambda t, x, y: pytest.skip("no torch") if x == 0 else _record(t, "tuple"))
    rows({"f_network": _network})
    tmps = SEEN["tmp"]; assert len(set(tmps)) == len(tmps) == 4, tmps
    assert len(set(SEEN["config"])) == 4
    assert len(set(SEEN["draw"])) == 1, SEEN["draw"]
    with pytest.raises(AssertionError, match="two rows are named 'a'"):
        rows({"a": "again"}, lambda t, label: None)
