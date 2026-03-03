from sed_tools.cli import _parse_multi_selection
from sed_tools.svo_filter_grabber import parse_multi_selection


def test_parse_multi_selection_accepts_ranges_and_ids() -> None:
    assert parse_multi_selection("1,3-5,5", 6) == [0, 2, 3, 4]
    assert _parse_multi_selection("2-1,4", 5) == [0, 1, 3]


def test_parse_multi_selection_rejects_out_of_bounds() -> None:
    try:
        parse_multi_selection("7", 6)
    except ValueError as exc:
        assert "out of bounds" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")
