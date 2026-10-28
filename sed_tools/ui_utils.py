"""UI utility functions for interactive prompts."""

from typing import Sequence, Optional, List, Union, Tuple
import math
import re
import shutil

from .parsing import parse_multi_selection

from .terminal_plots import terminal_color_enabled, terminal_width


#: ``default`` of :func:`ask_number` when a blank answer is not allowed (None is a default).
NO_DEFAULT = object()


def ask_number(prompt: str, default=NO_DEFAULT, *, whole: bool = False, positive: bool = False,
               words: Sequence[str] = ()):
    """Ask until the answer is a finite number (whole, positive if asked) or one of ``words``.

    A blank answer gives ``default`` when there is one (None included).
    Every interactive number question uses this, so a typo asks again.
    """
    kind = " or ".join([("a positive " if positive else "a ") + ("whole number" if whole else "number"),
                        *(repr(w) for w in words)])
    while True:
        raw = input(prompt).strip()
        if not raw and default is not NO_DEFAULT:
            return default
        if raw.lower() in words:
            return raw.lower()
        try:
            value = int(raw) if whole else float(raw)
        except ValueError:
            value = None
        if value is not None and math.isfinite(value) and (value > 0 or not positive):
            return value
        print(f"  {raw!r} is not {kind}; try again." if raw else f"  Please enter {kind}.")


def ask_numbers(prompt: str) -> List[float]:
    """Ask until the answer is a comma-separated list of finite numbers."""
    while True:
        raw = input(prompt).strip()
        try:
            values = [float(token) for token in raw.split(',')]
        except ValueError:
            values = []
        if values and all(math.isfinite(v) for v in values):
            return values
        print(f"  {raw!r} is not a comma-separated list of numbers; try again.")


def prompt_choice(
    options: Sequence,
    label: str,
    allow_back: bool = False,
    page_size: int = 100,
    max_label: int = -1,
    min_cols: int = 1,
    max_cols: int = 3,
    use_color: bool = True,
    multi: bool = False,
) -> Union[int, List[int], None]:
    """
    Plain-ASCII picker with stable IDs, paging, grid columns, and simple filters.
    If multi=True, allows input like "1, 3-5" and returns a List[int] of indices.
    Otherwise returns a single int.
    Returns None for quit, -1 for back (if allowed).
    """
    if not options:
        print(f"No {label} options available.")
        return None

    if max_label < 0:
        # Each column holds a 6-character id ("  12. ") before the label.
        max_label = max(len(getattr(x, "label", str(x))) for x in options) + 6

    # ---- setup ----
    labels: List[str] = [getattr(o, "label", str(o)) for o in options]
    N = len(labels)
    # A page is never taller than the terminal, so its first rows do not scroll away.
    page_size = min(page_size, _layout(max_label, min_cols, max_cols)[1] * _rows_on_screen())
    page = 1
    filt: Optional[Tuple[str, str]] = None  # ('substr'|'neg'|'regex', pattern)

    # color toggles
    use_color = use_color and terminal_color_enabled("auto")
    DIM = "\x1b[2m" if use_color else ""
    RED = "\x1b[31m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    while True:
        kept = _apply_filter(labels, filt)
        if not kept:
            print(f"{RED}No matches for filter.{RESET}")
            inp = input(f"{DIM}Press Enter to clear filter or 'q' to quit: {RESET}").strip()
            if inp.lower() == "q": return None
            filt = None
            page = 1
            continue

        pages = (len(kept) + page_size - 1) // page_size
        if page < 1: page = 1
        if page > pages: page = pages
        start = (page - 1) * page_size
        end = min(start + page_size, len(kept))

        _print_page(label, labels, kept[start:end], max_label, min_cols, max_cols, use_color)
        if pages > 1:
            ftxt = (f' {DIM}filter="!{filt[1]}"{RESET}' if filt and filt[0] == "neg" else
                   f' {DIM}filter="//{filt[1]}"{RESET}' if filt else "")
            print(f"{DIM}Page {page}/{pages}{ftxt}{RESET}")
        print(_controls(end < len(kept), multi, allow_back, use_color))

        inp = input("> ").strip()
        if not inp: continue
        low = inp.lower()

        if low in ("q", "quit", "exit"): return None
        if allow_back and low in ("b", "back"): return -1
        if low == "n": page += 1; continue
        if low == "p": page -= 1; continue
        if low.startswith("g "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit(): page = int(parts[1])
            continue
        if low == "clear": filt = None; page = 1; continue
        if low.startswith("//"):
            filt = _regex_filter(inp[2:].strip(), filt, RED, RESET); page = 1; continue
        if low.startswith("!"): patt = inp[1:].strip(); filt = ("neg", patt) if patt else None; page = 1; continue
        if low.startswith("/"): patt = inp[1:].strip(); filt = ("substr", patt) if patt else None; page = 1; continue
        if low.startswith("id "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit():
                k = int(parts[1])
                if 1 <= k <= N: return [k - 1] if multi else k - 1
            continue

        # --- Multi Selection Logic ---
        if multi and re.fullmatch(r"[\d\s,-]+", inp):
            try:
                selected = parse_multi_selection(inp, N)
            except ValueError as exc:
                print(f"{RED}{exc}{RESET}")
                continue
            if selected:
                return selected
            continue

        # --- Single Selection Logic ---
        if not multi and inp.isdigit():
            k = int(inp)
            if 1 <= k <= N: return k - 1
            continue

        # Default to substring filter
        filt = ("substr", inp); page = 1


def _apply_filter(labels: List[str], filt: Optional[Tuple[str, str]]) -> List[int]:
    """Indices of the labels that pass the filter (all of them without one)."""
    idx = list(range(len(labels)))
    if not filt:
        return idx
    kind, patt = filt
    if kind == "substr":
        return [i for i in idx if patt.lower() in labels[i].lower()]
    if kind == "neg":
        return [i for i in idx if patt.lower() not in labels[i].lower()]
    regex = re.compile(patt, re.IGNORECASE)  # checked by _regex_filter when it was entered
    return [i for i in idx if regex.search(labels[i])]


def _regex_filter(patt: str, current: Optional[Tuple[str, str]], red: str, reset: str):
    """The filter for '//PATTERN'; an invalid pattern is reported and the current filter kept."""
    if not patt:
        return None
    try:
        re.compile(patt, re.IGNORECASE)
    except re.error as exc:
        print(f"{red}Invalid regular expression {patt!r}: {exc}{reset}")
        return current
    return ("regex", patt)


def _print_page(label: str, labels: List[str], chunk: List[int], max_label: int,
                min_cols: int, max_cols: int, use_color: bool) -> None:
    """The heading and one page of numbered labels, laid out in columns."""
    BOLD = "\x1b[1m" if use_color else ""
    CYAN = "\x1b[36m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    tw = terminal_width()
    col_width, n_cols = _layout(max_label, min_cols, max_cols)
    n_rows = (len(chunk) + n_cols - 1) // n_cols

    print(f"\n{BOLD}{label}{RESET}")
    print("=" * min(tw, 60))

    for row in range(n_rows):
        row_items = []
        for col in range(n_cols):
            flat_idx = row + col * n_rows
            if flat_idx >= len(chunk):
                continue
            i = chunk[flat_idx]
            id_str = f"{i+1:4d}."
            room = col_width - 6
            label = labels[i] if len(labels[i]) <= room else labels[i][:max(room - 1, 0)] + "…"
            padded = label.ljust(room)
            row_items.append(f"{CYAN}{id_str}{RESET} {padded}")
        print("  ".join(row_items))

    print()


def _layout(max_label: int, min_cols: int, max_cols: int) -> Tuple[int, int]:
    """Column width and number of columns for entries up to ``max_label`` wide.

    A single column may use the whole line, so a label is cut (with '…') only
    when the terminal is narrower than it.
    """
    usable = terminal_width() - 10
    col_width = min(usable, max_label)
    n_cols = max(min_cols, min(max_cols, usable // col_width))
    if n_cols == 1:
        col_width = min(terminal_width() - 1, max_label)
    return col_width, n_cols


def _rows_on_screen() -> int:
    """Rows of entries that fit on the terminal with the heading, page and control lines and prompt."""
    return max(10, shutil.get_terminal_size(fallback=(80, 24)).lines - 8)


def _controls(more_pages: bool, multi: bool, allow_back: bool, use_color: bool) -> str:
    """The line listing the commands the picker accepts."""
    DIM = "\x1b[2m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    controls = f"{DIM}"
    if more_pages:
        controls += "n, p, g <page>, "

    controls += "/text, !text, //regex, "
    if multi:
        controls += "list (1,3) or range (1-5), "
    else:
        controls += "id <N> (or just N), "

    controls += "clear"
    if allow_back: controls += ", b"
    controls += ", q" + RESET
    return controls
