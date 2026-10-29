"""Prompts, pickers and input parsers shared by the menus (cli, ml, cube, filters.svo)."""

from typing import Sequence, Optional, List, Union, Tuple
import importlib.util
import math
import re
import shutil
import sys
import textwrap

from .terminal_plots import terminal_color_enabled, terminal_width


def parse_numeric_range(raw: str) -> Optional[Tuple[float, float]]:
    """Parse two numbers separated by comma, whitespace, colon, ``..`` or dash."""
    text = raw.strip()
    if not text:
        return None
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*(?:,|:|\.\.|\s+|-)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*",
        text,
    )
    if match is None:
        return None
    low, high = (float(value) for value in match.groups())
    return (low, high) if low <= high else (high, low)


def parse_multi_selection(spec: str, total: int) -> list[int]:
    """Parse 1-based IDs such as ``1,3-5`` into unique 0-based indexes."""
    chosen: set[int] = set()
    for chunk in (part.strip() for part in spec.split(",")):
        if not chunk:
            continue
        if "-" in chunk:
            bounds = [part.strip() for part in chunk.split("-", 1)]
            if not all(part.isdigit() for part in bounds):
                raise ValueError(f"Invalid range: {chunk}")
            start, end = map(int, bounds)
            if start > end:
                start, end = end, start
            if start < 1 or end > total:
                raise ValueError(f"Range out of bounds: {chunk}")
            chosen.update(range(start - 1, end))
        else:
            if not chunk.isdigit():
                raise ValueError(f"Invalid item: {chunk}")
            value = int(chunk)
            if value < 1 or value > total:
                raise ValueError(f"Selection out of bounds: {value}")
            chosen.add(value - 1)
    return sorted(chosen)


def ask_option(prompt: str, options: Sequence[str], default: Optional[str] = None) -> str:
    """Ask until the answer is one of ``options`` (any case); a blank answer gives ``default`` if there is one."""
    lowered = {o.lower(): o for o in options}
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        if raw.lower() in lowered:
            return lowered[raw.lower()]
        print(f"  Please enter one of: {', '.join(options)}.")


def ask_yes(prompt: str) -> bool:
    """A [y/N] question: only an answer starting with 'y' is yes."""
    return input(prompt).strip().lower().startswith("y")


def ask_range(prompt: str) -> Optional[Tuple[float, float]]:
    """Ask for a range like '3500,8000' until one parses; None for a blank answer (no cut)."""
    while True:
        raw = input(prompt).strip()
        parsed = parse_numeric_range(raw)
        if parsed is not None or not raw:
            return parsed
        print(f"  Could not parse range from '{raw}' — need two values (e.g. '3500,8000')")


def wrap(text: str, width: Optional[int] = None, indent: str = "", hanging: str = "") -> str:
    """``text`` filled to the terminal width; URLs and hyphenated words are never split."""
    return textwrap.fill(" ".join(str(text).split()), width or terminal_width(),
                         initial_indent=indent, subsequent_indent=hanging,
                         break_on_hyphens=False, break_long_words=False)


def cut(text: str, width: int) -> str:
    """``text`` cut to ``width`` characters, ending in '…' when anything was removed."""
    return text if len(text) <= width else text[:max(width - 1, 0)] + "…"


def banner(title: str) -> None:
    """``title`` between two rules of '=', after a blank line."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def section(title: str) -> None:
    """``title`` between two rules of '-', after a blank line."""
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def menu_choice(items: Sequence, prompt: str = "> "):
    """The item a 1-based number picks; prints "Invalid selection" and gives None for anything else."""
    selection = input(prompt).strip()
    try:
        idx = int(selection) - 1
    except ValueError:
        print("Invalid selection")
        return None
    if idx < 0 or idx >= len(items):
        print("Invalid selection")
        return None
    return items[idx]


#: ``default`` of :func:`ask_number` when a blank answer is not allowed (None is a default).
NO_DEFAULT = object()

#: Optional packages a workflow may import, and the extra (pyproject.toml) that installs each.
OPTIONAL_EXTRAS = {"scipy": "full", "torch": "full", "sklearn": "full", "optuna": "ml-tune"}
_PIP_NAMES = {"sklearn": "scikit-learn"}


def missing_packages(modules: Sequence[str]) -> Optional[str]:
    """``Error: missing <packages>: pip install "sed-tools[<extra>]"`` for those of ``modules``
    (keys of OPTIONAL_EXTRAS) that are not installed; None when all are."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    extras = ",".join(sorted({OPTIONAL_EXTRAS[m] for m in missing}))
    return f"Error: missing {', '.join(_PIP_NAMES.get(m, m) for m in missing)}: pip install \"sed-tools[{extras}]\""


def missing_package_error(exc: ImportError) -> Optional[str]:
    """The missing_packages line for an optional package that failed to import, else None (a bug)."""
    root = (exc.name or "").split(".")[0]
    return missing_packages([root]) if root in OPTIONAL_EXTRAS else None


def refused_without(*modules: str) -> bool:
    """True, after printing :func:`missing_packages` on stderr, when any of ``modules`` is not installed.

    The interactive menus ask it when an action is chosen, before that action's questions.
    """
    message = missing_packages(modules)
    if message:
        print(message, file=sys.stderr)
    return message is not None


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
    """Plain-ASCII picker with stable IDs, paging, grid columns, and simple filters.

    Returns the chosen index, or with multi=True (input like "1, 3-5") a list of
    indices; None for quit, -1 for back (if allowed).
    """
    if not options:
        print(f"No {label} options available.")
        return None

    if max_label < 0:
        # Each column holds a 6-character id ("  12. ") before the label.
        max_label = max(len(getattr(x, "label", str(x))) for x in options) + 6

    labels: List[str] = [getattr(o, "label", str(o)) for o in options]
    N = len(labels)
    # A page is never taller than the terminal, so its first rows do not scroll away.
    page_size = min(page_size, _layout(max_label, min_cols, max_cols)[1] * _rows_on_screen())
    page = 1
    filt: Optional[Tuple[str, str]] = None  # ('substr'|'neg'|'regex', pattern)

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

        if multi and re.fullmatch(r"[\d\s,-]+", inp):
            try:
                selected = parse_multi_selection(inp, N)
            except ValueError as exc:
                print(f"{RED}{exc}{RESET}")
                continue
            if selected:
                return selected
            continue

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

    A single column may use the whole line, so a label is cut only when the terminal is narrower than it.
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
