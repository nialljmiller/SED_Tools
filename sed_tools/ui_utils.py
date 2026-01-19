"""UI utility functions for interactive prompts."""

from typing import Sequence, Optional, List, Union, Tuple
import re
import os
import sys
import shutil


def _prompt_choice(
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
        max_label = max(len(getattr(x, "label", str(x))) for x in options) + 4

    # ---- setup ----
    labels: List[str] = [getattr(o, "label", str(o)) for o in options]
    N = len(labels)
    page = 1
    filt: Optional[Tuple[str, str]] = None  # ('substr'|'neg'|'regex', pattern)

    # color toggles
    use_color = use_color and sys.stdout.isatty() and ("NO_COLOR" not in os.environ)
    BOLD = "\x1b[1m" if use_color else ""
    DIM = "\x1b[2m" if use_color else ""
    CYAN = "\x1b[36m" if use_color else ""
    YELL = "\x1b[33m" if use_color else ""
    RED = "\x1b[31m" if use_color else ""
    GREEN = "\x1b[32m" if use_color else ""
    YELLOW = "\x1b[33m" if use_color else ""
    BLUE = "\x1b[34m" if use_color else ""
    MAGENTA = "\x1b[35m" if use_color else ""
    WHITE = "\x1b[37m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    def term_width() -> int:
        try: return shutil.get_terminal_size().columns
        except: return 80

    def apply_filter(idx: List[int]) -> List[int]:
        if not filt: return idx
        kind, patt = filt
        if kind == "substr":
            return [i for i in idx if patt.lower() in labels[i].lower()]
        elif kind == "neg":
            return [i for i in idx if patt.lower() not in labels[i].lower()]
        elif kind == "regex":
            try:
                regex = re.compile(patt, re.IGNORECASE)
                return [i for i in idx if regex.search(labels[i])]
            except re.error:
                return idx
        return idx

    while True:
        kept = apply_filter(list(range(N)))
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
        chunk = kept[start:end]

        # compute column layout
        tw = term_width()
        usable = tw - 10
        col_width = min(usable, max_label)
        n_cols = max(min_cols, min(max_cols, usable // col_width))
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
                label_text = labels[i]
                id_str = f"{i+1:4d}."
                padded = label_text[:col_width-6].ljust(col_width-6)
                row_items.append(f"{CYAN}{id_str}{RESET} {padded}")
            print("  ".join(row_items))

        print()
        if pages > 1:
            ftxt = (f' {DIM}filter="!{filt[1]}"{RESET}' if filt and filt[0] == "neg" else
                   f' {DIM}filter="//{filt[1]}"{RESET}' if filt else "")
            print(f"{DIM}Page {page}/{pages}{ftxt}{RESET}")

        controls = f"{DIM}"
        if end < len(kept):
            controls += "n, p, g <page>, "
        
        controls += "/text, !text, //regex, "
        if multi:
            controls += "list (1,3) or range (1-5), "
        else:
            controls += "id <N> (or just N), "

        controls += "clear"
        if allow_back: controls += ", b"
        controls += ", q" + RESET
        print(controls)

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
        if low.startswith("//"): patt = inp[2:].strip(); filt = ("regex", patt) if patt else None; page = 1; continue
        if low.startswith("!"): patt = inp[1:].strip(); filt = ("neg", patt) if patt else None; page = 1; continue
        if low.startswith("/"): patt = inp[1:].strip(); filt = ("substr", patt) if patt else None; page = 1; continue
        if low.startswith("id "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit():
                k = int(parts[1])
                if 1 <= k <= N: return k - 1
            continue

        # --- Multi Selection Logic ---
        if multi:
            if re.match(r"^[\d\s,-]+$", inp):
                selected_indices = []
                parts = inp.split(',')
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    if '-' in p:
                        a, b = p.split('-', 1)
                        selected_indices.extend(range(int(a), int(b) + 1))
                    else:
                        selected_indices.append(int(p))
                
                valid = [x - 1 for x in sorted(list(set(selected_indices))) if 1 <= x <= N]
                if valid:
                    return valid

        # --- Single Selection Logic ---
        if not multi and inp.isdigit():
            k = int(inp)
            if 1 <= k <= N: return k - 1
            continue

        # Default to substring filter
        filt = ("substr", inp); page = 1