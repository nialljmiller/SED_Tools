"""Model-free dispatcher for phase-4 block-A research chunks (Nill's request cf1ebced, manager Claude).

Run every few minutes (systemd user timer phase4-dispatch.timer). Per agent in research/chunks.tsv it keeps one
chunk running: Codex through the room bridge (message from user, as earlier phase watchers did),
Gemini as a direct agy job, Claude as a headless `claude -p` worker. When a chunk finishes it runs
verify_a.py on it. When verified chunks are waiting, it posts one integration request to @claude.
State: research/state.json. Log: research/dispatch.log. Stop: create research/STOP.
Grok was removed from all workflows by Nill on 2026-10-26: it is never launched, and its unfinished chunk A24 was split
into A25 (Codex) and A26 (Gemini). An optional fourth chunks.tsv column is appended to that chunk's prompt.
"""
import json, os, pathlib, re, sqlite3, subprocess, sys, time

P4 = pathlib.Path("/home/nill/SED_Workspace/filter-expansion/phase4")
R = P4 / "research"; STATE = R / "state.json"; LOG = R / "dispatch.log"
PY = "/home/nill/SED_Workspace/filter-expansion/.venv/bin/python"
CHAT = "/home/nill/.local/bin/agent-chat"
DB = pathlib.Path.home() / ".local/state/agent-chat/messages.db"
AGENTS = ("codex", "gemini", "claude")  # Grok removed 2026-10-26
MAX_ATTEMPTS = 3; STALE = 3 * 3600; INTEGRATE_BATCH = 3
now = time.time()


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%FT%TZ ", time.gmtime()) + msg + "\n")


def status(chunk):
    rep = R / "A" / chunk / "REPORT.md"
    m = re.search(r"^STATUS:\s*(\w+)", rep.read_text(), re.M) if rep.exists() else None
    return m.group(1).upper() if m else None


def activation_ended(path):
    con = sqlite3.connect(DB)
    # messages.db stores paths relative to the share root; accept either form.
    rel = os.path.relpath(path, "/home/nill/agent-share")
    row = con.execute("SELECT a.status FROM activations a JOIN messages m ON m.id=a.message WHERE m.path IN (?, ?)",
                      (path, rel)).fetchone()
    return bool(row) and row[0] not in ("running", "queued")


def alive(pid):
    return pid and pathlib.Path(f"/proc/{pid}").exists()


def send(agent, subject, body):
    r = subprocess.run([CHAT, "send", "--from", "user", "--to", f"@{agent}", "--project", "sed-tools-work",
                        "--labels", "research,filters", "--mode", "task", "--subject", subject, "--body", body],
                       capture_output=True, text=True)
    m = re.search(r"Sent: (\S+)", r.stdout)
    log(f"posted to @{agent}: {subject} -> {m.group(1) if m else r.stdout + r.stderr}")
    return m.group(1) if m else None


def prompt(chunk, instruments, agent, attempt, note=""):
    d = R / "A" / chunk
    again = (f" This is attempt {attempt}: files already in {d} are kept; finish the instruments that have no file yet."
             if attempt > 1 else "")
    voice = ("Your final room reply: two lines pointing to REPORT.md." if agent == "codex" else
             "You are a delegated worker for the Claude manager: do not post in the chat room.")
    return (f"Phase 4 research chunk {chunk} for Nill's request cf1ebced (filter metadata: credit, licence terms, "
            f"descriptions). Posted by the Claude manager, which verifies and integrates; Nill authorised this "
            f"delegation. Read and follow {R}/BRIEF-A.md. Your instruments (NJM Facility/Instrument): "
            f"{' '.join(instruments)}. Write only in {d}/ (REPORT.md plus one JSON per instrument). Research only: "
            f"no edits to NJM, the server repo or SED_Tools, no commits, no deleting. It is a small bounded chunk: "
            f"stay within your budget; if you must stop, write STATUS: PARTIAL with what's left.{again}{' ' + note if note else ''} {voice}")


def launch(chunk, agent, instruments, attempt, st, note=""):
    if agent not in AGENTS:
        log(f"refused to launch {chunk} on disabled agent {agent}"); return
    d = R / "A" / chunk; d.mkdir(parents=True, exist_ok=True)
    text = prompt(chunk, instruments, agent, attempt, note)
    rec = {"agent": agent, "attempt": attempt, "posted": now}
    if agent == "codex":
        rec["msg"] = send(agent, f"Phase 4 research chunk {chunk}", text)
    else:
        (d / f"prompt-{attempt}.txt").write_text(text)
        logf = open(d / f"run-{attempt}.log", "w")
        if agent == "gemini":
            model = "gemini-3.8-flash-high" if st.get("gemini_flash") else "gemini-4-argon-high"
            cmd = ["timeout", "3600", "/home/nill/.local/bin/agy", "--dangerously-skip-permissions", "--model", model,
                   "--output-format", "json", f"--print={text}"]
            rec["model"] = model
        else:
            cmd = ["timeout", "3600", "/home/nill/.local/bin/claude", "-p", "--dangerously-skip-permissions", "--output-format", "json", text]
        p = subprocess.Popen(cmd, cwd=d, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        rec["pid"] = p.pid
        log(f"launched {agent} {chunk} attempt {attempt} pid {p.pid} {rec.get('model', '')}")
    st["chunks"][chunk] = rec


def finished(chunk, rec):
    s = status(chunk)
    if s in ("DONE", "PARTIAL"):
        return s
    if rec.get("msg") and activation_ended(rec["msg"]):
        return f"ENDED({s})"
    if rec.get("pid") and not alive(rec["pid"]):
        return f"EXITED({s})"
    if now - rec["posted"] > STALE:
        return f"STALE({s})"
    return None


def main():
    if (R / "STOP").exists():
        return
    st = json.loads(STATE.read_text()) if STATE.exists() else {"chunks": {}}
    rows = [l.rstrip("\n").split("\t") for l in open(R / "chunks.tsv") if l.strip()]
    order = {r[0]: (r[1], r[2].split()) for r in rows}
    notes = {r[0]: r[3] for r in rows if len(r) > 3}
    for c, rec in st["chunks"].items():
        if "finished" in rec:
            continue
        f = finished(c, rec)
        if not f:
            continue
        rec["finished"], rec["result"] = now, f
        log(f"{c} ({rec['agent']}) finished: {f}")
        log_text = (R / "A" / c / f"run-{rec['attempt']}.log")
        if rec["agent"] == "gemini" and log_text.exists() and re.search(r"(?i)quota|resource.?exhausted|429",
                                                                        log_text.read_text()[-4000:]):
            st["gemini_flash"] = True; log("gemini quota error seen: later gemini jobs use flash")
        have = {p.stem for p in (R / "A" / c).glob("*__*.json")}
        missing = [i for i in order[c][1] if i.replace("/", "__", 1) not in have]
        if missing and f != "DONE" and rec["attempt"] < MAX_ATTEMPTS:
            rec["retry"] = True
        if have:
            v = subprocess.run([PY, "-I", str(P4 / "tools/verify_a.py"), str(R / "A" / c)],
                               capture_output=True, text=True, cwd="/tmp")
            rec["verified"] = v.returncode == 0
            log(f"{c} verified rc={v.returncode}: " + ", ".join(
                f"{k} {v.stdout.count(chr(9) + k + chr(9))}" for k in ("PASS", "PARTIAL", "REJECT")))
    for agent in AGENTS:
        busy = [c for c, r in st["chunks"].items() if r["agent"] == agent and "finished" not in r]
        if busy:
            continue
        retry = [c for c, r in st["chunks"].items() if r["agent"] == agent and r.pop("retry", False)]
        if retry:
            launch(retry[0], agent, order[retry[0]][1], st["chunks"][retry[0]]["attempt"] + 1, st,
                   notes.get(retry[0], "")); continue
        nxt = [c for c, (a, _) in sorted(order.items()) if a == agent and c not in st["chunks"]]
        if nxt:
            launch(nxt[0], agent, order[nxt[0]][1], 1, st, notes.get(nxt[0], ""))
    # Integration requests to @claude.
    pend = st.get("integration")
    if pend and not activation_ended(pend["msg"]):
        pass
    else:
        ready = [c for c, r in st["chunks"].items() if r.get("verified") and not (R / "A" / c / "INTEGRATED").exists()
                 and not (r.get("retry") or ("finished" not in r))]
        all_done = all(c in st["chunks"] and "finished" in st["chunks"][c] for c in order)
        if ready and (len(ready) >= INTEGRATE_BATCH or all_done):
            body = (f"Phase 4 block A integration for Nill's request cf1ebced (you are the manager's follow-up "
                    f"session). Chunks verified and waiting: {' '.join(sorted(ready))}. Read "
                    f"{P4}/INTEGRATE.md and follow it: review each chunk's VERIFY.tsv and verified files, "
                    f"then integrate with tools/integrate_a.py, mark each chunk INTEGRATED and post the milestone "
                    f"metrics table. Authorship Niall Miller <niall.j.miller@gmail.com>, no Claude trailers; server "
                    f"branch only; no deleting; token-efficient. (Posted automatically by phase4/tools/dispatch.py.)")
            msg = send("claude", "Phase 4: integrate verified research chunks", body)
            st["integration"] = {"msg": msg, "chunks": sorted(ready), "posted": now}
    # Block B/C/E worker reports: one review request to @claude once every listed worker has exited.
    pids = P4 / "workers/pids.txt"
    if pids.exists() and not st.get("bce_review"):
        ws = [l.split() for l in pids.read_text().splitlines() if l.strip()]
        if ws and not any(alive(int(pid)) for _, pid in ws):
            body = (f"Phase 4 blocks B, C and E for Nill's request cf1ebced (you are the manager's follow-up session). "
                    f"The Claude workers have finished. Read {P4}/INTEGRATE.md, section 'Blocks B, C, E', and follow it. "
                    f"Authorship Niall Miller <niall.j.miller@gmail.com>, no Claude trailers; server branch for data, "
                    f"PRs for SED_Tools code, never main; no deleting; one crawler on the shared IP; token-efficient. "
                    f"(Posted automatically by phase4/tools/dispatch.py.)")
            st["bce_review"] = {"msg": send("claude", "Phase 4: review blocks B, C and E", body), "posted": now}
    STATE.write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
