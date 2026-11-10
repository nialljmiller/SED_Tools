"""Test functions each declutter commit added or changed (277f358..origin/declutter), by commit."""
import ast, re, subprocess, sys, json
repo = sys.argv[1]
def git(*a): return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout
out = {}
for line in git("log", "--no-merges", "--format=%h %s", "277f358..origin/declutter", "--", "tests").splitlines():
    sha, subject = line.split(" ", 1)
    diff = git("show", "--format=", "-U0", sha, "--", "tests")
    touched, path = {}, None
    for l in diff.splitlines():
        if l.startswith("+++ "):
            path = l[6:] if l != "+++ /dev/null" else None
        elif l.startswith("@@") and path and path.startswith("tests/test_"):
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))?", l)
            start, n = int(m.group(1)), int(m.group(2) or 1)
            touched.setdefault(path, set()).update(range(start, start + max(n, 1)))
    funcs = []
    for path, lines in touched.items():
        try:
            src = git("show", f"{sha}:{path}")
        except subprocess.CalledProcessError:
            continue
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                if lines & set(range(first, node.end_lineno + 1)):
                    funcs.append(f"{path}::{node.name}")
    out[sha] = {"subject": subject, "tests": sorted(funcs)}
json.dump(out, sys.stdout, indent=1)
