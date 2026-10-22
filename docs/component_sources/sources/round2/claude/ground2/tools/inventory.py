import sys, os, re, csv, json, glob, gzip
root = sys.argv[1]
urls = {}
for m in glob.glob(os.path.join(root, '**/manifest_*.json'), recursive=True):
    for e in json.load(open(m)):
        if e.get('ok'): urls[os.path.abspath(e['path'].replace('evidence/', root.rstrip('/')+'/', 1))] = e['url']
rows = []
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.startswith('manifest_') or f.endswith(('.png','.gif','.jpg','.ps','.pdf','.eps','.zip','.xls','.sm','.smhist','.tar.gz','.gz')) and not f.endswith(('.dat.gz','.txt.gz','.tab.gz')): continue
        p = os.path.join(dp, f)
        op = gzip.open if p.endswith('.gz') else open
        try: txt = op(p, 'rt', errors='replace').read()
        except Exception: continue
        hdr = ''; wl = []; y = []
        for ln in txt.splitlines():
            s = ln.strip()
            if not s: continue
            if s[0] in '#\\' or not re.match(r'^[-+0-9.]', s):
                if not wl and len(hdr) < 120: hdr += s + ' | '
                continue
            t = re.split(r'[\s,;]+', s)
            try: a = float(t[0]); b = float(t[1])
            except Exception: continue
            wl.append(a); y.append(b)
        if len(wl) < 5: continue
        rows.append([os.path.relpath(p, root), urls.get(os.path.abspath(p), ''), len(wl), min(wl), max(wl), max(y), min(y), hdr[:110]])
rows.sort()
w = csv.writer(open(sys.argv[2], 'w', newline=''))
w.writerow(['path','source_url','rows','x_min','x_max','y_max','y_min','header'])
w.writerows(rows)
print(len(rows), 'numeric curves')
