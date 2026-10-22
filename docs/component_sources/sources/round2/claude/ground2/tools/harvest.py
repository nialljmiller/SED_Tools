import sys, time, re, os, hashlib, json, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
UA = {'User-Agent': 'curl/8.9.1'}
def get(url, binary=False):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40)
        d = r.read()
        return d if binary else d.decode('utf8', 'replace')
    except Exception as e:
        return None
DATA = re.compile(r'(?:\.ascii|\.dat|\.txt|\.fits|\.xls|\.xlsx|\.csv|\.zip|\.tgz|\.gz|\.asc|\.tbl)$', re.I)
manifest = []
def save(url, sub):
    d = get(url, True)
    if d is None or d[:200].lstrip().lower().startswith((b'<!doctype', b'<html')):
        manifest.append(dict(url=url, ok=False)); return
    p = os.path.join(out, sub, re.sub(r'[^A-Za-z0-9._+-]', '_', os.path.basename(urllib.parse.urlparse(url).path)))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').write(d)
    manifest.append(dict(url=url, ok=True, path=p, bytes=len(d), sha256=hashlib.sha256(d).hexdigest()))
jobs = []
mode = sys.argv[2]
if mode == 'eso':
    for u in sys.argv[3:]:
        h = get(u)
        if not h: manifest.append(dict(url=u, ok=False)); continue
        inst = u.split('/instruments/')[1].split('/')[0]
        for l in set(re.findall(r'href="([^"#]+)"', h)):
            full = urllib.parse.urljoin(u, l)
            if DATA.search(urllib.parse.urlparse(full).path) and 'eso.org' in full:
                jobs.append((full.replace('www.ls.eso.org','www.eso.org'), 'eso_'+inst))
elif mode == 'pages':
    for u in sys.argv[3:]:
        h = get(u)
        if not h: manifest.append(dict(url=u, ok=False)); continue
        tag = re.sub(r'[^A-Za-z0-9]+','_',urllib.parse.urlparse(u).netloc+urllib.parse.urlparse(u).path)[:50]
        for l in set(re.findall(r'href="([^"#?]+)"', h)):
            full = urllib.parse.urljoin(u, l)
            if DATA.search(urllib.parse.urlparse(full).path): jobs.append((full, 'pg_'+tag))
elif mode == 'noirlab':
    idx = sys.argv[3:]
    ids = set()
    for u in idx:
        h = get(u) or ''; time.sleep(1.5)
        ids |= set(re.findall(r'href="(?:https://noirlab.edu)?/science/filters/([A-Za-z0-9_.-]+)"', h))
    def one(i):
        h = get('https://noirlab.edu/science/filters/'+i) or ''
        m = re.search(r'href="/science/documents/([^"]+)"', h.split('<h4>Data</h4>')[-1]) if '<h4>Data</h4>' in h else None
        if not m: return None
        d = get('https://noirlab.edu/science/documents/'+m.group(1)) or ''
        f = re.search(r'href="(/science/sites/default/files/media/archives/documents/[^"]+\.(?:txt|dat|csv|xls))"', d)
        return ('https://noirlab.edu'+f.group(1), 'noirlab_filters') if f else None
    jobs = []
    for i in sorted(ids):
        j = one(i); time.sleep(1.5)
        if j: jobs.append(j)
    print('noirlab ids', len(ids), 'with data', len(jobs))
elif mode == 'urls':
    jobs = [(u, 'misc') for u in sys.argv[3:]]
if mode == 'noirlab':
    for j in jobs: save(*j); time.sleep(1.5)
else:
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: save(*j), jobs))
ok = [m for m in manifest if m['ok']]
print(mode, 'saved', len(ok), 'failed', len(manifest)-len(ok))
json.dump(manifest, open(os.path.join(out, 'manifest_%s.json' % mode), 'w'), indent=1)
