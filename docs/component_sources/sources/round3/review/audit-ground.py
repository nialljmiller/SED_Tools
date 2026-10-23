import json,pathlib,hashlib,collections,math,urllib.request,concurrent.futures
p=pathlib.Path(__file__).resolve().parents[1]/'ground-agent';d=json.loads((p/'inventory.json').read_text());checks=[]
for r in d:
 f=p/r['path'];a=[]
 for line in f.read_text(errors='replace').splitlines():
  try:v=list(map(float,line.replace(',',' ').split()))
  except ValueError:continue
  if len(v)>=2:a.append(v)
 errs=[]
 if hashlib.sha256(f.read_bytes()).hexdigest()!=r['sha256']:errs.append('hash')
 if len(a)!=r['numeric_rows']:errs.append('rows')
 if sorted(set(map(len,a)))!=r['columns']:errs.append('columns')
 checks.append({'path':r['path'],'errors':errs,'numeric_rows':len(a),'nonfinite_values':sum(not math.isfinite(x) for v in a for x in v),'columns':sorted(set(map(len,a)))})
res={'files':len(d),'unique_hashes':len({r['sha256'] for r in d}),'facility_groups':dict(collections.Counter(r['facility'] for r in d)),'checks':checks}
(p.parent/'review/independent-ground-audit.json').write_text(json.dumps(res,indent=2)+'\n');print({k:v for k,v in res.items() if k!='checks'});print([r for r in checks if r['errors']])
samples=[]
for fac in sorted({r['facility'] for r in d}):samples.append(next(r for r in d if r['facility']==fac))
def fetch(r):
 try:
  with urllib.request.urlopen(urllib.request.Request(r['url'],headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as h:b=h.read();status=h.status
  return {'facility':r['facility'],'url':r['url'],'http':status,'sha256_equal':hashlib.sha256(b).hexdigest()==r['sha256']}
 except Exception as e:return {'facility':r['facility'],'url':r['url'],'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:res=list(ex.map(fetch,samples))
(p.parent/'review/ground-source-refetch.json').write_text(json.dumps(res,indent=2)+'\n');print(res)
