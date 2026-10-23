import pathlib,json,csv,hashlib,math,collections,concurrent.futures,urllib.request
root=pathlib.Path(__file__).resolve().parents[1];out=root/'review'
def numeric(path):
 rows=[]
 for line in path.read_text(errors='replace').splitlines():
  if line.lstrip().startswith('#'):continue
  try:v=[float(x) for x in line.replace(',',' ').split()]
  except ValueError:continue
  if len(v)>=2:rows.append(v)
 return rows
reports={};suggest=[]
for group,file in [('codex','manifest.json'),('space','inventory.json')]:
 p=root/group;d=json.loads((p/file).read_text());checks=[]
 for r in d:
  f=p/r['path'];b=f.read_bytes();a=numeric(f);xs=[v[0] for v in a];ys=[v[1] for v in a];errs=[]
  if hashlib.sha256(b).hexdigest()!=r['sha256']:errs.append('hash')
  keys=['xmin','xmax','ymin','ymax'] if group=='codex' else ['x_min','x_max','response_min','response_max']
  if len(a)!=r['rows']:errs.append('rows')
  for key,val in zip(keys,[min(xs),max(xs),min(ys),max(ys)]):
   if not math.isclose(r[key],val,rel_tol=1e-10,abs_tol=1e-12):errs.append(key)
  if not all(math.isfinite(v) for v in xs+ys):errs.append('nonfinite first two columns')
  checks.append({'path':r['path'],'errors':errs,'column_counts':dict(collections.Counter(map(len,a))),'rows':len(a)})
  if group=='codex' and r['facility']=='NOT':
   name=f.name
   if r['xmin']>10000:unit='Angstrom (inferred from band identity and scale; not explicit in header)';resp='relative fraction (inferred from scale; normalization not certified)'
   elif r['xmin']>=100:unit='nm (explicit in header)';resp='percent (inferred from scale; not explicit in header)' if 'iac_lab' in name else 'fraction (inferred from scale; not explicit in header)'
   else:unit='micron (inferred from band identity and scale)';resp='percent (inferred from scale)'
   suggest.append({'path':r['path'],'wavelength_unit':unit,'response_unit':resp})
 reports[group]={'files':len(d),'unique_hashes':len({r['sha256'] for r in d}),'facilities':dict(collections.Counter(r['facility'] for r in d)),'checks':checks}
 if group=='space':
  downloads=json.loads((p/'downloads.json').read_text());reports[group]['raw_hash_checks']=[{'path':r['path'],'ok':hashlib.sha256((p/r['path']).read_bytes()).hexdigest()==r['sha256']} for r in downloads if r.get('ok') and 'sha256' in r]
(out/'independent-audit.json').write_text(json.dumps(reports,indent=2)+'\n');(out/'not-unit-suggestions.json').write_text(json.dumps(suggest,indent=2)+'\n')
print({k:{'files':v['files'],'facilities':v['facilities'],'unique_hashes':v['unique_hashes'],'errors':[r for r in v['checks'] if r['errors']]} for k,v in reports.items()})
# Four ground and one original source per space facility: re-fetch exact raw URL, compare bytes.
samples=[]
d=json.loads((root/'codex/manifest.json').read_text())
for fac in sorted({r['facility'] for r in d}):
 r=next(r for r in d if r['facility']==fac);samples.append((fac,r['source_url'],r['sha256']))
d=json.loads((root/'space/downloads.json').read_text())
for fac in sorted({r['facility'] for r in d if r.get('ok')}):
 r=next(r for r in d if r['facility']==fac and r.get('ok') and 'sha256' in r);samples.append((fac,r['url'],r['sha256']))
def fetch(t):
 fac,url,sha=t
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as h:b=h.read();status=h.status
  return {'facility':fac,'url':url,'http':status,'sha256_equal':hashlib.sha256(b).hexdigest()==sha,'bytes':len(b)}
 except Exception as e:return {'facility':fac,'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:results=list(ex.map(fetch,samples))
(out/'source-refetch.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
