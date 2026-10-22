from pathlib import Path
import csv,json,hashlib,gzip,re,collections,tarfile,zipfile,datetime
import numpy as np
from astropy.io import fits
P=Path('/home/nill/agent-share/projects/sed-tools-components/round2'); E=P/'claude/ground2/evidence'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def nums(p):
 s=(gzip.open(p,'rt',errors='replace').read() if p.suffix=='.gz' else p.read_text(errors='replace'))
 a=[]
 for l in s.splitlines():
  try:
   t=re.split(r'[\s,;]+',l.strip()); a.append([float(t[0]),float(t[1])])
  except (ValueError,IndexError): pass
 return np.array(a)
out={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All current owner CSV/JSONL structures, full ground2 inventory/hash checks, selected semantic and archive/numeric checks. Not every optical path certified.'}
out['owner_files']={}
for owner in ['claude','grok','gemini']:
 for f in (P/owner).glob('*'):
  if f.suffix not in ['.csv','.jsonl']:continue
  try:
   r=list(csv.DictReader(f.open())) if f.suffix=='.csv' else [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
   out['owner_files'][str(f.relative_to(P))]={'rows':len(r),'sha256':sha(f),'parse':'ok'}
  except Exception as ex: out['owner_files'][str(f.relative_to(P))]={'error':str(ex)}
r=list(csv.DictReader((E.parent/'inventory.csv').open())); hashes=collections.defaultdict(list); bad=[]; mismatch=[]
for x in r:
 p=E/x['path']
 if not p.exists():bad.append(x['path']);continue
 hashes[sha(p)].append(x['path']);a=nums(p)
 if len(a)!=int(x['rows']) or not np.isclose(a[:,0].min(),float(x['x_min'])) or not np.isclose(a[:,1].max(),float(x['y_max'])):mismatch.append(x['path'])
out['inventory']={'rows':len(r),'missing_source_urls':sum(not x['source_url'] for x in r),'missing_files':bad,'recomputed_stat_mismatches':mismatch,'unique_byte_contents':len(hashes),'duplicate_groups':[v for v in hashes.values() if len(v)>1],'groups':dict(collections.Counter('/'.join(x['path'].split('/')[:2]) for x in r))}
out['hash_manifests']={}
for root in [P/'claude/ground/evidence',E]:
 good=0;fails=[]
 for l in (root/'SHA256SUMS').read_text().splitlines():
  h,n=l.split(None,1);p=root/n.strip().lstrip('*')
  if p.exists() and sha(p)==h:good+=1
  else:fails.append(n)
 out['hash_manifests'][str(root.relative_to(P))]={'pass':good,'fail':fails}
out['vista']={}
for p in (E/'x/vircam').rglob('*.gz'):
 if not ('.dat.' in p.name or '.txt.' in p.name or '.tab.' in p.name):continue
 a=nums(p);out['vista'][p.name]={'rows':len(a),'xmin':float(a[:,0].min()),'xmax':float(a[:,0].max()),'ymax':float(a[:,1].max()),'header':gzip.open(p,'rt',errors='replace').read()[:250]}
out['archives']={}
for p in (P/'claude/ground/evidence').iterdir():
 if p.suffix=='.tgz':
  with tarfile.open(p) as t: out['archives'][p.name]={'members':len(t.getnames()),'fits_members':len([n for n in t.getnames() if n.endswith('.fits')])}
 if p.suffix=='.zip':
  with zipfile.ZipFile(p) as z:out['archives'][p.name]={'members':len(z.namelist())}
p=P/'claude/ground/evidence/scidoc1884_decam.fits'
with fits.open(p) as h:
 d=h[1].data;out['decam']={n:{'min':float(d[n].min()),'max':float(d[n].max())} for n in d.names};out['decam']['quotient_peaks']={n:float((d[n]/d['atm']).max()) for n in ['g','r','i','z','Y']}
base=E/'lsst_throughputs/baseline';out['lsst_product_errors']={}
for b in 'ugrizy':
 total=nums(base/f'total_{b}.dat');hw=nums(base/f'hardware_{b}.dat');atm=nums(base/'atmos_std.dat');prod=hw[:,1]*np.interp(hw[:,0],atm[:,0],atm[:,1]);out['lsst_product_errors'][b]=float(np.max(np.abs(total[:,1]-prod)))
(P/'codex/current_work_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['vista','inventory']},indent=2));print('inventory', {k:v for k,v in out['inventory'].items() if k not in ['duplicate_groups','groups']})
