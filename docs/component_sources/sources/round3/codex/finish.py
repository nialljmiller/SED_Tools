import pathlib,json,requests,hashlib,math,urllib.parse
P=pathlib.Path(__file__).parent
rows=json.loads((P/'manifest.json').read_text());errors=json.loads((P/'errors.json').read_text())
def parse(text):
 vals=[]
 for l in text.splitlines():
  try:
   s=l.replace(',',' ').split();x,y=float(s[0]),float(s[1]);
   if math.isfinite(x) and math.isfinite(y):vals.append((x,y))
  except (ValueError,IndexError):pass
 return vals
def record(f,i,url,page,path,context):
 b=path.read_bytes();v=parse(b.decode(errors='replace'))
 assert len(v)>2
 return dict(facility=f,instrument=i,page_url=page,source_url=url,path=str(path.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),rows=len(v),xmin=min(x for x,y in v),xmax=max(x for x,y in v),ymin=min(y for x,y in v),ymax=max(y for x,y in v),context=context,role='filter transmission')
remaining=[]
for e in errors:
 p=P/'evidence'/e['facility']/e['url'].split('/')[-1]
 try: rows.append(record(e['facility'],'LBC',e['url'],'https://scienceops.lbto.org/lbc/lbc-instrument-characteristics/filters/',p,'2026 posted scan, ASCII comma separated; Angstrom, percent; full vs bandpass variants retained'))
 except Exception:remaining.append(e)
for band in ['Us','B','V','r','i']:
 url=f'https://sncosmo.github.io/data/bandpasses/keplercam/{band}_Keplercam.txt';r=requests.get(url,timeout=15);r.raise_for_status();p=P/'evidence'/'FLWO'/url.split('/')[-1];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
 row=record('FLWO','KeplerCam',url,'https://github.com/sncosmo/sncosmo/blob/master/sncosmo/builtins.py',p,'Secondary sncosmo copy of JLA transmissions; official old CfA URL redirects, JLA origin 403, SVO refused. Wavelength Angstrom, relative response; system component inclusion unresolved.');row['role']='combined/relative passband, decomposition unresolved';rows.append(row)
u='https://ndownloader.figshare.com/files/7599991';r=requests.get(u,timeout=20);r.raise_for_status();p=P/'evidence'/'Liverpool'/'ioo_filter_throughput_and_qe.txt';p.write_bytes(r.content)
row=record('Liverpool','IOO',u,'https://figshare.com/articles/4659466',p,'Official instrument-page linked author dataset; MULTICOLUMN filter throughputs and CCD QE. Inspect all columns; first-pair stats only.');row['role']='multicolumn filters and separate CCD QE';rows.append(row)
for row in rows:
 f=row['facility'];row['wavelength_unit']=dict(NOT='micron (warm IAC lab files nm; see header)',LBT='Angstrom',Liverpool='nm',FLWO='Angstrom')[f]
 row['response_unit']='percent' if f in ['NOT','LBT','Liverpool'] else 'relative fraction'
 if 'not' in pathlib.Path(row['path']).name and '_iac_lab' in row['path']:row['wavelength_unit']='nm (header confirms)'
 row['source_class']='secondary research-library copy' if f=='FLWO' else 'facility primary or official linked author dataset'
(P/'manifest.json').write_text(json.dumps(rows,indent=2));(P/'errors.json').write_text(json.dumps(remaining,indent=2))
print('curves/tables',len(rows),'errors',len(remaining),'unique hashes',len(set(x['sha256'] for x in rows)))
print({f:sum(x['facility']==f for x in rows) for f in sorted(set(x['facility'] for x in rows))})

# Independently reviewed per-file NOT units override generic generation defaults.
suggestions=P.parent/'review'/'not-unit-suggestions.json'
if suggestions.exists():
 by_path={r['path']:r for r in json.loads(suggestions.read_text())}
 for r in rows:
  if r['path'] in by_path:r.update(by_path[r['path']])
 for r in rows:
  nonfinite=0
  for line in (P/r['path']).read_text(errors='replace').splitlines():
   try:
    v=line.replace(',',' ').split();x,y=float(v[0]),float(v[1])
    if not (math.isfinite(x) and math.isfinite(y)):nonfinite+=1
   except (ValueError,IndexError):pass
  r['finite_rows']=r['rows'];r['nonfinite_first_pair_rows']=nonfinite
 (P/'manifest.json').write_text(json.dumps(rows,indent=2))
