import requests,json,pathlib,hashlib,urllib.parse,concurrent.futures
p=pathlib.Path(__file__).parent;j=json.loads((p/'LICK/tree.json').read_text());(p/'LICK').mkdir(exist_ok=True)
a=[x['path'] for x in j['tree'] if 'KAIT_2018/' in x['path'] and (x['path'].endswith('.txt') or x['path'].split('/')[-1] in ['README','AAA_README']) and not x['path'].split('/')[-1].startswith(('#','.'))]
def get(path):
 u='https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/'+j['sha']+'/'+urllib.parse.quote(path);r=requests.get(u,timeout=30);r.raise_for_status();f=p/'LICK'/path.split('/')[-1];f.write_bytes(r.content)
 if f.suffix!='.txt':return
 rows=[]
 for line in r.text.splitlines():
  try:
   row=[float(x) for x in line.split()]
   if len(row)>=2:rows.append(row)
  except ValueError:pass
 return dict(facility='LICK',url=u,path=str(f.relative_to(p)),sha256=hashlib.sha256(r.content).hexdigest(),numeric_rows=len(rows),columns=sorted(set(map(len,rows))),first_rows=rows[:2],last_rows=rows[-2:])
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as e:r=[x for x in e.map(get,a) if x]
d=json.loads((p/'inventory.json').read_text());d=[x for x in d if x['facility']!='LICK']+r;(p/'inventory.json').write_text(json.dumps(d,indent=2));(p/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in d));(p/'LICK/commit.txt').write_text(j['sha']+'\n');print(len(r),len(d))
