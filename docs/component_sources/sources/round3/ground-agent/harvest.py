import requests, pathlib, hashlib, json, concurrent.futures, urllib.parse
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).parent
pages={'WHT':'https://www.ing.iac.es/Astronomy/instruments/ingrid_new/ingrid_configuration.html','IRTF':'https://irtfweb.ifa.hawaii.edu/~nsfcam2/Filter_Profiles.html','MMT':'https://lweb.cfa.harvard.edu/~bmcleod/Megacam/Filters/','CAHA':'https://www.caha.es/component/content/article/panic-filters?Itemid=1346&catid=670%3Ainstruments-caha-english','CAHA-BUSCA':'https://www.caha.es/guijarro/BUSCA/caracter.html','CAHA-LAICA':'https://www.caha.es/CAHA/Instruments/LAICA/manual.html','JCMT':'https://www.eaobservatory.org/jcmt/instrumentation/continuum/scuba-2/filters/'}
jobs=[]
for fac,u in pages.items():
 r=requests.get(u,timeout=20); (ROOT/fac).mkdir(exist_ok=True);(ROOT/fac/'source.html').write_bytes(r.content)
 for a in BeautifulSoup(r.text,'html.parser').find_all('a'):
  h=a.get('href',''); name=a.get_text()
  if (fac=='WHT' and h.startswith('filter/') and '.trans.' in h) or (fac=='CAHA' and 'ascii' in a.get_text()) or (fac=='CAHA-BUSCA' and h=='bands.txt') or (fac=='CAHA-LAICA' and (h.startswith('john_') or h.startswith('SDSS_'))) or (fac=='IRTF' and h.endswith('.dat')) or (fac=='MMT' and ('MMT.txt' in h or 'filterthroughput.megacam' in h and not h.endswith('.ps'))) or (fac=='JCMT' and any(x in name for x in ['model850','model450','s8c20140313','s8d20140313'])):
   jobs.append((fac,urllib.parse.urljoin(u,h)))
def get(job):
 fac,u=job;p=ROOT/fac/u.split('/')[-1]
 if p.exists(): content=p.read_bytes(); body=content.decode(errors='replace')
 else:
  r=requests.get(u,timeout=35);r.raise_for_status(); content=r.content;body=r.text;p.write_bytes(content)
 rows=[]
 for line in BeautifulSoup(body,'html.parser').get_text().splitlines():
  try:
   vals=[float(x) for x in line.replace(',',' ').split()]
   if len(vals)>=2: rows.append(vals)
  except ValueError:pass
 return dict(facility=fac,url=u,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(content).hexdigest(),numeric_rows=len(rows),columns=sorted(set(map(len,rows))),first_rows=rows[:2],last_rows=rows[-2:])
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e:
 out=[]
 for job,fut in [(job,e.submit(get,job)) for job in dict.fromkeys(jobs)]:
  try: out.append(fut.result())
  except Exception as exc: print('FAILED',job,str(exc))
(ROOT/'inventory.json').write_text(json.dumps(out,indent=2)); print('FILES',len(out));print({f:sum(x['facility']==f for x in out) for f in pages})
