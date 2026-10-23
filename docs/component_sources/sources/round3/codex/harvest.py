import requests,bs4,pathlib,urllib.parse,hashlib,csv,json,math,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parent
sources=[('NOT','NOTCam','https://www.not.iac.es/instruments/notcam/filters/'),('LBT','LBC','https://scienceops.lbto.org/lbc/lbc-instrument-characteristics/filters/'),('Liverpool','IOO','https://telescope.livjm.ac.uk/TelInst/Inst/IOO/')]
rows=[];errors=[]
def get(item):
 facility,inst,page,url,context=item
 try:
  r=requests.get(url,timeout=25);r.raise_for_status()
  name=urllib.parse.urlparse(url).path.split('/')[-1];p=ROOT/'evidence'/facility/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
  vals=[]
  for line in r.text.splitlines():
   try:
    a=line.strip().split();x,y=float(a[0]),float(a[1])
    if math.isfinite(x) and math.isfinite(y):vals.append((x,y))
   except (ValueError,IndexError):pass
  if len(vals)<3:raise ValueError('not numeric curve')
  return dict(facility=facility,instrument=inst,page_url=page,source_url=url,resolved_url=r.url,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(r.content).hexdigest(),rows=len(vals),xmin=min(x for x,y in vals),xmax=max(x for x,y in vals),ymin=min(y for x,y in vals),ymax=max(y for x,y in vals),context=context,role='filter transmission; no separate optics/QE/atmosphere certified')
 except Exception as e:return dict(error=type(e).__name__,facility=facility,url=url)
items=[]
for facility,inst,page in sources:
 r=requests.get(page,timeout=25);r.raise_for_status();p=ROOT/'evidence'/facility/'source.html';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
 s=bs4.BeautifulSoup(r.text,'html.parser');seen=set()
 for a in s.find_all('a',href=True):
  h=a['href'];url=urllib.parse.urljoin(page,h).replace('http://scienceops.lbto.org','https://scienceops.lbto.org')
  if not h.lower().endswith(('.dat','.txt')):continue
  if facility=='LBT' and 'LogOfLBCTwilight' in h:continue
  if url in seen:continue
  seen.add(url);tr=a.find_parent('tr');ctx=tr.get_text(' ',strip=True) if tr else a.get_text(' ',strip=True)
  items.append((facility,inst,page,url,ctx))
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 for row in ex.map(get,items):
  (errors if 'error' in row else rows).append(row)
(ROOT/'manifest.json').write_text(json.dumps(rows,indent=2));(ROOT/'errors.json').write_text(json.dumps(errors,indent=2))
print('Saved',len(rows),'curves',len(errors),'errors',dict((f,sum(x['facility']==f for x in rows)) for f,i,p in sources))
