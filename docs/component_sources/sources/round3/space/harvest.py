from pathlib import Path
import subprocess,json,hashlib,concurrent.futures,datetime
P=Path(__file__).parent
jobs=[]
def add(fac,name,url,basis='original',role='combined_response'):
 jobs.append(dict(facility=fac,path='raw/'+fac+'/'+name,url=url,source_basis=basis,role=role))
for f in ['n2','n3','n4','s7','s9w','s11','l15','l18w','l24']:
 add('AKARI',f+'.fad','https://www.ir.isas.jaxa.jp/AKARI/Observation/support/IRC/RSRF/'+f+'.fad')
add('AKARI','index.html','https://www.ir.isas.jaxa.jp/AKARI/Observation/support/IRC/RSRF/',role='documentation')
add('COBE','dirbe.asc','https://lambda.gsfc.nasa.gov/data/cobe/dirbe/ancil/spec_resp/DIRBE_SYSTEM_SPECTRAL_RESPONSE_TABLE.ASC')
add('COBE','description.html','https://lambda.gsfc.nasa.gov/product/cobe/c_spectral_res.html',role='documentation')
add('IRAS','tabC5.html','https://lambda.gsfc.nasa.gov/product/iras/docs/exp.sup/ch2/tabC5.html',role='filter_detector_system_table')
add('IRAS','tabC5-ipac.html','https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/tabC5.html',role='filter_detector_system_table')
add('WMAP','wmap_bandpass_v5.tar.gz','https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/response/wmap_bandpass_v5.tar.gz')
add('WMAP','description.html','https://lambda.gsfc.nasa.gov/product/wmap/current/bandpass_info.html',role='documentation')
for f in ['LFI_RIMO_R2.50.fits','HFI_RIMO_R2.00.fits']:
 add('PLANCK',f,'https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/'+f)
for f in 'acde':add('MSX','msx_rsr_'+f+'.tbl','https://irsa.ipac.caltech.edu/data/MSX/docs/rsr/msx_rsr_'+f+'.tbl')
add('MSX','description.html','https://irsa.ipac.caltech.edu/data/MSX/imageDescriptions.htm',role='documentation')
add('CHEOPS','response_functions.fits','https://raw.githubusercontent.com/pmaxted/pycheops/master/pycheops/data/response_functions/response_functions.fits',basis='mission_team_software_multi_mission')
add('CHEOPS','instrument.py','https://raw.githubusercontent.com/pmaxted/pycheops/master/pycheops/instrument.py',basis='mission_team_software',role='documentation')
add('XMM','OM_response_081008.txt','https://heasarc.gsfc.nasa.gov/FTP/xmm/data/responses/OM/OM_response_081008.txt')
for f in ['b','u','v','uvm2','uvw1','uvw2']:
 add('XMM','om_effarea_'+f+'_v2.0.rsp','https://heasarc.gsfc.nasa.gov/FTP/xmm/data/responses/OM/om_effarea_'+f+'_v2.0.rsp')
def get(j):
 p=P/j['path'];p.parent.mkdir(parents=True,exist_ok=True)
 r=subprocess.run(['curl','-fLsS','--max-time','60',j['url'],'-o',str(p)],capture_output=True)
 j.update(ok=r.returncode==0,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 if j['ok']: j.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 else:j['error']=r.stderr.decode()[-400:]
 return j
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex: out=list(ex.map(get,jobs))
(P/'downloads.json').write_text(json.dumps(out,indent=2))
print('files',len(out),'ok',sum(x['ok'] for x in out),'bytes',sum(x.get('bytes',0) for x in out));print([(x['path'],x.get('error')) for x in out if not x['ok']])
