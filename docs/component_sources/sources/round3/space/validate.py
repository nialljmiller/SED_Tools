from pathlib import Path
import json,hashlib,csv,io,tarfile,re
import numpy as np
from astropy.io import fits
from astropy.table import Table
from bs4 import BeautifulSoup
P=Path(__file__).parent; D=json.loads((P/'downloads.json').read_text()); R=[]; warnings=[]
def add(fac,key,a,unit,role,source,basis='original',note=''):
 a=np.array(a,float); assert a.ndim==2 and a.shape[1]==2 and len(a)>1
 assert np.isfinite(a).all()
 out=P/'curves'/fac/(key+'.csv');out.parent.mkdir(exist_ok=True,parents=True)
 np.savetxt(out,a,delimiter=',',header='spectral_coordinate,response',comments='')
 R.append(dict(facility=fac,curve=key,path=str(out.relative_to(P)),source_path=source,source_basis=basis,spectral_unit=unit,response_role=role,response_unit=('m^2' if role=='effective_area' else 'unknown' if role=='software_instrument_response' else 'dimensionless'),rows=len(a),x_min=float(a[:,0].min()),x_max=float(a[:,0].max()),response_min=float(a[:,1].min()),response_max=float(a[:,1].max()),strictly_increasing=bool(np.all(np.diff(a[:,0])>0)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),note=note))
for f in (P/'raw/AKARI').glob('*.fad'):
 a=np.loadtxt(f);add('AKARI',f.stem,a,'micron (inferred from values; not in file)','filter_detector_energy_response',str(f.relative_to(P)),note='Relative RSR includes wavelength factor for photon-to-energy conversion per JAXA index. Spectral unit inferred; no individual optics/QE tables in packet.')
f=P/'raw/COBE/dirbe.asc';a=np.loadtxt(io.StringIO('\n'.join(x for x in f.read_text().splitlines() if re.match(r'^\s*\d',x))))
for i in range(1,11):add('COBE','DIRBE_'+str(i),a[:,[0,i]],'micron','normalized_filter_detector_response',str(f.relative_to(P)),note='Dimensionless normalized system RSR; Pass 3B Nov 1996; shared grid.')
f=P/'raw/IRAS/tabC5-ipac.html';s=BeautifulSoup(f.read_text(),'html.parser');band=None;data={}
for tr in s.find('table').find_all('tr'):
 vals=[x.get_text(' ',strip=True) for x in tr.find_all(['td','th'],recursive=False)]
 txt=' '.join(vals);m=re.search(r'(12|25|60|100)\s*µm Band',txt)
 if m:band=m.group(1);data[band]=[];continue
 if len(vals)==4 and band:
  try:data[band].append([float(x) for x in vals])
  except ValueError:pass
for b,rows in data.items():
 for i,role in enumerate(['filter_transmission','relative_detector_response','relative_system_response'],1):
  note='Dimensionless published table, preserved as printed. 25 micron row 29.0 has detector=0.19 inconsistent with system=0.877; uncorrected source typo suspected, component reconstruction unsafe.' if b=='25' else 'Dimensionless published table. No telescope optics separated.'
  add('IRAS',b+'_'+role,np.array(rows)[:,[0,i]],'micron',role,str(f.relative_to(P)),note=note)
f=P/'raw/WMAP/wmap_bandpass_v5.tar.gz'
with tarfile.open(f) as t:
 for m in t.getmembers():
  if not m.isfile() or not m.name.endswith('.cbp'):continue
  a=np.loadtxt(t.extractfile(m))
  for i in [1,2]:add('WMAP',Path(m.name).stem+'_det'+str(i),a[:,[0,i]],'GHz','radiometer_detector_frequency_response',str(f.relative_to(P)),note='Dimensionless linear Rayleigh-Jeans response, peak normalized; detector curves not optical-element QE. Mission frequency drift caveat in description.')
for f in (P/'raw/PLANCK').glob('*.fits'):
 with fits.open(f) as h:
  for x in h:
   if not x.name.startswith('BANDPASS'):continue
   a=np.column_stack((x.data['WAVENUMBER'],x.data['TRANSMISSION']))
   add('PLANCK',f.stem+'_'+x.name,a,x.columns['WAVENUMBER'].unit,'detector_or_band_spectral_response',str(f.relative_to(P)),note='Dimensionless transmission per FITS; uncertainties/flags retained in raw file. HFI coordinates are inverse cm; LFI GHz. Native ordering retained; do not assume common normalization across LFI/HFI.')
for f in (P/'raw/MSX').glob('*.tbl'):
 t=Table.read(f,format='ascii.ipac');add('MSX',f.stem,np.column_stack((t['wavelength'],t['response'])),'micron','relative_system_response',str(f.relative_to(P)),note='Dimensionless RSR; whole system combined, PSC v1.2.')
f=P/'raw/CHEOPS/response_functions.fits'
with fits.open(f) as h:
 for fac,col in [('CHEOPS','CHEOPS'),('MOST','MOST'),('Corot','CoRoT'),('PLATO','PLATO')]:
  add(fac,col,np.column_stack((h[1].data['Wavelength'],h[1].data[col])),'nm','software_instrument_response',str(f.relative_to(P)),basis='mission_team_software' if fac=='CHEOPS' else 'secondary_software_host',note='171-row common software grid. Response unit/convention and calibration release not stated in FITS; original per-mission provenance unresolved. PLATO is planned and excluded from flown-facility count.' if fac=='PLATO' else '171-row common software grid. Response unit/convention and calibration release not stated in FITS; original per-mission provenance unresolved. Combined software response, not separate components.')
f=P/'raw/XMM/OM_response_081008.txt';a=np.loadtxt(f)
for i,key in enumerate(['UVW2','UVM2','UVW1','U','B','V'],1):add('XMM','OM_'+key,a[:,[0,i]]/np.array([1,1000]),'nm','effective_area',str(f.relative_to(P)),note='Converted response from source value = m^2 * 1000 to m^2 by dividing 1000; spectral coordinate unchanged. ASCII calibration date 081008; separate components not supplied.')
fitschecks=[]
for f in (P/'raw/XMM').glob('*.rsp'):
 with fits.open(f) as h:
  h.verify('exception');fitschecks.append(dict(path=str(f.relative_to(P)),extensions=[dict(name=x.name,rows=len(x.data) if x.data is not None else 0) for x in h]))
for x in D:
 assert x['ok'] and hashlib.sha256((P/x['path']).read_bytes()).hexdigest()==x['sha256']
(P/'inventory.json').write_text(json.dumps(R,indent=2))
with (P/'inventory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(R[0]));w.writeheader();w.writerows(R)
summary={k:sum(x['facility']==k for x in R) for k in sorted(set(x['facility'] for x in R))}
checks=dict(downloads=len(D),download_hashes_passed=len(D),curves=len(R),by_facility=summary,flown_facilities=len(summary)-1,planned_facilities=['PLATO'],non_increasing=[x['path'] for x in R if not x['strictly_increasing']],negative_responses=[x['path'] for x in R if x['response_min']<0],fits_checks=fitschecks)
(P/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2)[:3000])
