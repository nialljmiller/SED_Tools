"""astroquery against the live NJM SVO-compatible service (W1, 2026-10-26)."""
import sys, time, astropy.units as u
from astroquery.svo_fps import SvoFps
SvoFps.SVO_MAIN_URL = sys.argv[1] if len(sys.argv) > 1 else "https://nillmill.ddns.net/sed_tools/svo_fps/fps.php"
print("SvoFps.SVO_MAIN_URL =", SvoFps.SVO_MAIN_URL)
kw = dict(cache=False)
def pause(): time.sleep(1.5)
fl = SvoFps.get_filter_list(facility="2MASS", **kw); pause()
print("get_filter_list(2MASS):", len(fl), list(fl["filterID"]))
fl = SvoFps.get_filter_list(facility="Paranal", instrument="NACO", **kw); pause()
print("get_filter_list(Paranal, NACO):", len(fl))
idx = SvoFps.get_filter_index(16000 * u.AA, 16500 * u.AA, **kw); pause()
print("get_filter_index(16000-16500 A):", len(idx), list(idx["filterID"])[:8])
for fid in ["2MASS/2MASS.H", "Paranal/NACO.H", "WFIRST/WFI.R062", "Herschel/SPIRE.PSW", "Spitzer/MIPS.24mu", "GAIA/GAIA2r.Grp"]:
    t = SvoFps.get_transmission_data(fid, **kw); pause()
    print(f"get_transmission_data({fid}): {len(t)} rows, {t.colnames}, {t['Wavelength'].unit}, "
          f"{t[0]['Wavelength']}..{t[-1]['Wavelength']}")
try:
    SvoFps.get_transmission_data("Nope/Nope.X", **kw)
except IndexError as e:
    print("unknown ID -> IndexError:", e)
