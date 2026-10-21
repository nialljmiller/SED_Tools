# Manager checks and corrective handoff

These checks were performed on the Mac independently of the researchers. The
[retrieval log](checks.json) records exact URLs, hashes and numeric results.

- **WISE W1:** the [original per-photon response](https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/figures/RSR-W1.txt)
  contains 398 numeric rows and peaks at 0.8798. Claude's older record calling
  this file normalized to a peak of one is wrong; the newer record is consistent
  with these bytes. Preserve the distinction between the native response and a
  wavelength-weighted, normalized product. W2–W4 were not checked here.
- **MIPS:** the [instrument handbook](https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/6/)
  defines its posted response in electrons per energy unit and describes the
  combined detector and filter response. A blank convention in the numeric
  file does not justify ignoring this documentation. The owner must connect
  the exact file/release to this definition and retain that citation.
- **PACS blue:** inspected the [original FITS calibration product](https://archives.esac.esa.int/hsa/legacy/cal/PACS/user/PCalPhotometer_FilterTransmission_FM_v1.fits),
  with 15,902 blue samples, transmission first and wavelength in micrometres
  second. Interpolating it onto all 405 [SVO blue samples](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Herschel/Pacs.blue)
  after converting angstroms gives maximum absolute difference 5.000013e-7,
  or 9.669054e-7 relative to the SVO peak. This replaces a weak comparison of
  peaks with an actual full sampled-curve comparison. Numerical agreement alone
  does not prove whether detector response is included in the calibration
  product; that requires its definition. Other PACS bands were not checked.

These findings were routed to Claude (owner) and Grok (peer reviewer). Their
corrections still need to be reconciled with the final Codex audit.

## GOES-16 retrieval check

The [NOAA calibration page](https://ncc.nesdis.noaa.gov/GOESR/ABI.php) still
lists the vendor original workbook and its processed GOES-16 release, but a
direct connection to the NCC workbook timed out from both the Mac and Vega on
2026-10-20. This is a current retrieval problem, not evidence that NOAA removed
the data. The [EUMETSAT NWP SAF GOES-16 SRF archive](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_goes_16_abi_srf/rtcoef_goes_16_abi_srf.tar.gz)
was reachable with HTTP 200 and contains 16 channel text files (145,869 bytes,
SHA-256 `0138d296c79e4bd6e7d44ac4008228d2a5161ca551a2d36d6aab3182dc884fe2`).
The inspected channel 14 header identifies CWG processed version 3, dated
2016-03-10, and gives wavenumber in cm⁻¹ and filter response. This archive
provides a usable non-SVO **combined response** fallback while the NOAA vendor
workbook is unreachable; it does not supply the vendor workbook's separate
optical transmission and detector response columns. Other channels were listed
but not individually parsed.
