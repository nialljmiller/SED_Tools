
These files are response (QE) functions for the USNO 40-inch/SITe 1024^2
UV-coated CCD as best we understand at this point (011218).  The z
functions are quite uncertain, as we have no measurements of the USNO
CCD at all; though the response through the visible and even into the uv
is quite constant from device to device, the far IR response is quite
variable.  The QEs have been corrected for IR scattering and are the
appropriate responses for point sources, NOT extended objects. These 
corrections are probably as uncertain as the underlying response is.

The system as defined is NOT an AB system, quite, because the magnitudes
of the standards as defined in Fukugita et al are extrapolated to vacuum,
and agree reasonably well with the response functions here which have been
modified slightly for temperature effects. The USNO system, however, is
defined as the natural system of the USNO telescope/CCD combination, which
has a median airmass of about 1.3. The standards SHOULD HAVE been defined
with slightly different zero points: +17 should have had magnitudes
defined as

10.51 9.64 9.35 9.25 9.22

instead of the numbers

10.56 9.64 9.35 9.25 9.23

Which define the zeropoints of the system.

Likewise +26 2606 should have been DEFINED as

10.73 9.89 9.60 9.51 9.48;

the OBSERVED numbers (+17 defines the system alone, which is not a good idea)
are

10.76 9.89 9.60 9.50 9.49

as were used. Thus to get to the USNO AB system, one must add

-.04 0.0 0.0 0.0 -0.01

to the USNO magnitudes as observed.


The calculations for the colors from the energy distributions with the
response functions here are 


USNO 1.3 airmasses (V=0)

176 +17 4708       0.00   1.07  0.20 -0.09 -0.19 -0.22   0.88  0.29  0.10  0.02
177 +26 2606       0.00   1.04  0.20 -0.09 -0.18 -0.21   0.84  0.29  0.09  0.03
178 hd19445        0.00   1.02  0.21 -0.10 -0.20 -0.24   0.81  0.31  0.10  0.03
179 hd84937        0.00   0.98  0.16 -0.08 -0.14 -0.15   0.82  0.24  0.07  0.01


USNO 0 airmasses (V=0)

176 +17 4708       0.00   1.11  0.21 -0.09 -0.19 -0.22   0.91  0.30  0.10  0.03
177 +26 2606       0.00   1.08  0.21 -0.09 -0.18 -0.21   0.87  0.30  0.09  0.03
178 hd19445        0.00   1.06  0.22 -0.10 -0.20 -0.24   0.83  0.32  0.10  0.04
179 hd84937        0.00   1.02  0.17 -0.08 -0.14 -0.15   0.85  0.25  0.07  0.01

Observations (The +17 mags are DEFINED)

+17 4708  10.560, 9.640  9.350  9.250  9.230 
+26 2606  10.761  9.891  9.604  9.503  9.486 

V-based mags OBS

+17 4708  1.12  0.20 -0.09 -0.19  -0.21  (9.44)
+26 2606  1.07  0.20 -0.09 -0.19  -0.20  (9.69)

from AB calculations, add

      0.04  0.0   0.0   0.0    0.01

To get to observed values, or subtract to go from obs -> AB


jeg011218


