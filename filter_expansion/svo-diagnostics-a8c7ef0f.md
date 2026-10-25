# SVO FPS reachability investigation

Codex session a8c7ef0f; 2026-10-25, 15:51–15:54 UTC.
Status: investigation complete; diagnostic note claim released.
Scope: read-only diagnosis. No builder, filter data, deployed services or network settings changed. No detached jobs remain. Small focused investigation handled directly.

## Findings

- Local DNS and independent Google DNS-over-HTTPS agree: svo2.cab.inta-csic.es = 193.146.134.6. No AAAA record. Main SVO host svo.cab.inta-csic.es = 193.146.134.30.
- Vega curl probes explicitly bypass proxies (no proxy environment variables were present). Both port 80 and 443 on svo2 time out establishing TCP. curl exit 28; HTTP code 000; time_connect=0; time_appconnect=0; first byte=0. Seven-second connection deadlines expired.
- Same failure for host root, FPS Generic/Johnson.V query, historical /svo/theory/fps path, and VOSA. Python urllib independently reproduces it with eight-second deadlines. Thus it happens before TLS, HTTP, filter lookup or XML parsing, across applications on this host.
- Cirrus, via ssh -F /home/nill/.ssh/config.agent-network cirrus, resolves the same address and reproduces the HTTPS TCP connection timeout (7.003 seconds). Vega and Cirrus share the same household network and are not independent public vantage points.
- External web fetch also reports timeout for the FPS root and representative query. This supports reachability trouble beyond the local clients, but does not prove a worldwide outage.
- Controls from Vega: https://svo.cab.inta-csic.es/ returns HTTP 200 (~1.13 seconds); https://cab.inta-csic.es/ returns HTTP 200 (~4.66 seconds), both at 193.146.134.30. External web fetch of main SVO site also succeeds.
- Live main SVO homepage still links FPS to http://svo2.cab.inta-csic.es/theory/fps/. Official astroquery documentation uses https://svo2.cab.inta-csic.es/theory/fps/fps.php. No evidence the configured canonical endpoint is obsolete.
- Main SVO host does not supply a drop-in replacement: /theory/fps/fps.php and /svo/theory/fps/fps.php return 404. A main-site-linked alternate host toucan.cab.inta-csic.es (193.146.134.10) refuses HTTPS connection for the candidate /svo2/theory/fps path; not a verified mirror.
- UDP traceroute reaches 130.206.215.42 at hop 14 (~125 ms), then no replies through hop 16. Missing traceroute replies cannot locate a fault. TCP traceroute unavailable without additional privileges; not pursued because application probes already isolate the failure stage.

## Conclusion and practical implications

The observed failure is reachability of the svo2 host/web ports, not slow filter processing, detector metadata, TLS certificates, Python permissions, IPv6, local DNS mismatch or a too-short read timeout. A down host/service, remote firewall, or route/drop affecting that address remains possible. Exact remote cause and whether our public address is selectively blocked cannot be established without provider-side evidence or more independent network probes. No HTTP response means there is no observed 403/429 rate-limit response.

Changing the read timeout or repeatedly retrying individual bands is unlikely to resolve TCP establishment failure. Keep existing cached verified data and original-source fallbacks; retry host connectivity before another batch. If this persists, SVO support can investigate with destination IP, time and failure stage. Contact is listed in the official IVOA registry; no message sent.

## Reproduce

```sh
curl --noproxy '*' -sS -o /dev/null --connect-timeout 7 --max-time 12 \
  -w 'code=%{http_code} dns=%{time_namelookup} connect=%{time_connect} tls=%{time_appconnect} first=%{time_starttransfer} total=%{time_total}\n' \
  'https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Johnson.V'
```

Sources: live main site https://svo.cab.inta-csic.es/ ; official astroquery documentation https://astroquery.readthedocs.io/en/stable/_modules/astroquery/svo_fps.html ; official registry https://vao.stsci.edu/directory/getRecord.aspx?id=ivo%3A%2F%2Fsvo.cab%2Ffps ; local builder tools/fxlib.py lines 51–72. Registry active status is registration metadata, not a live availability guarantee.

Budget: provider/account remaining allowance not exposed by tools; no reserve percentage claimed. Kept work bounded to direct diagnostics within the visible context allowance, without delegated/model polling calls.
