# Round 3 — 10 ground facilities and 10 flown space missions

Request 9253fd2ba9540be0d9191916; Codex sole bridge owner, 2026-10-23 UTC. User explicitly requested teams and reciprocal checks; work used isolated delegated researchers, without claiming to be other agents' interactive sessions. Previous researcher files remain intact. This is research/download coverage; no SED application integration.

Newness is relative to all original and round2 reviewed facility records (56 previous facility/group keys, 165 instrument pairs), not the discovery-only SVO catalog. Physical aliases and umbrella groups are excluded from the new-facility count.

## Ground

NOT, LBT, Liverpool, FLWO, IRTF, MMT, CAHA, JCMT, WHT, Lick. KAIT/Nickel count together as Lick; multiple CAHA instruments count once.

[Codex packet](codex/report.md) has NOT/NOTCam, LBT/LBC, Liverpool/IO:O and FLWO/KeplerCam; [second ground packet](ground-agent/report.md) has the remaining six. Direct facility sources dominate. FLWO is a clearly labeled secondary JLA copy via sncosmo; Lick is a pinned Pantheon+ research calibration release with raw/shifted epochs retained. Neither is represented as a directly measured pure filter component.

Especially useful component gains: Liverpool's author dataset separates seven filter responses, CCD QE and cryostat window; IRTF includes blocker data; CAHA BUSCA includes a beam splitter; JCMT supplies multicolumn model tables and measured responses with different conventions; model column roles remain unresolved and its second column must not be assumed throughput. These do not certify complete optical paths.

## Space

AKARI, COBE, IRAS, WMAP, Planck, MSX, CHEOPS, MOST, CoRoT and XMM-Newton. PLATO is a free planned-mission extra and excluded from the ten flown count.

[Space packet](space/report.md): 32 raw downloads and 171 extracted series, including one PLATO series. Seven flown missions have primary mission/archive data; CHEOPS uses team software, MOST/CoRoT secondary software-host copies with original provenance unresolved. IRAS has separate filter/detector/system series. Radio/far-IR spectral coordinates and XMM effective area are preserved in their native scientific conventions.

## Checks and remaining limits

[Independent review](review/report.md), [machine-readable audit](review/independent-audit.json), and [space researcher's ground crosscheck](space/ground-crosscheck.md) record verification. Researchers corrected NOT's mixed per-file units after review; inference is explicitly labeled. LBT NaNs, negative measured samples and all source bytes remain preserved. SHA256 and numeric parsability verify the saved evidence, not calibration validity.

Ground packets total 238 numeric files/tables (104 + 134), not 238 physical filters. MMT sparse tab columns require tab-aware parsing; its z scan has variable-width spreadsheet material. PANIC blocking scans are optical density, not transmission. BUSCA beam-splitter “QE” is not detector QE.

IRAS 25 micron has a suspected source transcription error; Planck's 857 GHz curves have non-increasing native coordinates; some instrument versions/epochs and software-host response conventions remain unresolved. Count files/series separately from unique physical filters. SVO was attempted as the requested rapid fallback but its server refused connections during this run; no successful download is attributed to SVO.

Next completeness work: obtain original MOST/CoRoT and current CHEOPS releases; resolve IRAS transcription; add AKARI FIS; inspect more separated optics/QE and atmosphere where only combined responses exist; revisit AAO/APEX/TNG/SALT and NOIRLab when numeric endpoints respond.
