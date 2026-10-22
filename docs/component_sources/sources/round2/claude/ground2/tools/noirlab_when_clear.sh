#!/bin/zsh
# wait for the NOIRLab WAF challenge to clear (no bypass), then harvest serially
cd /home/nill/agent-share/projects/sed-tools-components/round2/claude/ground2
for n in {1..24}; do
  c=$(curl -s -m 30 -o /dev/null -w '%{http_code}' https://noirlab.edu/science/filters/kp1051)
  echo "$(date -u +%FT%TZ) probe $c"
  [[ $c == 200 ]] && break; sleep 300
done
[[ $c == 200 ]] || { echo gave-up; exit 1; }
N=https://noirlab.edu/science/programs
python3 -I tools/harvest.py evidence/noirlab noirlab $N/ctio/filters/Dark-Energy-Camera $N/ctio/filters/MOSAIC-Filters $N/ctio/filters/ISPI $N/ctio/filters/Hydra $N/ctio/filters/andicam $N/ctio/filters/3x3-inch-and-4x4-inch-filters $N/kpno/filters/newfirm $N/kpno/filters/mosaic $N/kpno/filters/2inch $N/kpno/filters/4inch $N/kpno/filters/wiyn-09 $N/kpno/filters/flamingos $N/kpno/filters/kosmos $N/kpno/filters/goldcam $N/kpno/filters/rc $N/kpno/filters/echelle $N/kpno/filters/sgt $N/kpno/filters/phoenix $N/ctio/telescopes/soar-telescope/filters $N/kpno/instruments/whirc/filters
echo done
