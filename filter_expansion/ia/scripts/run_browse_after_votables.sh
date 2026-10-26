#!/usr/bin/env bash
# Waits for the VOTable downloader (PID given as $1) to exit, then downloads the fps_browse queue
# with the same 4.3 s throttle and manifest. One crawler at a time: Mac, Cirrus and vega share one IP.
set -u
IA=/home/nill/SED_Workspace/filter-expansion/phase3/ia
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) votable downloader $1 exited; starting fps_browse" >> "$IA/download_browse.log"
cd /tmp && exec nice -n 10 /home/nill/SED_Workspace/filter-expansion/.venv/bin/python -I \
  "$IA/scripts/download_browse.py" "$IA/cdx/queue_browse.json" 4.3 >> "$IA/download_browse.log" 2>&1
