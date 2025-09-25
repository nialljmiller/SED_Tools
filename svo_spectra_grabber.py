#!/usr/bin/env python3
# svo_spectra_grabber.py — fast, bounded SVO discovery & downloader

import csv, os, re, io, time, json, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

try:
    from astropy.io.votable import parse_single_table as vot_parse_single_table
    _HAS_VOTABLE = True
except Exception:
    _HAS_VOTABLE = False


class SVOSpectraGrabber:
    """
    Fast/robust discovery:
      1) SSAP VOTable (streamed, max bytes, hard timeout) → parse URLs → FIDs
      2) Fallback: scrape model index for 'fid='
      3) Fallback: bounded parallel HEAD probes (sparse → expand locally), with a small budget

    Download:
      - ASCII spectra to <base>/<model>/
      - Build lookup_table.csv from parsed headers
    """

    def __init__(self, base_dir="../data/stellar_models/", max_workers=8):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools/1.0 (SVO)"})

        # Endpoints
        self.model_index_url = "http://svo2.cab.inta-csic.es/theory/newov2/index.php"
        self.spectra_base_url = "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php"

        # Budgets (tweak if needed)
        self.ssap_timeout = 15         # seconds
        self.ssap_max_bytes = 2_000_000  # 2 MB cap when streaming SSAP
        self.head_timeout = 6          # seconds per HEAD
        self.bruteforce_budget = 350   # maximum total HEADs
        self.expand_window = 12
        self.expand_max_gap = 20

        # Cache
        self.cache_root = os.path.join(self.base_dir, ".cache", "svo_fids")
        os.makedirs(self.cache_root, exist_ok=True)

    # -------------------- model list --------------------

    def discover_models(self):
        print("Discovering available models from SVO...")
        try:
            r = self.session.get(self.model_index_url, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[SVO] model index failed: {e}")
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        models = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "models=" in href:
                val = href.split("models=", 1)[1].split("&", 1)[0]
                if val:
                    models.add(val)
        if not models:
            for m in re.findall(r"models=([A-Za-z0-9._\-]+)", r.text):
                models.add(m)
        out = sorted(models)
        if not out:
            print("  [warn] no models found on SVO index page.")
        return out

    # -------------------- spectra discovery --------------------

    def get_model_metadata(self, model_name):
        print(f"  Fetching metadata for {model_name}...")

        # cache hit?
        cache_path = os.path.join(self.cache_root, f"{model_name}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    print(f"    (cache) {len(data)} spectra")
                    return [{"fid": int(x)} for x in sorted(set(int(y) for y in data))]
            except Exception:
                pass

        fids = set()

        # A) SSAP VOTable (streamed, bounded)
        fids |= self._discover_ssap_stream(model_name)

        # B) model index scrape
        if not fids:
            fids |= self._discover_index_page(model_name)

        # C) bounded sparse probe + expansion
        if not fids:
            fids |= self._bounded_probe(model_name)

        out = [{"fid": int(fid)} for fid in sorted(fids)]
        print(f"    Found {len(out)} spectra for {model_name}")

        # cache for next time
        if out:
            try:
                with open(cache_path, "w") as f:
                    json.dump([int(d["fid"]) for d in out], f)
            except Exception:
                pass

        return out

    # ---- A) SSAP (streamed) ----

    def _discover_ssap_stream(self, model_name):
        params = {"model": model_name, "REQUEST": "queryData", "FORMAT": "votable"}
        try:
            with self.session.get(self.spectra_base_url, params=params,
                                  timeout=self.ssap_timeout, stream=True) as r:
                if r.status_code != 200:
                    return set()
                buf = io.BytesIO()
                total = 0
                for chunk in r.iter_content(chunk_size=65536):
                    if not chunk:
                        break
                    buf.write(chunk)
                    total += len(chunk)
                    if total >= self.ssap_max_bytes:
                        break
                raw = buf.getvalue()
        except Exception as e:
            print(f"    [SVO] SSAP stream failed: {e}")
            return set()

        # Try Astropy parser first (if present)
        if _HAS_VOTABLE:
            try:
                table = vot_parse_single_table(io.BytesIO(raw)).to_table()
                url_cols = [c for c in table.colnames if any(k in c.lower() for k in ("access", "acref", "url"))]
                fids = set()
                for row in table:
                    for c in url_cols:
                        val = str(row[c])
                        m = re.search(r"[?&]fid=(\d+)", val)
                        if m:
                            fids.add(int(m.group(1))); break
                    else:
                        for c in table.colnames:
                            val = str(row[c])
                            m = re.search(r"[?&]fid=(\d+)", val)
                            if m: fids.add(int(m.group(1))); break
                if fids:
                    print(f"    SSAP: {len(fids)} fids (parsed)")
                    return fids
            except Exception as e:
                print(f"    [SVO] VOTable parse failed: {e}")

        # Fallback: regex over raw
        text = raw.decode("utf-8", "ignore")
        fids = set(int(x) for x in re.findall(r"[?&]fid=(\d+)", text))
        if fids:
            print(f"    SSAP(regex): {len(fids)} fids")
        return fids

    # ---- B) index scrape ----
    def _discover_index_page(self, model_name):
        try:
            url = f"{self.model_index_url}?models={urllib.parse.quote(model_name)}"
            r = self.session.get(url, timeout=20)
            if r.status_code != 200:
                return set()
            fids = set(int(x) for x in re.findall(r"[?&]fid=(\d+)", r.text))
            if fids:
                print(f"    index: {len(fids)} fids")
            return fids
        except Exception:
            return set()

    # ---- C) bounded probe ----
    def _bounded_probe(self, model_name):
        print("    Probing sparsely (bounded)...")
        # Geometric-ish sparse probes; tuned to keep budget small
        probes = [1,2,3,4,5,6,7,8,9,10,
                  12,15,20,25,30,40,50,60,75,100,
                  125,150,200,250,300,400,500,650,800,1000,
                  1200,1500,2000,2500,3000,4000,5000,6500,8000,10000]
        hits = []
        budget = min(self.bruteforce_budget, len(probes))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, 16)) as ex:
            futs = {ex.submit(self._head_exists, model_name, fid): fid for fid in probes[:budget]}
            for fut in as_completed(futs):
                fid = futs[fut]
                try:
                    if fut.result():
                        hits.append(fid)
                except Exception:
                    pass
        if not hits:
            return set()
        # expand locally around each hit, with early-stop on gaps
        found = set(hits)
        for seed in sorted(hits):
            found |= self._expand_around(model_name, seed)
        print(f"    probe+expand: {len(found)} fids")
        return found

    def _head_exists(self, model_name, fid):
        params = {"model": model_name, "fid": int(fid), "format": "ascii"}
        try:
            r = self.session.head(self.spectra_base_url, params=params,
                                  timeout=self.head_timeout, allow_redirects=True)
            if r.status_code == 200:
                cl = r.headers.get("content-length")
                if cl is None or cl == "0":
                    g = self.session.get(self.spectra_base_url, params=params, timeout=self.head_timeout, stream=True)
                    ok = (g.status_code == 200)
                    if ok:
                        # read tiny chunk
                        for chunk in g.iter_content(chunk_size=256):
                            if chunk:
                                return True
                        return False
                return True
            return False
        except Exception:
            return False

    def _expand_around(self, model_name, seed):
        found = set([seed])
        for direction in (-1, +1):
            misses = 0
            step = 1
            while misses < self.expand_max_gap:
                fid = seed + direction * (self.expand_window + step - 1)
                step += 1
                if fid <= 0:
                    misses += 1
                    continue
                if self._head_exists(model_name, fid):
                    found.add(fid)
                    misses = 0
                else:
                    misses += 1
        return found

    # -------------------- download + lookup --------------------

    def download_model_spectra(self, model_name, spectra_info):
        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        if not spectra_info:
            print(f"  No spectra to download for {model_name}.")
            return 0

        print(f"Downloading {len(spectra_info)} spectra for {model_name}...")
        rows, ok = [], 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {}
            for spec in spectra_info:
                fid = int(spec["fid"])
                fname = f"{model_name}_fid{fid}.txt"
                fpath = os.path.join(out_dir, fname)

                if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
                    meta = self._parse_header(fpath); meta["file_name"] = fname
                    rows.append(meta); ok += 1
                    continue

                futures[ex.submit(self._download_one, model_name, fid, fpath)] = (fid, fname, fpath)

            for fut in as_completed(futures):
                fid, fname, fpath = futures[fut]
                try:
                    if fut.result():
                        meta = self._parse_header(fpath); meta["file_name"] = fname
                        rows.append(meta); ok += 1
                    else:
                        if os.path.exists(fpath): os.remove(fpath)
                except Exception as e:
                    print(f"    Error processing FID {fid}: {e}")

        if rows:
            self._write_lookup(out_dir, rows)
            print(f"  Successfully downloaded {ok} spectra")
        else:
            print(f"  No spectra downloaded for {model_name}")
        return ok

    def _download_one(self, model_name, fid, out_path):
        params = {"model": model_name, "fid": int(fid), "format": "ascii"}
        try:
            r = self.session.get(self.spectra_base_url, params=params, timeout=60, stream=True)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk: break
                        f.write(chunk)
                return os.path.getsize(out_path) > 1024
            return False
        except Exception:
            return False

    def _parse_header(self, file_path):
        meta = {}
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#") and "=" in s:
                        try:
                            key, val = s.split("=", 1)
                            key = key.strip("#").strip()
                            val = val.split("(")[0].strip()
                            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
                            meta[key] = m.group(0) if m else val
                        except Exception:
                            continue
                    elif not s.startswith("#"):
                        break
        except Exception:
            pass
        return meta

    def _write_lookup(self, out_dir, rows):
        path = os.path.join(out_dir, "lookup_table.csv")
        keys = set()
        for r in rows: keys.update(r.keys())
        header = ["file_name"] + sorted(k for k in keys if k != "file_name")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("#" + ",".join(header) + "\n")
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writerows(rows)
        print(f"    Lookup table saved: {path}")
