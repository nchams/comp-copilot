"""Download the REAL public datasets that back this project.

Two sources, both genuinely public and free:

  1. H-1B LCA Disclosure Data (US DOL, Office of Foreign Labor Certification).
     Quarterly/annual Excel files of actual offered wages by employer, job
     title, SOC code, and worksite. This is the closest public proxy to real
     offer data and is the spine of the benchmarking model once leveled via
     src/leveling.py.
     Landing page: https://www.dol.gov/agencies/eta/foreign-labor/performance

  2. BLS OEWS (Occupational Employment & Wage Statistics).
     Wage percentiles (p10/p25/p50/p75/p90) by SOC code and metro area -- used
     to ground "market percentile" claims in an authoritative source.
     Landing page: https://www.bls.gov/oes/tables.htm

NOTE: these files are large (the H-1B annual file is ~150-200MB). The default
pipeline uses data/generate_sample.py so the repo runs instantly. Run this
script when you want to demonstrate on real disclosure data.

Because the exact file URLs change every release, the canonical URLs are read
from data/sources.json (edit that file with the current release links) rather
than hard-coded here.
"""

import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def _load_sources():
    path = os.path.join(HERE, "sources.json")
    if not os.path.exists(path):
        sys.exit(
            "data/sources.json not found. Create it with the current release URLs, e.g.:\n"
            '{\n  "h1b_lca": "https://.../LCA_Disclosure_Data_FY2024.xlsx",\n'
            '  "bls_oews": "https://www.bls.gov/oes/special-requests/oesm23nat.zip"\n}'
        )
    with open(path) as f:
        return json.load(f)


def download(name: str, url: str):
    os.makedirs(RAW, exist_ok=True)
    fname = os.path.join(RAW, os.path.basename(url.split("?")[0]))
    if os.path.exists(fname):
        print(f"[skip] {name}: already have {os.path.basename(fname)}")
        return fname
    print(f"[get ] {name}: {url}")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "comp-copilot/1.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r       {done/1e6:6.1f} / {total/1e6:6.1f} MB", end="")
        print()
    return fname


if __name__ == "__main__":
    srcs = _load_sources()
    for name, url in srcs.items():
        try:
            download(name, url)
        except Exception as e:  # noqa: BLE001 -- surface and continue to next source
            print(f"[fail] {name}: {e}")
    print(f"\nDone. Raw files in {RAW}")
    print("Next: write a leveling pass that maps JOB_TITLE -> (role, level) via "
          "src.leveling.normalize_title, then point train.py at the result.")
