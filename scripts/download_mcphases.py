"""
Download the mcPHASES dataset from PhysioNet (restricted access).

    python scripts/download_mcphases.py

mcPHASES is the dataset that would let the twin LEARN menstrual-cycle effects instead of assuming
them: 42 women, ~3 months each (three full cycles), Dexcom G6 CGM, Fitbit Sense, and daily LH / E3G /
PdG urine assays with labelled cycle phases. See "Future work" in the README.

BEFORE THIS WILL WORK you must, in a browser, as yourself:
  1. Register at https://physionet.org and log in.
  2. Open https://www.physionet.org/content/mcphases/1.0.0/ and SIGN the
     "PhysioNet Restricted Health Data Use Agreement 1.5.0" for this project.
     That is a legal agreement in your own name - nobody can sign it for you.
Signing is immediate; there is no CITI training and no approval wait for this dataset.

This script uses only the Python standard library (no wget needed - wget is not installed on most
Windows machines, which is the usual reason the documented wget command fails). Your password is
typed at a prompt, is never stored, never written to disk, and never appears in your shell history.

Safe to stop with Ctrl+C and re-run: finished files are skipped.
"""
from __future__ import annotations

import base64
import getpass
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://physionet.org/files/mcphases/1.0.0/"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mcphases"


class LinkParser(HTMLParser):
    """Collect hrefs from a PhysioNet directory listing page."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)


def opener_for(user: str, password: str):
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    op = urllib.request.build_opener()
    op.addheaders = [("Authorization", f"Basic {auth}"),
                     ("User-Agent", "GlucoTwin-mcphases-downloader")]
    return op


def listing(op, url: str) -> list[str]:
    with op.open(url, timeout=60) as r:
        html = r.read().decode("utf-8", errors="replace")
    p = LinkParser()
    p.feed(html)
    out = []
    for h in p.links:
        if h.startswith(("?", "#", "/")) or "://" in h or h in ("../",):
            continue
        out.append(h)
    return out


def walk(op, url: str, dest: Path, depth: int = 0) -> tuple[int, int]:
    """Recursively download a PhysioNet directory. Returns (files_ok, bytes)."""
    dest.mkdir(parents=True, exist_ok=True)
    n_ok = n_bytes = 0
    for name in listing(op, url):
        child_url, child_dest = url + name, dest / name.rstrip("/")
        if name.endswith("/"):
            a, b = walk(op, child_url, child_dest, depth + 1)
            n_ok += a
            n_bytes += b
            continue
        if child_dest.exists() and child_dest.stat().st_size > 0:
            continue
        try:
            with op.open(child_url, timeout=300) as r:
                data = r.read()
            child_dest.write_bytes(data)
            n_ok += 1
            n_bytes += len(data)
            print(f"  {'  ' * depth}{name}  ({len(data) / 1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"  {'  ' * depth}{name}: FAILED ({type(e).__name__}) - re-run to retry", flush=True)
    return n_ok, n_bytes


def main() -> int:
    print(__doc__.split("BEFORE")[0].strip())
    user = input("\nPhysioNet username: ").strip()
    if not user:
        print("No username given.")
        return 1
    password = getpass.getpass("PhysioNet password (not shown, not stored): ")

    op = opener_for(user, password)
    print(f"\nChecking access to {BASE} ...")
    try:
        listing(op, BASE)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print(f"\n  HTTP {e.code}: access denied.\n"
                  "  Either the username/password is wrong, or you have not signed the data use\n"
                  "  agreement yet. Log in at https://www.physionet.org/content/mcphases/1.0.0/\n"
                  "  and sign the agreement on that page, then run this again.")
        else:
            print(f"\n  HTTP {e.code}: {e.reason}")
        return 1
    except Exception as e:
        print(f"\n  Could not reach PhysioNet ({type(e).__name__}). Check your connection.")
        return 1

    print("  access OK - downloading\n")
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        n, b = walk(op, BASE, ROOT)
    except KeyboardInterrupt:
        print("\nStopped. Finished files are kept; re-run to continue.")
        return 130
    print(f"\nDone: {n} new file(s), {b / 1e6:,.0f} MB")
    print(f"Data folder: {ROOT}")
    print("\nNext: show the file listing to Claude so the loader can be written against the real\n"
          "schema rather than a guessed one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
