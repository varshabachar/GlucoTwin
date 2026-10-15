"""
Download the BIG IDEAs files GlucoTwin needs from PhysioNet, smallest and most important first.

    python scripts/download_data.py            # CORE: glucose, meals, heart rate, beat-to-beat  (~360 MB)
    python scripts/download_data.py --extra    # + skin temperature and EDA                     (~2.6 GB more)
    python scripts/download_data.py --acc      # + accelerometer                                (~13 GB more)

CORE is enough to run everything. Without the accelerometer, GlucoTwin estimates activity and sleep
from heart rate (see src/load_data.py). Skips BVP (raw pulse waveform) entirely - not used.

Safe to stop: close the window or press Ctrl+C, then re-run. Finished files are skipped and partial
files resume where they stopped. Uses only the Python standard library (no pip install needed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "big-ideas"
CORE = ["Dexcom", "Food_Log", "HR", "IBI"]
EXTRA = ["TEMP", "EDA"]
ACC = ["ACC"]
PIDS = [f"{i:03d}" for i in range(1, 17)]

lock = threading.Lock()
stop = threading.Event()
done_bytes = 0


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "GlucoTwin-downloader"})
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch(url: str, dest: Path, retries: int = 6) -> str:
    """Download url -> dest with resume. Returns a status string."""
    global done_bytes
    if dest.exists():
        return "already have"
    part = dest.with_name(dest.name + ".part")
    err = "?"
    for attempt in range(1, retries + 1):
        if stop.is_set():
            return "stopped"
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "GlucoTwin-downloader"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(part, mode) as f:
                    while not stop.is_set():
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        with lock:
                            done_bytes += len(chunk)
            if stop.is_set():
                return "stopped"
            part.replace(dest)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "not on server"
            if e.code == 416:  # nothing left to fetch -> already complete
                part.replace(dest)
                return "ok"
            err = f"HTTP {e.code}"
        except Exception as e:  # network hiccup -> retry
            err = type(e).__name__
        time.sleep(min(30, 3 * attempt))
    return f"failed ({err}) - just run the script again to resume"


def run(jobs: list[tuple[str, Path]], workers: int, label: str) -> int:
    todo = [(u, d) for u, d in jobs if not d.exists()]
    print(f"\n== {label} ==")
    print(f"   {len(jobs) - len(todo)} of {len(jobs)} files already downloaded")
    if not todo:
        return 0
    print("   checking file sizes ...", flush=True)
    with cf.ThreadPoolExecutor(16) as ex:
        sizes = list(ex.map(lambda j: remote_size(j[0]), todo))
    have_part = sum((d.with_name(d.name + ".part").stat().st_size if d.with_name(d.name + ".part").exists() else 0)
                    for _, d in todo)
    total = sum(s for s in sizes if s) - have_part
    print(f"   about {max(total, 0) / 1e6:,.0f} MB to download, {workers} files at a time", flush=True)

    start, base = time.time(), done_bytes
    finished = [0]
    failed = 0

    def ticker():
        while not stop.wait(30):
            got = done_bytes - base
            speed = got / max(time.time() - start, 1)
            left = (total - got) / speed / 60 if speed > 0 and total > 0 else float("nan")
            eta = f", about {left:,.0f} min left" if left == left else ""
            print(f"   {finished[0]}/{len(todo)} files done, {got / 1e6:,.0f} MB, {speed / 1e6:.2f} MB/s{eta}", flush=True)

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    ex = cf.ThreadPoolExecutor(workers)
    try:
        futs = {ex.submit(fetch, url, dest): dest for url, dest in todo}
        for fu in cf.as_completed(futs):
            status = fu.result()
            finished[0] += 1
            if status.startswith("failed"):
                failed += 1
            if status not in ("ok", "already have"):
                print(f"   {futs[fu].parent.name}/{futs[fu].name}: {status}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping... partial files are kept and will resume next time.", flush=True)
        stop.set()
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true", help="also skin temperature + EDA (~2.6 GB)")
    ap.add_argument("--acc", action="store_true", help="also accelerometer (~13 GB)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=BASE)
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    for pid in PIDS:
        (ROOT / pid).mkdir(exist_ok=True)

    def jobs_for(kinds):
        return [(f"{args.base_url}/{pid}/{k}_{pid}.csv", ROOT / pid / f"{k}_{pid}.csv") for pid in PIDS for k in kinds]

    failed = 0
    try:
        core = [(f"{args.base_url}/Demographics.csv", ROOT / "Demographics.csv")] + jobs_for(CORE)
        failed += run(core, args.workers, "CORE: glucose, meals, heart rate, beat-to-beat (16 people)")
        if failed == 0:
            print("\nCORE DONE. You can now run:  python run_pipeline.py")
        if args.extra:
            failed += run(jobs_for(EXTRA), args.workers, "EXTRA: skin temperature + EDA")
        if args.acc:
            failed += run(jobs_for(ACC), args.workers, "ACCELEROMETER")
        if args.extra or args.acc:
            print("\nAfter new files arrive, rebuild the sensor tables:  python run_pipeline.py --no-cache")
    except KeyboardInterrupt:
        return 130

    print(f"\nData folder: {ROOT}")
    if failed:
        print(f"{failed} file(s) did not finish - run this script again to resume them.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
