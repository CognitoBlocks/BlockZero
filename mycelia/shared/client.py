#!/usr/bin/env python3
# download_checkpoint.py
import argparse
import os
import sys
import requests
from time import time

CHUNK = 1024 * 1024  # 1 MiB

def human(n):
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"

def download(url: str, token: str, out: str, resume: bool = False, timeout: int = 30):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    mode = "wb"
    start_at = 0

    if resume and os.path.exists(out):
        start_at = os.path.getsize(out)
        if start_at > 0:
            headers["Range"] = f"bytes={start_at}-"
            mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code in (401, 403):
            sys.exit(f"Auth failed (HTTP {r.status_code}). Check your token.")
        if r.status_code == 416:
            print("Nothing to resume; file already complete.")
            return
        if resume and r.status_code not in (200, 206):
            print(f"Server did not honor range request (HTTP {r.status_code}). Restarting full download.")
            # retry full download
            headers.pop("Range", None)
            mode = "wb"
            start_at = 0
            r.close()
            r = requests.get(url, headers=headers, stream=True, timeout=timeout)

        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total = int(total) + start_at if total is not None else None

        downloaded = start_at
        t0 = time()
        last_print = t0

        with open(out, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time()
                if now - last_print >= 0.5:
                    if total:
                        pct = downloaded / total * 100
                        bar = f"{human(downloaded)} / {human(total)} ({pct:5.1f}%)"
                    else:
                        bar = f"{human(downloaded)}"
                    rate = (downloaded - start_at) / max(1e-6, (now - t0))
                    print(f"\rDownloading: {bar} @ {human(rate)}/s", end="", flush=True)
                    last_print = now

        # final line
        elapsed = max(1e-6, time() - t0)
        rate = (downloaded - start_at) / elapsed
        if total:
            bar = f"{human(downloaded)} / {human(total)} (100.0%)"
        else:
            bar = f"{human(downloaded)}"
        print(f"\rDone:       {bar} in {elapsed:.1f}s @ {human(rate)}/s")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download a checkpoint with optional bearer auth.")
    p.add_argument("--url", default="http://localhost:8000/checkpoint", help="Download URL")
    p.add_argument("--token", default="supersecrettoken", help="Bearer auth token (omit if not required)")
    p.add_argument("-o", "--output", default="model.pt", help="Output file path")
    p.add_argument("--resume", action="store_true", help="Resume if partial file exists")
    p.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    args = p.parse_args()

    try:
        download(args.url, args.token, args.output, resume=args.resume, timeout=args.timeout)
    except requests.RequestException as e:
        sys.exit(f"Network error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nCanceled.")