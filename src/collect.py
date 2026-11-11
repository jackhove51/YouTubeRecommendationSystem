"""
YouTube Data API v3 collection script.

Writes raw, untouched API responses to data/raw/ (one JSON object per line).
Cleaning into Parquet is a separate step, so we can re-clean without spending quota.

Usage:
    python src/collect.py download-kaggle --seed
    python src/collect.py status
    python src/collect.py categories
    python src/collect.py seed-kaggle [csv ...]      (defaults to data/kaggle/*.csv)
    python src/collect.py fetch --max-videos 2000
    python src/collect.py expand-channels --max-channels 100 --pages-per-channel 2

Typical loop:
    download-kaggle --seed -> fetch -> expand-channels -> fetch -> ...

The API key is read from the YOUTUBE_API_KEY environment variable or a .env file. Never commit the key.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import requests
import sys
import time
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = "https://www.googleapis.com/youtube/v3"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
QUOTA = 10000
SAFETY_MARGIN = 500  # Margin to avoid hitting the quota limit
COST = {"videos": 1, "playlistItems": 1, "videoCategories": 1}  # units per call
BATCH_SIZE = 50  # max ids per videos.list call
VIDEO_PARTS = "snippet,statistics,contentDetails"
PT = ZoneInfo("America/Los_Angeles")  # Pacific Time for daily quota tracking

KAGGLE_DATASET = "datasnaek/youtube-new"

KAGGLE_DIR = Path("data/kaggle")
RAW_DIR = Path("data/raw")
SEED_DIR = Path("data/seeds")
QUOTA_FILE = RAW_DIR / "quota_state.json"
SEED_FILE = SEED_DIR / "video_ids.txt"
MISSING_FILE = RAW_DIR / "missing_ids.txt"
CHANNELS_DONE_FILE = SEED_DIR / "channels_done.txt"


class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass


class APIError(Exception):
    """Raised when the API returns an error."""
    def __init__(self, status_code, reason, message):
        super().__init__(f"HTTP {status_code} ({reason}): {message}")
        self.status_code = status_code
        self.reason = reason
        self.message = message


def today() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return datetime.now(PT).strftime("%Y-%m-%d")


class Quota:
    def __init__(self, used: int = 0):
        self.date = today()
        self.used = used
        if QUOTA_FILE.exists():
            state = json.loads(QUOTA_FILE.read_text())
            if state.get("date") == self.date:
                self.used = state.get("used", 0)

    def _save(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        QUOTA_FILE.write_text(json.dumps({"date": self.date, "used": self.used}))

    def consume(self, units: int):
        if self.date != today():
            self.date = today()
            self.used = 0
        if self.used + units > QUOTA - SAFETY_MARGIN:
            raise QuotaExceededError(f"Quota exceeded: {self.used + units} > {QUOTA - SAFETY_MARGIN}")
        self.used += units
        self._save()

    def mark_exceeded(self):
        self.used = QUOTA
        self._save()

    @property
    def remaining(self) -> int:
        return max(0, QUOTA - SAFETY_MARGIN - self.used)


def api_get(endpoint: str, params: dict, quota: Quota, max_retries: int = 5) -> dict:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        sys.exit("YOUTUBE_API_KEY is not set. Put it in your environment or a .env file.")

    params = {**params, "key": key}
    url = f"{API_BASE_URL}/{endpoint}"

    for attempt in range(max_retries):
        quota.consume(COST[endpoint])  # charged per attempt, to stay conservative
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"Network error ({exc}); retrying in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        reason, message = "unknown", resp.text[:200]
        try:
            err = resp.json()["error"]
            reason = err["errors"][0]["reason"]
            message = err.get("message", message)
        except (ValueError, KeyError, IndexError):
            pass

        if resp.status_code == 403 and reason in ("quotaExceeded", "dailyLimitExceeded"):
            quota.mark_exceeded()
            raise QuotaExceededError("Google reports the daily quota is exceeded.")
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"HTTP {resp.status_code}; retrying in {wait}s")
            time.sleep(wait)
            continue
        raise APIError(resp.status_code, reason, message)

    raise RuntimeError(f"Gave up on {endpoint} after {max_retries} attempts.")


def read_lines(path: Path) -> list[str]:
    """Read a text file and return a list of stripped lines."""
    if not path.exists():
        return []
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_lines(path: Path, lines: list[str]) -> None:
    """Append lines to a text file, creating it if necessary."""
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def iter_raw_items():
    """Yield parsed items from all JSONL files in the raw directory."""
    for path in sorted(RAW_DIR.glob("videos_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def seed_from_csvs(paths: list[Path]) -> None:
    """Extract video ids from Kaggle Trending CSVs (needs a 'video_id' column)."""
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    seen = set(read_lines(SEED_FILE))
    new_ids = []

    for path in paths:
        # utf-8-sig strips a byte-order mark if the file has one; otherwise the
        # first column would read as '\ufeffvideo_id' and the check below would fail.
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            if "video_id" not in (reader.fieldnames or []):
                sys.exit(f"{path} has no 'video_id' column: found {reader.fieldnames}")
            for row in reader:
                vid = row["video_id"].strip()
                # The Kaggle files contain junk ids like '#NAME?'; skip them.
                if ID_PATTERN.match(vid) and vid not in seen:
                    seen.add(vid)
                    new_ids.append(vid)
        print(f"  read {Path(path).name}")

    append_lines(SEED_FILE, new_ids)
    print(f"Added {len(new_ids):,} new ids ({len(seen):,} total) to {SEED_FILE}")

# Commands

def cmd_status(args):
    quota = Quota()
    seeds = set(read_lines(SEED_FILE))
    collected = {item["id"] for item in iter_raw_items()}
    missing = set(read_lines(MISSING_FILE))
    todo = seeds - collected - missing
    print(f"Quota used today {quota.date}: {quota.used}/{QUOTA} "
          f"local budget remaining: {quota.remaining}")
    print(f"Seed ids:       {len(seeds):,}")
    print(f"Collected:      {len(collected):,}")
    print(f"Missing/private:{len(missing):,}")
    print(f"Left to fetch:  {len(todo):,}")


def cmd_categories(args):
    quota = Quota()
    data = api_get(
        "videoCategories",
        {"part": "snippet", "regionCode": "US"},
        quota,
    )
    out = RAW_DIR / "categories.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(data.get('items', []))} categories to {out}")


def cmd_download_kaggle(args):
    """Download the Kaggle Trending dataset with kagglehub and copy its CSVs into data/kaggle/."""
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed. Run: pip install kagglehub")

    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Downloaded to cache: {cache_path}")

    csvs = sorted(cache_path.rglob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {cache_path}")

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in csvs:
        dest = KAGGLE_DIR / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
        print(f"  copied {src.name} -> {dest}")

    if args.seed:
        seed_from_csvs(copied)
    else:
        print("Run 'seed-kaggle' to add these ids to the seed list.")


def cmd_seed_kaggle(args):
    paths = [Path(p) for p in args.paths] or sorted(KAGGLE_DIR.glob("*.csv"))
    if not paths:
        sys.exit(f"No CSV files given and none found in {KAGGLE_DIR}. "
                 "Run 'download-kaggle' first.")
    seed_from_csvs(paths)


def cmd_fetch(args):
    quota = Quota()
    seeds = read_lines(SEED_FILE)
    done = {item["id"] for item in iter_raw_items()} | set(read_lines(MISSING_FILE))
    todo = [vid for vid in seeds if vid not in done]
    if args.max_videos:
        todo = todo[:args.max_videos]

    print(f"{len(todo):,} ids to fetch. Budget: {quota.remaining} units "
          f"(~{quota.remaining * BATCH_SIZE:,} videos).")

    fetched = missing_total = 0
    try:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            data = api_get(
                "videos",
                {"part": VIDEO_PARTS, "id": ",".join(batch), "maxResults": BATCH_SIZE},
                quota,
            )
            items = data.get("items", [])

            # One file per day. Append and flush every batch so a crash loses at most one batch.
            out = RAW_DIR / f"videos_{today()}.jsonl"
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with open(out, "a", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            # Ids that came back empty (deleted/private) are remembered so we
            # never spend quota on them again.
            returned = {item["id"] for item in items}
            gone = [vid for vid in batch if vid not in returned]
            append_lines(MISSING_FILE, gone)

            fetched += len(items)
            missing_total += len(gone)
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"fetched {fetched:,} (missing {missing_total:,}), "
                      f"quota {quota.used}/{QUOTA}")
            time.sleep(0.05)
    except QuotaExceededError as exc:
        print(f"\nStopping: {exc} Re-run after midnight Pacific time to continue.")

    print(f"Done this run: {fetched:,} videos saved, {missing_total:,} missing/private.")


def cmd_expand_channels(args):
    """Take channel ids from collected videos, add their recent uploads to the seeds."""
    quota = Quota()
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    channels, seen_ch = [], set()
    for item in iter_raw_items():
        cid = item.get("snippet", {}).get("channelId")
        if cid and cid not in seen_ch:
            seen_ch.add(cid)
            channels.append(cid)
    done_channels = set(read_lines(CHANNELS_DONE_FILE))
    channels = [c for c in channels if c not in done_channels][: args.max_channels]

    known = set(read_lines(SEED_FILE))
    added = 0
    print(f"Expanding {len(channels)} channels up to {args.pages_per_channel} pages each.")

    try:
        for cid in channels:
            # A channel's uploads playlist id is its channel id with 'UC' -> 'UU'.
            playlist_id = "UU" + cid[2:]
            page_token, new_ids = None, []
            try:
                for _ in range(args.pages_per_channel):
                    params = {"part": "contentDetails", "playlistId": playlist_id,
                              "maxResults": 50}
                    if page_token:
                        params["pageToken"] = page_token
                    data = api_get("playlistItems", params, quota)
                    for it in data.get("items", []):
                        vid = it["contentDetails"]["videoId"]
                        if vid not in known:
                            known.add(vid)
                            new_ids.append(vid)
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
            except APIError as exc:
                if exc.reason not in ("playlistNotFound", "forbidden"):
                    raise
                print(f"Skipping {cid}: {exc.reason}")

            append_lines(SEED_FILE, new_ids)
            append_lines(CHANNELS_DONE_FILE, [cid])
            added += len(new_ids)
    except QuotaExceededError as exc:
        print(f"\nStopping: {exc}")

    print(f"Added {added:,} new seed ids. Run 'fetch' to collect them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show quota and collection progress").set_defaults(func=cmd_status)

    p = sub.add_parser("categories", help="Save the video category lookup (1 unit)")
    p.set_defaults(func=cmd_categories)

    p = sub.add_parser("download-kaggle", help=f"Download {KAGGLE_DATASET} via kagglehub into {KAGGLE_DIR}/")
    p.add_argument("--seed", action="store_true", help="Also add the downloaded video ids to the seed list")
    p.set_defaults(func=cmd_download_kaggle)

    p = sub.add_parser("seed-kaggle", help="Add video ids from Kaggle Trending CSVs")
    p.add_argument("paths", nargs="*")
    p.set_defaults(func=cmd_seed_kaggle)

    p = sub.add_parser("fetch", help="Fetch metadata for seed ids not yet collected")
    p.add_argument("--max-videos", type=int, default=None)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("expand-channels", help="Add channel upload playlists to the seeds")
    p.add_argument("--max-channels", type=int, default=100)
    p.add_argument("--pages-per-channel", type=int, default=2)
    p.set_defaults(func=cmd_expand_channels)

    args = parser.parse_args()
    try:
        args.func(args)
    except QuotaExceededError as exc:
        print(f"Stopping: {exc}")


if __name__ == "__main__":
    main()