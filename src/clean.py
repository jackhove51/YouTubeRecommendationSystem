"""
Cleans raw API items (data/raw/videos_*.jsonl) into the processed schema from
docs/data_schema.md and writes a small random sample to
data/sample/videos_sample_100.parquet.

Usage: python src/clean.py
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/sample/videos_sample_100.parquet")

SAMPLE_SIZE = 100
SEED = 42
DESCRIPTION_MAX_CHARS = 500  # text_clean uses description[:500]
MIN_TEXT_LEN = 20            # drop records whose text_clean is shorter than this

# Must stay in sync with VIDEO_SCHEMA in schema.py (checked at runtime).
PA_SCHEMA = pa.schema([
    pa.field("video_id", pa.string(), nullable=False),
    pa.field("title", pa.string(), nullable=False),
    pa.field("description", pa.string(), nullable=False),
    pa.field("tags", pa.list_(pa.field("item", pa.string(), nullable=False)), nullable=False),
    pa.field("text_clean", pa.string(), nullable=False),
    pa.field("category_id", pa.string(), nullable=False),
    pa.field("channel_id", pa.string(), nullable=False),
    pa.field("published_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("view_count", pa.int64(), nullable=True),
    pa.field("like_count", pa.int64(), nullable=True),
    pa.field("comment_count", pa.int64(), nullable=True),
    pa.field("language", pa.string(), nullable=True),
    pa.field("publish_year", pa.int32(), nullable=False),
])

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
HASHTAG_RE = re.compile(r"#(\w)")
WS_RE = re.compile(r"\s+")


def clean_display(text: str | None) -> str:
    """Title/description cleaning: no HTML, no URLs, collapsed whitespace, casing kept."""
    text = TAG_RE.sub(" ", text or "")
    text = html.unescape(text)  # after tag stripping, so '&lt;3' is not read as a tag
    text = URL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def build_text_clean(title: str, description: str, tags: list[str]) -> str:
    """title + description[:500] + tags, with timestamps and hashtag symbols
    removed, lowercased. The channel name is deliberately excluded:
    relevance is defined by channel/category, so including it would leak labels."""
    text = f"{title} {description[:DESCRIPTION_MAX_CHARS]} {' '.join(tags)}"
    text = text.replace("\u2019", "'").lower()
    text = TIMESTAMP_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    return WS_RE.sub(" ", text).strip()


def to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_item(item: dict) -> tuple[dict | None, str | None]:
    """Return (row, None) for a valid item, or (None, drop_reason)."""
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    video_id = item.get("id")
    if not video_id:
        return None, "missing_video_id"

    title = clean_display(snippet.get("title"))
    if not title:
        return None, "missing_title"

    category_id = snippet.get("categoryId")
    if not category_id:
        return None, "missing_category_id"

    channel_id = snippet.get("channelId")
    if not channel_id:
        return None, "missing_channel_id"

    try:
        published_at = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        return None, "bad_published_at"
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    description = clean_display(snippet.get("description"))

    tags, seen = [], set()
    for tag in snippet.get("tags") or []:
        tag = WS_RE.sub(" ", html.unescape(str(tag))).strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    text_clean = build_text_clean(title, description, tags)
    if len(text_clean) < MIN_TEXT_LEN:
        return None, "text_too_short"

    return {
        "video_id": video_id,
        "title": title,
        "description": description,
        "tags": tags,
        "text_clean": text_clean,
        "category_id": str(category_id),
        "channel_id": channel_id,
        "published_at": published_at,
        "view_count": to_int(stats.get("viewCount")),
        "like_count": to_int(stats.get("likeCount")),
        "comment_count": to_int(stats.get("commentCount")),
        "language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage"),
        "publish_year": published_at.year,
    }, None


def iter_raw_items():
    """Yield each saved video item (parsed JSON) from the raw directory."""
    for path in sorted(RAW_DIR.glob("videos_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def check_against_schema_py() -> None:
    """Stop if PA_SCHEMA drifts from VIDEO_SCHEMA in schema.py (skipped without pyspark)."""
    try:
        from schema import VIDEO_SCHEMA
    except ImportError:
        return
    spark_cols = [(f.name, f.nullable) for f in VIDEO_SCHEMA.fields]
    arrow_cols = [(f.name, f.nullable) for f in PA_SCHEMA]
    if spark_cols != arrow_cols:
        sys.exit("PA_SCHEMA in clean.py is out of sync with VIDEO_SCHEMA in schema.py.")


def main():
    check_against_schema_py()
    if not any(RAW_DIR.glob("videos_*.jsonl")):
        sys.exit(f"No raw files found in {RAW_DIR}. Run collect.py fetch first.")

    # One pass: clean, dedupe, and reservoir-sample. Raw files are ordered by
    # country, so 'first N rows' would be biased; the reservoir gives a uniform
    # random sample without loading all records into memory.
    rng = random.Random(SEED)
    stats, seen_ids, reservoir = Counter(), set(), []
    for item in iter_raw_items():
        row, reason = clean_item(item)
        if row is None:
            stats[f"dropped: {reason}"] += 1
            continue
        if row["video_id"] in seen_ids:
            stats["dropped: duplicate"] += 1
            continue
        seen_ids.add(row["video_id"])
        stats["kept"] += 1

        i = stats["kept"] - 1
        if i < SAMPLE_SIZE:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < SAMPLE_SIZE:
                reservoir[j] = row

    print("Cleaning report:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count:,}")
    if not reservoir:
        sys.exit("No valid records survived cleaning.")
    if len(reservoir) < SAMPLE_SIZE:
        print(f"Warning: only {len(reservoir)} valid records (wanted {SAMPLE_SIZE}).")

    reservoir.sort(key=lambda r: r["video_id"])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reservoir, schema=PA_SCHEMA), OUT_FILE)

    table = pq.read_table(OUT_FILE)
    print(f"\nWrote {table.num_rows} rows to {OUT_FILE}")
    print(f"  empty tags:         {sum(1 for t in table.column('tags').to_pylist() if not t)}")
    print(f"  null like_count:    {table.column('like_count').null_count}")
    print(f"  null comment_count: {table.column('comment_count').null_count}")
    print(f"  languages:          {Counter(table.column('language').to_pylist()).most_common(5)}")


if __name__ == "__main__":
    main()