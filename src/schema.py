from pyspark.sql.types import (
    ArrayType, IntegerType, LongType, StringType, StructField, StructType, TimestampType
)

# Processed videos table (data/processed/videos_vN/), partitioned by publish_year.
VIDEO_SCHEMA = StructType([
    StructField("video_id", StringType(), nullable=False),       # primary key
    StructField("title", StringType(), nullable=False),
    StructField("description", StringType(), nullable=False),    # "" if missing
    StructField("tags", ArrayType(StringType(), containsNull=False), nullable=False),  # [] if none
    StructField("text_clean", StringType(), nullable=False),     # input for shingling / TF-IDF
    StructField("category_id", StringType(), nullable=False),
    StructField("channel_id", StringType(), nullable=False),
    StructField("published_at", TimestampType(), nullable=False),  # UTC
    StructField("view_count", LongType(), nullable=True),
    StructField("like_count", LongType(), nullable=True),        # null if hidden, never 0-filled
    StructField("comment_count", LongType(), nullable=True),     # null if disabled, never 0-filled
    StructField("language", StringType(), nullable=True),
    StructField("publish_year", IntegerType(), nullable=False),  # partition column
])

# Lookup: video category names (from videoCategories.list).
CATEGORY_SCHEMA = StructType([
    StructField("category_id", StringType(), nullable=False),
    StructField("category_title", StringType(), nullable=False),
])

# Lookup: contiguous integer index per video, stable within one dataset version.
ITEM_INDEX_SCHEMA = StructType([
    StructField("video_id", StringType(), nullable=False),
    StructField("item_idx", IntegerType(), nullable=False),
])

PARTITION_COLUMN = "publish_year"


def write_videos(df, path: str) -> None:
    """Write a cleaned videos DataFrame as partitioned Parquet.

    Never overwrite a released version: write to a new videos_vN/ path instead.
    """
    (
        df.select([f.name for f in VIDEO_SCHEMA.fields])
        .write.mode("errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(path)
    )


def read_videos(spark, path: str):
    """Read a processed videos dataset with the expected schema."""
    return spark.read.parquet(path)


def validate_videos(df) -> None:
    """Raise if the DataFrame doesn't match VIDEO_SCHEMA's column names and types."""
    expected = {f.name: f.dataType for f in VIDEO_SCHEMA.fields}
    actual = {f.name: f.dataType for f in df.schema.fields}

    missing = expected.keys() - actual.keys()
    extra = actual.keys() - expected.keys()
    wrong_type = {
        name: (expected[name], actual[name])
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    }

    problems = []
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected columns: {sorted(extra)}")
    if wrong_type:
        problems.append(f"type mismatches: {wrong_type}")
    if problems:
        raise ValueError("Schema validation failed: " + "; ".join(problems))