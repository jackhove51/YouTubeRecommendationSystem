

import sys
import math
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType

# --- Configuration ---
DATA_PATH = "data/sample/videos_sample_100.parquet"
K_SHINGLE = 4         # Length of character shingles
NUM_HASHES = 100      # Total signature length (must equal bands * rows)

# Prime number for hashing (larger than the max hash of a shingle)
PRIME = 4294967311 

def get_spark_session():
    """Initialize the Spark Session."""
    return SparkSession.builder \
        .appName("YouTubeLSHPipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

# --- Step 1: Shingling ---
def shingle_text(text, k):
    """Convert text to a set of character k-shingles."""
    if not text:
        return []
    return list(set([text[i:i+k] for i in range(len(text) - k + 1)]))

shingle_udf = udf(lambda text: shingle_text(text, K_SHINGLE), ArrayType(StringType()))

# --- Step 2: MinHash Implementation ---
def generate_hash_functions(num_hashes):
    """Generate random (a, b) pairs for the hash functions: h(x) = (a*x + b) % PRIME"""
    random.seed(42)  # For reproducibility
    hash_funcs = []
    for _ in range(num_hashes):
        a = random.randint(1, PRIME - 1)
        b = random.randint(0, PRIME - 1)
        hash_funcs.append((a, b))
    return hash_funcs

HASH_FUNCS = generate_hash_functions(NUM_HASHES)

def minhash_signature(shingles):
    """Compute the MinHash signature for a list of shingles."""
    # Hash the shingles to integers first
    shingle_hashes = [hash(s) % PRIME for s in shingles]
    signature = []
    
    # For each hash function, find the minimum hash value across all shingles
    for a, b in HASH_FUNCS:
        min_hash = float('inf')
        for h in shingle_hashes:
            val = (a * h + b) % PRIME
            if val < min_hash:
                min_hash = val
        signature.append(min_hash)
    return signature

# --- Step 3: LSH Banding ---
def lsh_banding(video_id, signature, bands, rows):
    """Partition signature into bands and emit bucket keys."""
    candidates = []
    for b in range(bands):
        # Extract the slice of the signature for this band
        band_sig = tuple(signature[b * rows : (b + 1) * rows])
        # Hash the band to create a bucket key
        bucket_key = hash(band_sig)
        candidates.append(((b, bucket_key), video_id))
    return candidates

# --- Step 4: Exact Jaccard (Ground Truth) ---
def exact_jaccard(shingles_a, shingles_b):
    set_a, set_b = set(shingles_a), set(shingles_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / union if union > 0 else 0.0

def evaluate_parameter_sweep(df_shingled, exact_pairs_dict, bands, rows):
    """Run LSH for a specific (bands, rows) config and calculate Precision/Recall."""
    print(f"\n--- Sweeping bands={bands}, rows={rows} (Total Hashes={bands*rows}) ---")
    
    # 1. Compute MinHash Signatures (Map)
    sig_rdd = df_shingled.rdd.map(lambda row: (row['video_id'], minhash_signature(row['shingles'])))
    
    # 2. LSH Banding (MapReduce)
    # Map: (bucket_key, video_id)
    banded_rdd = sig_rdd.flatMap(lambda x: lsh_banding(x[0], x[1], bands, rows))
    
    # Reduce: Group by bucket_key and generate candidate pairs
    candidate_pairs_rdd = banded_rdd.groupByKey() \
        .flatMap(lambda x: [(a, b) for a in x[1] for b in x[1] if a < b]) \
        .distinct()
    
    # Collect LSH pairs into a set for quick lookup
    lsh_pairs = set(candidate_pairs_rdd.collect())
    
    # 3. Evaluate against Ground Truth
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Define a similarity threshold for what constitutes a "relevant" recommendation
    SIMILARITY_THRESHOLD = 0.5 
    
    # Actual true pairs (based on exact Jaccard)
    actual_relevant_pairs = {pair for pair, sim in exact_pairs_dict.items() if sim >= SIMILARITY_THRESHOLD}
    
    for pair in lsh_pairs:
        if pair in actual_relevant_pairs:
            true_positives += 1
        else:
            false_positives += 1
            
    # Calculate False Negatives
    false_negatives = len(actual_relevant_pairs) - true_positives

    # Compute Metrics
    precision = true_positives / len(lsh_pairs) if lsh_pairs else 0.0
    recall = true_positives / len(actual_relevant_pairs) if actual_relevant_pairs else 0.0
    
    print(f"Actual Similar Pairs (Jaccard >= {SIMILARITY_THRESHOLD}): {len(actual_relevant_pairs)}")
    print(f"LSH Candidate Pairs: {len(lsh_pairs)}")
    print(f"True Positives: {true_positives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    return precision, recall


def main():
    spark = get_spark_session()
    
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = spark.read.parquet(DATA_PATH).select("video_id", "text_clean")
    
    # Filter out empty texts to prevent errors
    df = df.filter(col("text_clean") != "")
    
    # 2. Apply Shingling
    print("Generating shingles...")
    df_shingled = df.withColumn("shingles", shingle_udf(col("text_clean")))
    
    # 3. Compute Exact Ground Truth
    print("Computing exact Jaccard similarity (Ground Truth)...")
    # Since we only have 100 videos, we can cross join them to get all pairs
    # (100 * 99 / 2 = 4,950 pairs)
    pairs_df = df_shingled.alias("a").crossJoin(df_shingled.alias("b")) \
        .filter("a.video_id < b.video_id") \
        .select(
            col("a.video_id").alias("id1"),
            col("b.video_id").alias("id2"),
            col("a.shingles").alias("shingles1"),
            col("b.shingles").alias("shingles2")
        )
    
    # Compute Jaccard and store in a dictionary {(id1, id2): similarity}
    exact_pairs_list = pairs_df.rdd.map(
        lambda row: ((row['id1'], row['id2']), exact_jaccard(row['shingles1'], row['shingles2']))
    ).collect()
    exact_pairs_dict = dict(exact_pairs_list)
    
    # 4. Parameter Sweep
    # We keep total hashes at 100, and vary how many bands/rows we use.
    sweep_configs = [
        (10, 10),  # b=10, r=10 -> threshold ~ 1/10^(1/10) = 0.79
        (20, 5),   # b=20, r=5  -> threshold ~ 0.55
        (50, 2),   # b=50, r=2  -> threshold ~ 0.30
        (100, 1)   # b=100, r=1 -> threshold ~ 0.01 (will generate tons of candidates)
    ]
    
    results = []
    for bands, rows in sweep_configs:
        if bands * rows != NUM_HASHES:
            print(f"Skipping (bands={bands}, rows={rows}) as it doesn't equal NUM_HASHES ({NUM_HASHES})")
            continue
        p, r = evaluate_parameter_sweep(df_shingled, exact_pairs_dict, bands, rows)
        results.append((bands, rows, p, r))
        
    print("\n=== Final Sweep Results ===")
    print(f"{'Bands':<10} | {'Rows':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 45)
    for bands, rows, p, r in results:
        print(f"{bands:<10} | {rows:<10} | {p:.4f}     | {r:.4f}")

if __name__ == "__main__":
    main()