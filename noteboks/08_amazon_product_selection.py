
# %%
# ReviewLens M3 - Find laptop product candidates

from pathlib import Path
from datasets import load_dataset
import json

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

# Start with the first of 10 metadata shards
PARQUET_FILE = (
    f"hf://datasets/{REPO_ID}/"
    "raw_meta_Electronics/full-00000-of-00010.parquet"
)

print("Connecting to Amazon Electronics metadata...")

metadata = load_dataset(
    "parquet",
    data_files={"train": PARQUET_FILE},
    split="train",
    streaming=True,
    columns=[
        "title",
        "parent_asin",
        "average_rating",
        "rating_number",
        "main_category"
    ]
)

# %%
# Preview the actual data structure

first_record = next(iter(metadata))

print("\n=== FIRST METADATA RECORD ===")
print(json.dumps(first_record, indent=2, ensure_ascii=False))


# %%
# Search for laptop candidates in a limited sample

KEYWORDS = [
    "laptop",
    "macbook",
    "chromebook",
    "thinkpad",
    "ideapad",
    "vivobook",
    "zenbook",
    "ultrabook",
    "notebook pc",
    "notebook computer"
]

EXCLUDE = [
    "sleeve",
    "charger",
    "adapter",
    "screen protector",
    "replacement",
    "keyboard cover",
    "laptop bag",
    "laptop stand",
    "laptop case"
]

MAX_SCAN = 20000
candidates = []
scanned = 0

for row in metadata:

    if scanned >= MAX_SCAN:
        break

    scanned += 1

    title = str(row.get("title") or "")
    title_lower = title.lower()

    is_laptop = any(k in title_lower for k in KEYWORDS)
    is_accessory = any(k in title_lower for k in EXCLUDE)

    if is_laptop and not is_accessory:
        candidates.append({
            "parent_asin": row.get("parent_asin"),
            "title": title,
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number")
        })

# Sort using product-page rating counts
candidates.sort(
    key=lambda r: r["rating_number"] or 0,
    reverse=True
)

print("\n=== LAPTOP CANDIDATES ===")
print("Metadata records scanned:", scanned)
print("Candidate products found:", len(candidates))

for i, product in enumerate(candidates[:15], 1):

    print(f"\n{i}. {product['title'][:110]}")
    print("   Parent ASIN:", product["parent_asin"])
    print("   Average rating:", product["average_rating"])
    print("   Rating number:", product["rating_number"])

# %%
# Save candidates locally

OUTPUT_DIR = ROOT / "data" / "processed" / "amazon"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "laptop_candidates_preview.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in candidates:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nSaved:", output_file)
