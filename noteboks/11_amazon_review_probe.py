
# %%
# ReviewLens M3 - Probe Amazon reviews for target laptops
# Bounded remote scan; does not download the whole file

import json
from pathlib import Path
from collections import Counter
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data/processed/amazon"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_PATH = (
    "datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "raw/review_categories/Electronics.jsonl"
)

# Five laptop candidates from our metadata exploration
TARGET_PRODUCTS = {
    "B0C2ZMJW53": "Acer Aspire 3",
    "B081FY9HRR": "MacBook Pro 2019",
    "B0B5DFFJ6S": "HP Chromebook 14",
    "B0B4F9H4DF": "MacBook Air M2",
    "B096X5YYZM": "HP 14 Ryzen 5"
}

MAX_SCAN = 2_000_000

OUTPUT_FILE = (
    OUTPUT_DIR / "target_reviews_probe_2m.jsonl"
)

# %%
# Stream and filter reviews

fs = HfFileSystem()

matched_counts = Counter()
usable_counts = Counter()

first_10k_parents = Counter()

scanned = 0

print("Starting bounded review scan...")
print("Maximum records:", MAX_SCAN)

with fs.open(REVIEW_PATH, "r") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_no, line in enumerate(fin, start=1):

        if line_no > MAX_SCAN:
            break

        scanned += 1
        record = json.loads(line)

        parent_asin = record.get("parent_asin")

        # Examine how product IDs are distributed
        if line_no <= 10_000 and parent_asin:
            first_10k_parents[parent_asin] += 1

        if parent_asin in TARGET_PRODUCTS:

            matched_counts[parent_asin] += 1

            text = str(record.get("text") or "").strip()

            # Keep a basic usable-text threshold
            if len(text) >= 40:

                usable_counts[parent_asin] += 1

                clean_record = {
                    "review_id": f"electronics_{line_no}",
                    "parent_asin": parent_asin,
                    "product_name": TARGET_PRODUCTS[parent_asin],
                    "asin": record.get("asin"),
                    "rating": record.get("rating"),
                    "title": record.get("title"),
                    "text": text,
                    "timestamp": record.get("timestamp"),
                    "verified_purchase": record.get(
                        "verified_purchase"
                    ),
                    "helpful_vote": record.get("helpful_vote")
                }

                fout.write(
                    json.dumps(clean_record, ensure_ascii=False)
                    + "\n"
                )

        if scanned % 25_000 == 0:
            print(f"Scanned: {scanned:,} records")

# %%
# Summarize the preview

print("\n=== SCAN SUMMARY ===")
print("Total scanned:", scanned)
print("Distinct product IDs in first 10k:",
      len(first_10k_parents))

print("\nMost common IDs in first 10k:")
for asin, count in first_10k_parents.most_common(5):
    print(asin, count)

print("\n=== TARGET PRODUCT MATCHES ===")

for asin, name in TARGET_PRODUCTS.items():
    print(f"\n{name}")
    print("Parent ASIN:", asin)
    print("All matching records:", matched_counts[asin])
    print("Usable text reviews:", usable_counts[asin])

print("\nSaved preview to:", OUTPUT_FILE)
print("Probe completed!")
