# %%
# ReviewLens M3 - Prepare canonical product reviews

import json
import re
import html
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/amazon"

SOURCE = DATA_DIR / "acer_aspire3_candidate_reviews.jsonl"
OUTPUT = DATA_DIR / "acer_aspire5_reviews_clean.jsonl"

TARGET_ASIN = "B07RF1XD36"
PRODUCT_NAME = "Acer Aspire 5 Slim Laptop"
PRODUCT_MODEL = "A515-43-R19L"


# %%
# Cell 1: Clean HTML while preserving paragraph boundaries

class ReviewHTMLCleaner(HTMLParser):

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "li", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "li", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def clean_review(text):
    parser = ReviewHTMLCleaner()
    parser.feed(str(text))
    parser.close()

    cleaned = html.unescape("".join(parser.parts))

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)

    return cleaned.strip()


# %%
# Cell 2: Verify ASIN and prepare canonical records

with open(SOURCE, encoding="utf-8") as f:
    reviews = [json.loads(line) for line in f if line.strip()]

assert len(reviews) == 189

assert all(
    r["asin"] == TARGET_ASIN
    for r in reviews
), "Found reviews from a different child ASIN!"

processed = []

for r in reviews:

    processed.append({
        "review_id": r["review_id"],
        "product_name": PRODUCT_NAME,
        "product_model": PRODUCT_MODEL,
        "asin": r["asin"],
        "parent_asin": r["parent_asin"],
        "rating": r["rating"],
        "title": r.get("title"),
        "raw_text": r["text"],
        "clean_text": clean_review(r["text"]),
        "timestamp": r.get("timestamp"),
        "verified_purchase": r.get("verified_purchase"),
        "helpful_vote": r.get("helpful_vote")
    })


# %%
# Cell 3: Quality checks

assert all(r["clean_text"] for r in processed)

review_ids = [r["review_id"] for r in processed]
assert len(set(review_ids)) == len(review_ids)

print("=== CANONICAL DATASET ===")
print("Product:", PRODUCT_NAME, PRODUCT_MODEL)
print("Child ASIN:", TARGET_ASIN)
print("Reviews:", len(processed))

print("\nRating distribution:")
print(Counter(r["rating"] for r in processed))

print("\nFirst cleaned review:")
print(processed[0]["clean_text"][:600])


# %%
# Cell 4: Save

with open(OUTPUT, "w", encoding="utf-8") as f:
    for record in processed:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nSaved:", OUTPUT)
print("Data preparation completed!")