
# %%
# ReviewLens M3 - Refine laptop candidates
# Uses saved metadata; no additional download required

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/amazon"

INPUT_FILE = DATA_DIR / "laptop_candidates_preview.jsonl"
OUTPUT_FILE = DATA_DIR / "laptop_candidates_refined.jsonl"

with open(INPUT_FILE, encoding="utf-8") as f:
    candidates = [
        json.loads(line)
        for line in f if line.strip()
    ]

print("Original candidates:", len(candidates))


# %%
# Cell 2: Define filtering rules

# Common laptop product families
DEVICE_FAMILIES = re.compile(
    r"\b(?:"
    r"acer\s+(?:aspire|swift|nitro|predator)|"
    r"dell\s+(?:inspiron|latitude|xps|vostro)|"
    r"lenovo\s+(?:thinkpad|ideapad|yoga|legion)|"
    r"hp\s+(?:pavilion|envy|spectre|elitebook|probook|victus|omen)|"
    r"asus\s+(?:vivobook|zenbook|rog|tuf)|"
    r"macbook\s+(?:air|pro)|"
    r"microsoft\s+surface\s+laptop|"
    r"chromebook"
    r")\b",
    re.IGNORECASE
)

# General device words
DEVICE_WORDS = re.compile(
    r"\b(?:laptop|notebook|chromebook)\b",
    re.IGNORECASE
)

# Hardware specifications typical of complete computers
HARDWARE_SPECS = re.compile(
    r"\b(?:"
    r"intel|ryzen|celeron|pentium|snapdragon|"
    r"processor|core\s+i[3579]|"
    r"ram|ssd|windows\s+1[01]|chrome\s*os"
    r")\b",
    re.IGNORECASE
)

# Exclude obvious accessories and unrelated products
ACCESSORIES = re.compile(
    r"\b(?:"
    r"backpack|briefcase|bag|sleeve|case|cover|"
    r"skin|charger|adapter|cable|mouse|"
    r"earbuds?|headphones?|"
    r"screen protector|stand|dock|"
    r"external hard drive|tablet|cooling pad"
    r")\b|compatible with|designed for",
    re.IGNORECASE
)


def is_laptop_device(title):
    if ACCESSORIES.search(title):
        return False

    # Recognizable laptop series
    if DEVICE_FAMILIES.search(title):
        return True

    # Generic laptop title with hardware specifications
    return bool(
        DEVICE_WORDS.search(title)
        and HARDWARE_SPECS.search(title)
    )


# %%
# Cell 3: Apply filters

refined = []
seen_asins = set()

for product in candidates:
    title = product.get("title") or ""
    asin = product.get("parent_asin")

    if not asin or not is_laptop_device(title):
        continue

    if asin in seen_asins:
        continue

    seen_asins.add(asin)
    refined.append(product)

refined.sort(
    key=lambda x: int(x.get("rating_number") or 0),
    reverse=True
)

print("\n=== REFINED LAPTOP CANDIDATES ===")
print("Original:", len(candidates))
print("Refined:", len(refined))

for i, p in enumerate(refined[:20], 1):
    print(f"\n{i}. {p['title']}")
    print("Parent ASIN:", p["parent_asin"])
    print("Rating count:", p.get("rating_number"))
    print("Average rating:", p.get("average_rating"))


# %%
# Cell 4: Save results

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in refined:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nSaved:", OUTPUT_FILE)
