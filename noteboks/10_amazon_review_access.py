
# %%
# ReviewLens M3 - Test Amazon review streaming
# Read only a few records without downloading the full file

import json
from huggingface_hub import HfFileSystem

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

REVIEW_PATH = (
    f"datasets/{REPO_ID}/"
    "raw/review_categories/Electronics.jsonl"
)

fs = HfFileSystem()

# %%
# Preview the first 3 reviews

print("Connecting to Amazon Electronics reviews...")

with fs.open(REVIEW_PATH, "r") as f:

    for i in range(3):
        line = f.readline()

        if not line:
            break

        record = json.loads(line)

        print(f"\n=== REVIEW {i + 1} ===")
        print("Available fields:", list(record.keys()))
        print("Parent ASIN:", record.get("parent_asin"))
        print("ASIN:", record.get("asin"))
        print("Rating:", record.get("rating"))
        print("Title:", record.get("title"))
        print("Review text:", str(record.get("text") or "")[:300])

print("\nStreaming preview completed!")
