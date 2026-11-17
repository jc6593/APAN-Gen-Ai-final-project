# %%
from huggingface_hub import HfApi

repo_id = "McAuley-Lab/Amazon-Reviews-2023"

files = HfApi().list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"
)

candidates = [
    p for p in files
    if "electronics" in p.lower()
    and ("meta" in p.lower() or "review" in p.lower())
]

print("Electronics candidate files:")
for path in candidates:
    print(path)