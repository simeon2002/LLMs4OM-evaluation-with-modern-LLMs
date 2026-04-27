import os
os.environ["HF_HOME"] = os.path.join(os.environ["VSC_SCRATCH"], ".cache/huggingface")

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
print(f"HF_HOME: {os.environ['HF_HOME']}")
print(f"HF token set: {bool(token)}, preview: {token[:4] if token else 'MISSING'}...")
print("Starting download of google/gemma-4-26B-A4B-it ...")

snapshot_download(
    repo_id="google/gemma-4-26B-A4B-it",
    cache_dir=os.environ["HF_HOME"],
    token=token,
    resume_download=True,
)

print("Download complete.")
