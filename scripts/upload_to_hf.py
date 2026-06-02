import os
import re
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "Sylvan-Vale-Moon/ReTrace-Bench"
FOLDER_PATH = "release/huggingface/ReTrace-Bench"
KEEP_REMOTE_ONLY = {".gitattributes"}


def local_files(folder_path):
    root = Path(folder_path)
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def verify_readme_split_paths(folder_path):
    """Ensure every split path declared in the dataset card exists locally.

    Uploading a README that references a missing scenarios.jsonl makes the
    Hugging Face dataset viewer fail with "dataset viewer is not available for
    this split", so we fail fast before pushing.
    """
    readme = Path(folder_path) / "README.md"
    if not readme.exists():
        raise FileNotFoundError(f"Dataset card not found: {readme}")
    text = readme.read_text(encoding="utf-8")
    front_matter = text.split("---", 2)
    frontmatter = front_matter[1] if len(front_matter) >= 3 else text
    declared_paths = re.findall(r"^\s*path:\s*(\S+)\s*$", frontmatter, flags=re.MULTILINE)
    missing = [p for p in declared_paths if not (Path(folder_path) / p).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset card references split files that do not exist in "
            f"'{folder_path}': {', '.join(missing)}. "
            "Re-run scripts/package_hf_retrace_bench.py before uploading."
        )
    print(f"Verified {len(declared_paths)} declared split path(s) exist locally.")


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set. Please provide it before running.")

    verify_readme_split_paths(FOLDER_PATH)

    print(f"Initializing HfApi with token for repository: {REPO_ID}...")
    api = HfApi(token=token)

    expected = local_files(FOLDER_PATH)
    remote = set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    stale_files = sorted(remote - expected - KEEP_REMOTE_ONLY)
    for path_in_repo in stale_files:
        print(f"Deleting stale remote file: {path_in_repo}")
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Remove stale release file {path_in_repo}",
        )

    print(f"Uploading folder '{FOLDER_PATH}' to dataset repository...")
    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload ReTrace-Bench Hugging Face release package",
    )
    print("Upload completed successfully!")

if __name__ == "__main__":
    main()
