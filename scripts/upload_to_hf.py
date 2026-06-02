import os
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


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set. Please provide it before running.")

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
