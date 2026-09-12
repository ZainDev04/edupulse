"""Assemble and push the API to a Hugging Face Space (Docker SDK).

Usage::

    python scripts/deploy_space.py --repo <user>/edupulse-api

Requires a prior ``hf auth login`` (or ``HF_TOKEN`` in the environment). The
upload holds only what serving needs: the package, the API, the raw CSV, the
registered models including their joblib pipelines, and the Space files under
``deploy/space``. Nothing is trained on the Space, so the served numbers match
the committed model cards.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ROOT / "deploy" / "space"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")


def assemble(target: Path) -> None:
    for name in ("Dockerfile", "README.md", "requirements.txt"):
        shutil.copy2(SPACE_FILES / name, target / name)
    shutil.copy2(ROOT / "pyproject.toml", target / "pyproject.toml")
    shutil.copytree(ROOT / "src", target / "src", ignore=IGNORE)
    shutil.copytree(ROOT / "api", target / "api", ignore=IGNORE)
    (target / "data").mkdir()
    shutil.copytree(ROOT / "data" / "raw", target / "data" / "raw")
    shutil.copytree(ROOT / "models", target / "models", ignore=shutil.ignore_patterns("tuning_history.csv"))
    missing = [p for p in (target / "models").glob("*/*") if p.is_dir() and not (p / "pipeline.joblib").exists()]
    if missing:
        raise SystemExit(f"Model versions without a pipeline.joblib (run `edupulse train --all` first): {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="Space id, e.g. zaindev04/edupulse-api")
    ap.add_argument("--dry-run", action="store_true", help="Assemble only, print the file list")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="edupulse-space-") as tmp:
        target = Path(tmp) / "space"
        target.mkdir()
        assemble(target)
        files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        total = sum((target / f).stat().st_size for f in files) / 1e6
        print(f"{len(files)} files, {total:.1f} MB")
        if args.dry_run:
            print("\n".join(files))
            return

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            folder_path=str(target),
            repo_id=args.repo,
            repo_type="space",
            commit_message="Deploy EduPulse API",
            delete_patterns=["**"],
        )
        print(f"Pushed. Build logs and URL: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
