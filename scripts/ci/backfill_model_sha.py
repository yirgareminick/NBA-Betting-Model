"""Backfill missing model_sha256 values in model metadata files.

Usage:
  python scripts/ci/backfill_model_sha.py

This script scans the `models/` directory for metadata files and, when a
`model_sha256` key is missing but the model file exists, computes the SHA256
checksum and writes it into the metadata YAML file.
"""
from pathlib import Path
import hashlib
import yaml
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_metadata_files(models_dir: Path):
    return sorted(models_dir.glob("*_metadata.yml")) + sorted(models_dir.glob("*_metadata.yaml"))


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    if not models_dir.exists():
        logger.error("Models directory not found: %s", models_dir)
        return 1

    meta_files = find_metadata_files(models_dir)
    if not meta_files:
        logger.info("No metadata files found to backfill")
        return 0

    changed = []
    for mpath in meta_files:
        try:
            with open(mpath, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to read %s: %s", mpath, e)
            continue

        if meta.get('model_sha256'):
            logger.info("Already has checksum: %s", mpath.name)
            continue

        model_file = meta.get('model_file')
        if not model_file:
            logger.warning("No 'model_file' in metadata: %s", mpath.name)
            continue

        # Resolve model path relative to models_dir when necessary
        model_path = Path(model_file)
        if not model_path.is_absolute():
            model_path = models_dir / model_path.name

        if not model_path.exists():
            logger.warning("Model file not found for metadata %s: %s", mpath.name, model_path)
            continue

        try:
            sha = file_sha256(model_path)
            meta['model_sha256'] = sha
            with open(mpath, 'w', encoding='utf-8') as f:
                yaml.dump(meta, f, default_flow_style=False)
            logger.info("Backfilled %s with checksum %s", mpath.name, sha)
            changed.append(mpath)
        except Exception as e:
            logger.warning("Failed to compute/write checksum for %s: %s", mpath.name, e)

    if changed:
        logger.info("Updated %d metadata files", len(changed))
    else:
        logger.info("No metadata files were updated")

    return 0


if __name__ == '__main__':
    sys.exit(main())
