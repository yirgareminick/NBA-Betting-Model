"""Backfill missing `model_sha256` entries in model metadata YAML files.

For each metadata file under `models/` missing `model_sha256`, compute the
SHA256 of the referenced model file and write it into the metadata YAML.

This script makes a `.bak` copy of each metadata file before updating.
"""
from pathlib import Path
import hashlib
import yaml
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_metadata(models_dir: Path):
    return sorted(models_dir.glob("*_metadata.yml")) + sorted(models_dir.glob("*_metadata.yaml"))


def backfill(models_dir: Path) -> int:
    meta_files = find_metadata(models_dir)
    if not meta_files:
        logger.error("No metadata files found in %s", models_dir)
        return 1

    updated = 0
    for mpath in meta_files:
        try:
            meta = yaml.safe_load(mpath.read_text(encoding='utf-8')) or {}
        except Exception as e:
            logger.error("Failed to read %s: %s", mpath, e)
            continue

        if meta.get('model_sha256'):
            logger.info("Already has checksum: %s", mpath.name)
            continue

        model_file = meta.get('model_file')
        if not model_file:
            logger.warning("No model_file field in %s; skipping", mpath.name)
            continue

        model_path = Path(model_file)
        if not model_path.is_absolute():
            model_path = models_dir / model_path.name

        if not model_path.exists():
            logger.warning("Model file missing for %s: %s", mpath.name, model_path)
            continue

        sha = file_sha256(model_path)

        # backup
        bak = mpath.with_suffix(mpath.suffix + '.bak')
        shutil.copy2(mpath, bak)

        meta['model_sha256'] = sha
        mpath.write_text(yaml.dump(meta, default_flow_style=False), encoding='utf-8')
        logger.info("Updated %s with SHA256 %s", mpath.name, sha)
        updated += 1

    logger.info("Done. Updated %s metadata files.", updated)
    return 0


if __name__ == '__main__':
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    models_dir = repo_root / 'models'
    code = backfill(models_dir)
    sys.exit(code)
