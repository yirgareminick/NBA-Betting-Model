"""CI helper: verify model metadata contains SHA256 and matches model file.

Exit codes:
 0 - all good
 1 - missing metadata files or checksums
 2 - checksum mismatch
"""
import sys
from pathlib import Path
import hashlib
import yaml
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_metadata_files(models_dir: Path):
    return list(models_dir.glob("*_metadata.yml")) + list(models_dir.glob("*_metadata.yaml"))


def main():
    # scripts/ci/check_model_metadata.py -> project root is three levels up
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    if not models_dir.exists():
        logger.error("Models directory not found: %s", models_dir)
        return 1

    meta_files = find_metadata_files(models_dir)
    if not meta_files:
        logger.error("No metadata files found in %s", models_dir)
        return 1

    exit_code = 0
    for mpath in meta_files:
        try:
            with open(mpath, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to read metadata %s: %s", mpath, e)
            exit_code = 1
            continue

        model_file = meta.get('model_file')
        if not model_file:
            logger.error("Missing 'model_file' in metadata: %s", mpath)
            exit_code = 1
            continue

        model_path = (models_dir / Path(model_file).name) if not Path(model_file).is_absolute() else Path(model_file)
        if not model_path.exists():
            logger.error("Model file referenced by %s not found: %s", mpath, model_path)
            exit_code = 1
            continue

        recorded = meta.get('model_sha256')
        if not recorded:
            if os.getenv('IGNORE_MISSING_SHA', '0') == '1':
                logger.warning("No 'model_sha256' in metadata (ignored): %s", mpath)
                continue
            logger.error("No 'model_sha256' in metadata: %s", mpath)
            exit_code = 1
            continue

        actual = file_sha256(model_path)
        if actual != recorded:
            logger.error("Checksum mismatch for %s: recorded=%s actual=%s", model_path.name, recorded, actual)
            exit_code = 2
        else:
            logger.info("OK: %s matches metadata", model_path.name)

    return exit_code


if __name__ == '__main__':
    code = main()
    sys.exit(code)
