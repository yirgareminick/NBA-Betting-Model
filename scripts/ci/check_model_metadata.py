"""CI helper: verify model metadata contains SHA256 and matches model file.

Model binaries (``models/*.joblib``) are intentionally not tracked in Git, so a
clean CI runner only has the metadata. Policy:

* Metadata structure is always enforced: ``model_file`` must be present and
  ``model_sha256`` must be a well-formed SHA256 hex digest.
* ``model_file`` is resolved to ``models/<basename>``; absolute paths (e.g.
  from a developer's machine) are accepted but reported as a warning.
* When the referenced artifact exists, its checksum must match.
* When it does not exist, the checksum cannot be verified. This is a warning by
  default and an error with ``--require-artifacts`` or
  ``REQUIRE_MODEL_ARTIFACTS=1``. A summary line always reports how many
  checksums were verified versus skipped.

Exit codes:
 0 - all good
 1 - missing/invalid metadata, checksums, or (when required) artifacts
 2 - checksum mismatch (takes precedence over 1)
"""
import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_metadata_files(models_dir: Path):
    return sorted(models_dir.glob("*_metadata.yml")) + sorted(models_dir.glob("*_metadata.yaml"))


def check_metadata(models_dir: Path, require_artifacts: bool = False) -> int:
    if not models_dir.exists():
        logger.error("Models directory not found: %s", models_dir)
        return 1

    meta_files = find_metadata_files(models_dir)
    if not meta_files:
        logger.error("No metadata files found in %s", models_dir)
        return 1

    exit_code = 0
    verified = 0
    unverified = []

    def fail(code: int) -> None:
        nonlocal exit_code
        exit_code = max(exit_code, code)

    for mpath in meta_files:
        try:
            with open(mpath, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to read metadata %s: %s", mpath, e)
            fail(1)
            continue

        if not isinstance(meta, dict):
            logger.error("Metadata is not a mapping: %s", mpath)
            fail(1)
            continue

        model_file = meta.get('model_file')
        if not model_file:
            logger.error("Missing 'model_file' in metadata: %s", mpath)
            fail(1)
            continue

        if Path(str(model_file)).is_absolute():
            logger.warning(
                "Absolute 'model_file' in %s; resolving by filename within %s",
                mpath.name,
                models_dir,
            )
        model_path = models_dir / Path(str(model_file)).name

        recorded = meta.get('model_sha256')
        if not recorded:
            if os.getenv('IGNORE_MISSING_SHA', '0') == '1':
                logger.warning("No 'model_sha256' in metadata (ignored): %s", mpath)
                continue
            logger.error("No 'model_sha256' in metadata: %s", mpath)
            fail(1)
            continue

        recorded = str(recorded).strip().lower()
        if not SHA256_RE.match(recorded):
            logger.error("Malformed 'model_sha256' in %s: %r", mpath.name, recorded)
            fail(1)
            continue

        if not model_path.exists():
            unverified.append(mpath.name)
            if require_artifacts:
                logger.error("Model artifact missing for %s: %s", mpath.name, model_path.name)
                fail(1)
            else:
                logger.warning(
                    "Model artifact unavailable; checksum not verified for %s: %s",
                    mpath.name,
                    model_path.name,
                )
            continue

        actual = file_sha256(model_path)
        if actual != recorded:
            logger.error("Checksum mismatch for %s: recorded=%s actual=%s", model_path.name, recorded, actual)
            fail(2)
        else:
            verified += 1
            logger.info("OK: %s matches metadata", model_path.name)

    logger.info(
        "Summary: %d metadata file(s), %d checksum(s) verified, %d unverified (artifact unavailable)",
        len(meta_files),
        verified,
        len(unverified),
    )
    return exit_code


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        default=os.getenv("REQUIRE_MODEL_ARTIFACTS", "0") == "1",
        help="Fail when a referenced model artifact is missing (env: REQUIRE_MODEL_ARTIFACTS=1)",
    )
    args = parser.parse_args(argv)
    # scripts/ci/check_model_metadata.py -> project root is three levels up
    project_root = Path(__file__).resolve().parent.parent.parent
    return check_metadata(project_root / "models", require_artifacts=args.require_artifacts)


if __name__ == '__main__':
    sys.exit(main())
