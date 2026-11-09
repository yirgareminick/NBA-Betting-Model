import hashlib
import logging

import pytest
import yaml

from scripts.ci.check_model_metadata import check_metadata

MODEL_BYTES = b"model"
MODEL_SHA = hashlib.sha256(MODEL_BYTES).hexdigest()


def write_metadata(models_dir, name="sample", **fields):
    metadata_path = models_dir / f"{name}_metadata.yml"
    metadata_path.write_text(yaml.safe_dump(fields), encoding="utf-8")
    return metadata_path


def write_model(models_dir, name="sample.joblib", data=MODEL_BYTES):
    path = models_dir / name
    path.write_bytes(data)
    return path


def test_missing_models_dir_fails(tmp_path):
    assert check_metadata(tmp_path / "nope") == 1


def test_no_metadata_files_fails(tmp_path):
    assert check_metadata(tmp_path) == 1


def test_missing_artifact_warns_and_passes_by_default(tmp_path, caplog):
    write_metadata(tmp_path, model_file="sample.joblib", model_sha256=MODEL_SHA)

    with caplog.at_level(logging.INFO):
        assert check_metadata(tmp_path) == 0

    assert "checksum not verified" in caplog.text
    assert "0 checksum(s) verified, 1 unverified" in caplog.text


def test_missing_artifact_fails_when_required(tmp_path):
    write_metadata(tmp_path, model_file="sample.joblib", model_sha256=MODEL_SHA)

    assert check_metadata(tmp_path, require_artifacts=True) == 1


def test_relative_path_checksum_match(tmp_path):
    write_model(tmp_path)
    write_metadata(tmp_path, model_file="sample.joblib", model_sha256=MODEL_SHA)

    assert check_metadata(tmp_path, require_artifacts=True) == 0


def test_absolute_path_resolves_to_models_dir(tmp_path, caplog):
    write_model(tmp_path)
    write_metadata(
        tmp_path,
        model_file="/some/other/machine/models/sample.joblib",
        model_sha256=MODEL_SHA,
    )

    with caplog.at_level(logging.INFO):
        assert check_metadata(tmp_path, require_artifacts=True) == 0

    assert "Absolute 'model_file'" in caplog.text
    assert "1 checksum(s) verified" in caplog.text


def test_checksum_mismatch_fails(tmp_path):
    write_model(tmp_path, data=b"tampered")
    write_metadata(tmp_path, model_file="sample.joblib", model_sha256=MODEL_SHA)

    assert check_metadata(tmp_path) == 2


def test_mismatch_exit_code_not_downgraded_by_later_error(tmp_path):
    write_model(tmp_path, "a.joblib", data=b"tampered")
    write_metadata(tmp_path, name="a", model_file="a.joblib", model_sha256=MODEL_SHA)
    write_metadata(tmp_path, name="b", model_file="b.joblib")  # missing sha -> 1

    assert check_metadata(tmp_path) == 2


def test_missing_model_file_field_fails(tmp_path):
    write_metadata(tmp_path, model_sha256=MODEL_SHA)

    assert check_metadata(tmp_path) == 1


def test_missing_checksum_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("IGNORE_MISSING_SHA", raising=False)
    write_model(tmp_path)
    write_metadata(tmp_path, model_file="sample.joblib")

    assert check_metadata(tmp_path) == 1


def test_missing_checksum_ignored_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IGNORE_MISSING_SHA", "1")
    write_metadata(tmp_path, model_file="sample.joblib")

    assert check_metadata(tmp_path) == 0


@pytest.mark.parametrize("bad_sha", ["abc", "z" * 64, "a" * 63])
def test_malformed_checksum_fails_even_without_artifact(tmp_path, bad_sha):
    write_metadata(tmp_path, model_file="sample.joblib", model_sha256=bad_sha)

    assert check_metadata(tmp_path) == 1


def test_uppercase_checksum_accepted(tmp_path):
    write_model(tmp_path)
    write_metadata(tmp_path, model_file="sample.joblib", model_sha256=MODEL_SHA.upper())

    assert check_metadata(tmp_path) == 0


def test_unreadable_yaml_fails(tmp_path):
    (tmp_path / "bad_metadata.yml").write_text("model_file: [unclosed", encoding="utf-8")

    assert check_metadata(tmp_path) == 1
