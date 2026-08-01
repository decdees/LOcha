from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import ocha.models as models


def test_local_model_resolver_is_application_code() -> None:
    assert importlib.util.find_spec("ocha.models") is not None


def test_resolver_uses_the_hugging_face_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool]] = []

    def fake_snapshot(repo: str, *, local_files_only: bool) -> str:
        seen.append((repo, local_files_only))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    assert models.resolve_cached_model("example/model") == tmp_path
    assert seen == [("example/model", True)]
