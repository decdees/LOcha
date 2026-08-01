"""Local-only model resolution.

Runtime never downloads weights. Provisioning is an explicit operator action;
startup fails if the requested snapshot is absent from the Hugging Face cache.
"""

from __future__ import annotations

from pathlib import Path


class LocalModelMissing(RuntimeError):
    """The configured model has not been provisioned on this Mac."""


def resolve_cached_model(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo, local_files_only=True))
    except Exception as exc:
        raise LocalModelMissing(
            f"model {repo!r} is not in the local Hugging Face cache; "
            "download it explicitly during setup before starting Ocha"
        ) from exc
