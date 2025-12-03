from __future__ import annotations

import hashlib
import urllib.request
from typing import TYPE_CHECKING

import pytest
import torch
from emmet.core.utils import get_hash_blocked

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def pytest_runtest_setup(item: Any) -> None:
    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)
    # For consistent performance across hardware, explicitly set device to CPU
    torch.set_default_device("cpu")


@pytest.fixture(scope="session", autouse=True)
def download_deepmd_pretrained_model(test_dir: Path) -> None:
    # Download DeepMD pretrained model from GitHub
    file_url = "https://raw.github.com/sliutheorygroup/UniPero/main/model/graph.pb"
    local_path = test_dir / "forcefields" / "deepmd_graph.pb"
    ref_md5 = "2814ae7f2eb1c605dd78f2964187de40"
    _, http_message = urllib.request.urlretrieve(file_url, local_path)  # noqa: S310
    if "Content-Type: text/html" in http_message:
        raise RuntimeError(f"Failed to download from: {file_url}")

    # Check MD5 to ensure file integrity
    if (file_md5 := get_hash_blocked(local_path, hasher=hashlib.md5())) != ref_md5:
        raise RuntimeError(f"MD5 mismatch: {file_md5} != {ref_md5}")
