from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from emmet.core.utils import get_hash_blocked

if TYPE_CHECKING:
    from typing import Any


def pytest_runtest_setup(item: Any) -> None:
    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)
    # For consistent performance across hardware, explicitly set device to CPU
    torch.set_default_device("cpu")

    # To use CHGNet and M3GNet, need dgl
    os.environ["MATGL_BACKEND"] = "DGL"


@pytest.fixture(scope="session", autouse=True)
def get_deepmd_pretrained_model_path(test_dir: Path) -> Path:
    # Download DeepMD pretrained model from GitHub
    file_url = "https://raw.github.com/sliutheorygroup/UniPero/main/model/graph.pb"
    local_path = tempfile.NamedTemporaryFile(suffix=".pb")  # noqa : SIM115
    ref_md5 = "2814ae7f2eb1c605dd78f2964187de40"
    _, http_message = urllib.request.urlretrieve(file_url, local_path.name)  # noqa: S310
    if "Content-Type: text/html" in http_message:
        raise RuntimeError(f"Failed to download from: {file_url}")

    # Check MD5 to ensure file integrity
    if (file_md5 := get_hash_blocked(local_path.name, hasher=hashlib.md5())) != ref_md5:
        raise RuntimeError(f"MD5 mismatch: {file_md5} != {ref_md5}")
    yield Path(local_path.name)
    local_path.close()
