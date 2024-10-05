from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests
import torch

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
    local_path = test_dir / "forcefields" / "deepmd" / "graph.pb"
    response = requests.get(file_url, timeout=10)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            file.write(response.content)
    else:
        raise requests.RequestException(f"Failed to download: {file_url}")
