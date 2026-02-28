from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from emmet.core.utils import get_hash_blocked

from atomate2.forcefields.utils import MLFF, ase_calculator

if TYPE_CHECKING:
    from typing import Any

_INSTALLED_MLFF: dict[str, bool] = {MLFF.Forcefield.name: False}
for mlff in (x for x in MLFF if x.value != "Forcefield"):
    try:
        _ = ase_calculator(mlff)
        _INSTALLED_MLFF[mlff.name] = True
    except (ImportError, ValueError):
        _INSTALLED_MLFF[mlff.name] = False
    except Exception:  # noqa: BLE001
        # Some calculators, like GAP, require extra potential files
        # Generally, thesea re
        _INSTALLED_MLFF[mlff.name] = True


def mlff_is_installed(mlff: str | MLFF) -> bool:
    if not isinstance(mlff, str | MLFF):
        raise TypeError(f"Unknown `MLFF = {MLFF}` type, {type(mlff)}")

    ff: str = (MLFF(mlff.split("MLFF.", 1)[-1]) if isinstance(mlff, str) else mlff).name
    return _INSTALLED_MLFF[ff]


def pytest_runtest_setup(item: Any) -> None:
    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)
    # For consistent performance across hardware, explicitly set device to CPU
    torch.set_default_device("cpu")


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
