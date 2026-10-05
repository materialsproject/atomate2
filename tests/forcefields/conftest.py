from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator, all_changes
from emmet.core.utils import get_hash_blocked

from atomate2.forcefields.jobs import ForceFieldDielectricMaker
from atomate2.forcefields.utils import MLFF, _get_pkg_version

if TYPE_CHECKING:
    from typing import Any

_INSTALLED_MLFF: dict[str, bool] = {
    mlff.name: (
        isinstance(_get_pkg_version(mlff), str) if mlff.name != "Forcefield" else False
    )
    for mlff in MLFF
}
# the MACE-Field fork installs as mace-torch, so only its model class tells them apart
try:
    _INSTALLED_MLFF["MACE_Field"] = hasattr(
        import_module("mace.modules.extensions"), "MACEField"
    )
except ImportError:
    _INSTALLED_MLFF["MACE_Field"] = False


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
    _, http_message = urllib.request.urlretrieve(file_url, local_path.name)
    if "Content-Type: text/html" in http_message:
        raise RuntimeError(f"Failed to download from: {file_url}")

    # Check MD5 to ensure file integrity
    if (file_md5 := get_hash_blocked(local_path.name, hasher=hashlib.md5())) != ref_md5:
        raise RuntimeError(f"MD5 mismatch: {file_md5} != {ref_md5}")
    yield Path(local_path.name)
    local_path.close()


class FakeDielectricCalculator(Calculator):
    """Born charges of +2 and -2 and a susceptibility of 3.

    The atoms of the first species get +2, all others -2. Both results are
    flattened, as MACE-Field returns them.
    """

    implemented_properties = ("energy", "becs", "polarizability")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        sign = np.where(self.atoms.numbers == self.atoms.numbers[0], 1.0, -1.0)
        self.results = {
            "energy": 0.0,
            "becs": (2 * sign[:, None, None] * np.eye(3)).reshape(-1, 9),
            "polarizability": (3 * np.eye(3)).reshape(9),
        }


@pytest.fixture
def fake_dielectric_calculator(monkeypatch):
    monkeypatch.setattr(
        ForceFieldDielectricMaker,
        "_get_calculator",
        lambda _self: FakeDielectricCalculator(),
    )
