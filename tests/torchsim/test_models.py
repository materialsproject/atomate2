from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from atomate2.torchsim.core import pick_model
from atomate2.torchsim.schema import TorchSimModelType

try:
    from huggingface_hub.utils._auth import get_token

    HAS_HF = True
except ImportError:
    HAS_HF = False


def _is_backend_missing(package_names: str | list[str]) -> bool:
    if isinstance(package_names, str):
        package_names = [package_names]

    missing = False
    try:
        for pkg_name in package_names:
            importlib.metadata.distribution(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        missing = True

    return missing


_SKIP_FAIRCHEM = _is_backend_missing("fairchem-core")
_SKIP_MACE = _is_backend_missing("mace-torch")
_SKIP_MATTERSIM = _is_backend_missing("mattersim")
_SKIP_METATOMIC = _is_backend_missing(["metatomic_torchsim", "upet"])
_SKIP_NEQUIP = _is_backend_missing("nequip")
_SKIP_ORB = _is_backend_missing("orb_models")
_SKIP_SEVENNET = _is_backend_missing("sevenn")


@pytest.mark.skipif(
    not HAS_HF or get_token() is None,
    reason="Hugging Face is not installed or token is not available.",
)
@pytest.mark.skipif(_SKIP_FAIRCHEM, reason="FAIRCHEM is not installed.")
def test_pick_model_fairchem() -> None:
    pick_model(TorchSimModelType.FAIRCHEM, model_path="uma-s-1p1")


@pytest.mark.skipif(_SKIP_MACE, reason="MACE is not installed.")
def test_pick_model_mace() -> None:
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    path = download_mace_mp_checkpoint("small")
    pick_model(TorchSimModelType.MACE, model_path=path)


@pytest.mark.skipif(_SKIP_MATTERSIM, reason="MATTERSIM is not installed.")
def test_pick_model_mattersim() -> None:
    pick_model(TorchSimModelType.MATTERSIM, model_path="mattersim-v1.0.0-1m.pth")


# Upstreamed in torchsim v0.6.0
@pytest.mark.skipif(_SKIP_METATOMIC, reason="METATOMIC or UPET is not installed.")
def test_pick_model_metatomic() -> None:
    from upet import get_upet

    # get_upet returns an instance of AtomisticModel and not a path
    # which will break the type checker but is actually supported by
    # MetatomicModel so its good enough for testing
    model = get_upet(model="pet-mad", size="s")
    pick_model(TorchSimModelType.METATOMIC, model_path=model)


# Upstreamed in torchsim v0.5.1
@pytest.mark.skipif(_SKIP_NEQUIP, reason="NEQUIP is not installed.")
def test_pick_model_nequip() -> None:
    path = (
        Path(__file__).parent.parent
        / "test_data"
        / "forcefields"
        / "nequip"
        / "nequip_ff_sr_ti_o3.nequip.pth"
    )
    pick_model(TorchSimModelType.NEQUIPFRAMEWORK, model_path=path)


# Upstreamed in torchsim 0.6.0
@pytest.mark.skipif(_SKIP_ORB, reason="ORB is not installed.")
def test_pick_model_orb() -> None:
    pick_model(TorchSimModelType.ORB, model_path="orb-v2")


# Upstreamed in torchsim 0.6.0
@pytest.mark.skipif(_SKIP_SEVENNET, reason="SEVENN is not installed.")
def test_pick_model_sevennet() -> None:
    pick_model(TorchSimModelType.SEVENNET, model_path="7net-0")
