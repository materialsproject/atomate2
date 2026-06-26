"""Tests for TorchSim models APIs."""
# ruff: noqa: E402

from __future__ import annotations

import pytest

ts = pytest.importorskip("torch_sim")

from atomate2.torchsim.core import pick_model
from atomate2.torchsim.schema import TorchSimModelType

try:
    from huggingface_hub.utils._auth import get_token

    HAS_HF = True
except ImportError:
    HAS_HF = False

from .conftest import (
    _SKIP_FAIRCHEM,
    _SKIP_MACE,
    _SKIP_MATTERSIM,
    _SKIP_METATOMIC,
    _SKIP_NEQUIP,
    _SKIP_ORB,
    _SKIP_SEVENNET,
)


@pytest.mark.skipif(
    not HAS_HF or get_token() is None,
    reason="Hugging Face is not installed or token is not available.",
)
@pytest.mark.skipif(_SKIP_FAIRCHEM, reason="fairchem-core is not installed.")
def test_pick_model_fairchem() -> None:
    pick_model(TorchSimModelType.FAIRCHEM, model_path="uma-s-1p1")


@pytest.mark.skipif(_SKIP_MACE, reason="mace-torch is not installed.")
def test_pick_model_mace(test_dir) -> None:
    path = f"{test_dir}/forcefields/mace/MACE.model"
    pick_model(TorchSimModelType.MACE, model_path=path)


@pytest.mark.skipif(_SKIP_MATTERSIM, reason="mattersim is not installed.")
def test_pick_model_mattersim() -> None:
    pick_model(TorchSimModelType.MATTERSIM, model_path="mattersim-v1.0.0-1m.pth")


# Upstreamed in torchsim v0.6.0
@pytest.mark.skipif(
    _SKIP_METATOMIC, reason="metatomic_torchsim or upet is not installed."
)
def test_pick_model_metatomic() -> None:
    from upet import get_upet

    # get_upet returns an instance of AtomisticModel and not a path
    # which will break the type checker but is actually supported by
    # MetatomicModel so its good enough for testing
    model = get_upet(model="pet-mad", size="s")
    pick_model(TorchSimModelType.METATOMIC, model_path=model)


# Upstreamed in torchsim v0.5.1
@pytest.mark.skipif(_SKIP_NEQUIP, reason="nequip is not installed.")
def test_pick_model_nequip(test_dir) -> None:
    path = f"{test_dir}/forcefields/nequip/nequip_ff_sr_ti_o3.nequip.pth"
    pick_model(TorchSimModelType.NEQUIPFRAMEWORK, model_path=path)


# Upstreamed in torchsim 0.6.0
@pytest.mark.skipif(_SKIP_ORB, reason="orb_models is not installed.")
def test_pick_model_orb() -> None:
    pick_model(TorchSimModelType.ORB, model_path="orb-v2")


# Upstreamed in torchsim 0.6.0
@pytest.mark.skipif(_SKIP_SEVENNET, reason="sevenn is not installed.")
def test_pick_model_sevennet() -> None:
    pick_model(TorchSimModelType.SEVENNET, model_path="7net-0")
