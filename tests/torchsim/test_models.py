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
    _SKIP_NVALCHEMIOPS,
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


def _dummy_d3_params(max_z: int = 18):
    """Build a D3Parameters instance with arbitrary (non-physical) values."""
    import torch
    from torch_sim.models.dispersion import D3Parameters

    return D3Parameters(
        rcov=torch.rand(max_z + 1, dtype=torch.float64),
        r4r2=torch.rand(max_z + 1, dtype=torch.float64),
        c6ab=torch.rand(max_z + 1, max_z + 1, 5, 5, dtype=torch.float64),
        cn_ref=torch.rand(max_z + 1, max_z + 1, 5, 5, dtype=torch.float64),
    )


@pytest.mark.skipif(_SKIP_NVALCHEMIOPS, reason="nvalchemiops is not installed.")
def test_pick_model_dispersion() -> None:
    """A D3 dispersion correction should be summed with the base model.

    The base model's cutoff must be preserved, while the D3 model should fall
    back to its own default cutoff rather than inheriting the base model's.
    """
    from torch_sim.models.dispersion import D3DispersionModel
    from torch_sim.models.interface import SumModel
    from torch_sim.models.lennard_jones import LennardJonesModel

    model = pick_model(
        TorchSimModelType.LENNARD_JONES,
        model_path="",
        sigma=3.405,
        epsilon=0.0104,
        cutoff=6.0,
        dispersion=True,
        a1=0.4289,
        a2=4.4407,
        s8=0.7875,
        d3_params=_dummy_d3_params(),
    )

    assert isinstance(model, SumModel)
    base_model, d3_model = model.models
    assert isinstance(base_model, LennardJonesModel)
    assert isinstance(d3_model, D3DispersionModel)

    assert base_model.cutoff == pytest.approx(6.0)
    assert d3_model.cutoff != pytest.approx(6.0)


@pytest.mark.skipif(_SKIP_NVALCHEMIOPS, reason="nvalchemiops is not installed.")
def test_pick_model_dispersion_missing_params() -> None:
    """Missing required D3 parameters should raise a clear KeyError."""
    with pytest.raises(KeyError, match="a2"):
        pick_model(
            TorchSimModelType.LENNARD_JONES,
            model_path="",
            dispersion=True,
            a1=0.4289,
            s8=0.7875,
            d3_params=_dummy_d3_params(),
        )
