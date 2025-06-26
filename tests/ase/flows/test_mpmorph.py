"""Test MPMoprh flow using Lennard-Jones."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from jobflow import run_locally

from atomate2.ase.jobs import LennardJonesStaticMaker
from atomate2.common.flows.mpmorph import EquilibriumVolumeMaker

if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize("vol_scale", [1.25, 0.75])
def test_vol_scale(
    lj_fcc_ne_pars: dict[str, float], fcc_ne_structure: Structure, vol_scale: float
) -> None:
    # Perform single points here instead of MD for
    # (1) time cost and (2) reliability of output
    # This is a coarse test to ensure that the volume up and down scaling can occur

    test_structure = fcc_ne_structure.to_conventional()
    test_structure = test_structure.scale_lattice(vol_scale * test_structure.volume)
    test_structure = test_structure * (2, 2, 2)
    init_vol_per_atom = test_structure.volume

    init_strain = 0.05
    vscale = ((1 - init_strain) ** 3, (1 + init_strain) ** 3)
    flow = EquilibriumVolumeMaker(
        md_maker=LennardJonesStaticMaker(
            calculator_kwargs=lj_fcc_ne_pars,
        ),
        min_strain=0.05,
        initial_strain=0.05,
    ).make(test_structure)

    resp = run_locally(flow, ensure_success=True)
    output = None
    for _output in resp.values():
        if isinstance(_output[1].output, dict) and (
            output := _output[1].output.get("working_outputs")
        ):
            break

    # ensure that at least one refinement on the volume range was undertaken
    assert len(output["relax"]["volume"]) > 3

    final_v_ratio = output["V0"] / init_vol_per_atom
    if vol_scale > 1.0:
        assert final_v_ratio <= vscale[1]
    elif vol_scale < 1.0:
        assert vscale[0] <= final_v_ratio
