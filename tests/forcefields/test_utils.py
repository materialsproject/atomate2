"""Test machine learning forcefield utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from atomate2.forcefields import MLFF
from atomate2.forcefields.utils import ase_calculator, revert_default_dtype

from .conftest import mlff_is_installed

if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize("mlff", MLFF)
def test_mlff(mlff: MLFF):
    assert mlff == MLFF(str(mlff)) == MLFF(str(mlff).split(".")[-1])


@pytest.mark.parametrize(
    "mlff", [mlff for mlff in ["MACE", MLFF.SevenNet] if mlff_is_installed(mlff)]
)
def test_ext_load(mlff: str | MLFF, test_dir, si_structure: Structure):
    decode_dict = {
        "MACE": {"@module": "mace.calculators", "@callable": "mace_mp"},
        MLFF.SevenNet: {
            "@module": "sevenn.sevennet_calculator",
            "@callable": "SevenNetCalculator",
        },
    }[mlff]
    formatted_mlff = MLFF(mlff)
    calc_from_decode = ase_calculator(decode_dict)
    calc_from_preset = ase_calculator(str(formatted_mlff))
    calc_from_enum = ase_calculator(formatted_mlff)

    for other in (calc_from_preset, calc_from_enum):
        assert type(calc_from_decode) is type(other)
        assert calc_from_decode.name == other.name
        assert calc_from_decode.parameters == other.parameters == {}

    atoms = si_structure.to_ase_atoms()

    atoms.calc = calc_from_preset
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float | np.floating)
    assert energy < 0
    assert forces.shape == (2, 3)
    assert abs(forces.sum()) < 1e-6, f"unexpectedly large net {forces=}"


def test_raises_error():
    with pytest.raises(ValueError, match="Could not create"):
        ase_calculator("not_a_calculator")


@pytest.mark.skipif(not mlff_is_installed("MACE"), reason="mace_torch is not installed")
def test_mace_explicit_dispersion(ba_ti_o3_structure: Structure):
    from ase.calculators.mixing import SumCalculator
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    energies = {"mace": -39.969810485839844, "d3": -1.3136245271781846}

    model_path = download_mace_mp_checkpoint("medium-mpa-0")

    atoms = ba_ti_o3_structure.to_ase_atoms()

    with revert_default_dtype():
        calc_no_path = ase_calculator(MLFF.MACE_MPA_0, dispersion=True)
        assert isinstance(calc_no_path, SumCalculator)
        assert calc_no_path.get_potential_energy(atoms=atoms) == pytest.approx(
            sum(energies.values())
        )

        calc_path = ase_calculator(MLFF.MACE_MPA_0, model=model_path)
        assert not isinstance(calc_path, SumCalculator)
        assert calc_path.get_potential_energy(atoms=atoms) == pytest.approx(
            energies["mace"]
        )

        calc_path = ase_calculator(MLFF.MACE_MPA_0, model=model_path, dispersion=True)
        assert isinstance(calc_no_path, SumCalculator)
        assert calc_path.get_potential_energy(atoms=atoms) == pytest.approx(
            sum(energies.values())
        )
