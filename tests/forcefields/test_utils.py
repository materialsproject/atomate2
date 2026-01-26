"""Test machine learning forcefield utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from atomate2.forcefields import MLFF
from atomate2.forcefields.utils import ase_calculator, revert_default_dtype

if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize("mlff", MLFF)
def test_mlff(mlff: MLFF):
    assert mlff == MLFF(str(mlff)) == MLFF(str(mlff).split(".")[-1])


@pytest.mark.parametrize("mlff", ["MACE", MLFF.MatterSim, MLFF.SevenNet])
def test_ext_load(mlff: str | MLFF, test_dir, si_structure: Structure):
    decode_dict = {
        "MACE": {"@module": "mace.calculators", "@callable": "mace_mp"},
        MLFF.MatterSim: {
            "@module": "mattersim.forcefield",
            "@callable": "MatterSimCalculator",
        },
        MLFF.SevenNet: {
            "@module": "sevenn.sevennet_calculator",
            "@callable": "SevenNetCalculator",
        },
    }[mlff]
    calc_from_decode = ase_calculator(decode_dict)
    calc_from_preset = ase_calculator(str(MLFF.MACE_MP_0))
    calc_from_enum = ase_calculator(MLFF.MACE_MP_0)

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


@pytest.mark.skip(reason="M3GNet requires DGL which is PyTorch 2.4 incompatible")
def test_m3gnet_pot():
    import matgl
    from matgl.ext.ase import PESCalculator

    kwargs_calc = {"path": "M3GNet-MP-2021.2.8-DIRECT-PES", "stress_weight": 2.0}
    kwargs_default = {"stress_weight": 2.0}

    m3gnet_calculator = ase_calculator(calculator_meta="MLFF.M3GNet", **kwargs_calc)

    # uses "M3GNet-MP-2021.2.8-PES" per default
    m3gnet_default = ase_calculator(calculator_meta="MLFF.M3GNet", **kwargs_default)

    potential = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    m3gnet_pes_calc = PESCalculator(potential=potential, stress_weight=2.0)

    assert str(m3gnet_pes_calc.potential) == str(m3gnet_calculator.potential)
    # casting necessary because <class 'matgl.apps.pes.Potential'> can't be compared
    assert str(m3gnet_pes_calc.potential) != str(m3gnet_default.potential)
    assert m3gnet_pes_calc.stress_weight == m3gnet_calculator.stress_weight
    assert m3gnet_pes_calc.stress_weight == m3gnet_default.stress_weight


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
