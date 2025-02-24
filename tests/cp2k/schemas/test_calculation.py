"""Tests for CP2K calculation schemas."""

from pymatgen.core import Lattice, Structure
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.cp2k.outputs import Cp2kOutput

from atomate2.cp2k.schemas.calculation import CalculationInput, RunStatistics


def test_run_statistics(cp2k_output_path):
    """Test RunStatistics schema."""
    output = Cp2kOutput(cp2k_output_path)
    stats = RunStatistics.from_cp2k_output(output)
    assert isinstance(stats.total_time, float)
    assert stats.total_time > 0


def test_calculation_input(cp2k_output_path):
    """Test CalculationInput schema."""
    output = Cp2kOutput(cp2k_output_path)

    struct = Structure(
        lattice=Lattice.cubic(3),
        species=("Si", "Si"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    output.initial_structure = struct  # mock initial_structure
    calc_input = CalculationInput.from_cp2k_output(output)

    # test basic properties
    assert isinstance(calc_input.structure, Structure)
    assert isinstance(calc_input.atomic_kind_info, dict | None)
    assert isinstance(calc_input.cp2k_input, Cp2kInput)
    assert isinstance(calc_input.dft, dict)
    assert isinstance(calc_input.cp2k_global, dict)
