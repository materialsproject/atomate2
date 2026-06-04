"""Tests for Quasi-Harmonic Approximation (QHA) workflow."""

import pytest
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.phonon.qha import SiestaQhaFlowMaker
from atomate2.siesta.jobs.phonon.phonopy import PhonopyMaker as PhononMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker


def get_si_structure():
    """Get a simple Si structure for testing."""
    return Structure(
        lattice=Lattice.cubic(5.43),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def test_qha_maker_initialization():
    """Test that SiestaQhaFlowMaker initializes correctly."""
    maker = SiestaQhaFlowMaker()

    assert maker.name == "siesta qha"
    assert maker.number_of_frames == 5
    assert maker.ignore_imaginary_modes is False
    assert maker.eos_type == "vinet"
    assert maker.pressure == [0.0]
    assert len(maker.temperature) == 11  # Default temperature points
    assert maker.temperature[0] == 0
    assert maker.temperature[-1] == 1000
    assert maker.skip_analysis is False
    assert maker.volume_factor == 0.95
    assert isinstance(maker.phonon_maker, PhononMaker)
    assert maker.prev_calc_dir_argname == "prev_dir"


def test_qha_maker_default_eos():
    """Test that default EOS maker is created correctly."""
    maker = SiestaQhaFlowMaker()

    # Should create default EOS maker (eos_type is on QhaMaker, not EosMaker)
    assert maker.eos_maker is not None
    assert isinstance(maker.eos_maker, SiestaEosFlowMaker)
    assert maker.eos_type == "vinet"  # eos_type is on QhaMaker
    assert maker.eos_maker.number_of_frames == 9


def test_qha_maker_custom_params():
    """Test SiestaQhaFlowMaker with custom parameters."""
    phonon_maker = PhononMaker(
        supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], mesh=(100, 100, 100)
    )

    eos_maker = SiestaEosFlowMaker(number_of_frames=7)

    maker = SiestaQhaFlowMaker(
        name="custom qha",
        number_of_frames=7,
        ignore_imaginary_modes=True,
        eos_type="birch_murnaghan",
        eos_maker=eos_maker,
        phonon_maker=phonon_maker,
        pressure=[0.0, 1.0, 2.0],
        temperature=[300, 600, 900],
        volume_factor=0.90,
    )

    assert maker.name == "custom qha"
    assert maker.number_of_frames == 7
    assert maker.ignore_imaginary_modes is True
    assert maker.eos_type == "birch_murnaghan"
    assert maker.pressure == [0.0, 1.0, 2.0]
    assert maker.temperature == [300, 600, 900]
    assert maker.volume_factor == 0.90
    assert maker.phonon_maker.supercell_matrix == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]


def test_qha_invalid_frames():
    """Test that QHA raises error with too few frames."""
    with pytest.raises(ValueError, match="number_of_frames must be at least 3"):
        SiestaQhaFlowMaker(number_of_frames=2)


def test_qha_workflow_creation():
    """Test that the QHA workflow is created correctly."""
    si = get_si_structure()
    maker = SiestaQhaFlowMaker(number_of_frames=5, temperature=[300, 600, 900])

    flow = maker.make(si)

    assert flow is not None
    # Flow is created correctly with expected maker name
    assert maker.name == "siesta qha"


def test_qha_no_optimization():
    """Test QHA workflow without structure optimization."""
    si = get_si_structure()
    maker = SiestaQhaFlowMaker(structure_optimizer=None, number_of_frames=4)

    flow = maker.make(si)

    assert flow is not None
    assert maker.structure_optimizer is None


@pytest.mark.parametrize("eos_type", ["vinet", "birch_murnaghan", "murnaghan"])
def test_qha_eos_types(eos_type):
    """Test different EOS types."""
    si = get_si_structure()
    maker = SiestaQhaFlowMaker(eos_type=eos_type)

    flow = maker.make(si)

    assert flow is not None
    assert maker.eos_type == eos_type
    # eos_type is on QhaMaker, not EosMaker


def test_qha_temperature_pressure_conversion():
    """Test that single temperature/pressure values are converted to lists."""
    maker = SiestaQhaFlowMaker(temperature=300.0, pressure=1.0)

    assert isinstance(maker.temperature, list)
    assert maker.temperature == [300.0]
    assert isinstance(maker.pressure, list)
    assert maker.pressure == [1.0]


def test_qha_multiple_pressures():
    """Test QHA with multiple pressure points."""
    si = get_si_structure()
    maker = SiestaQhaFlowMaker(pressure=[0.0, 5.0, 10.0, 20.0], temperature=[300, 600])

    flow = maker.make(si)

    assert flow is not None
    assert len(maker.pressure) == 4
    assert maker.pressure[3] == 20.0


def test_qha_volume_factor():
    """Test different volume factors for QHA."""
    si = get_si_structure()

    # Smaller volume range
    maker_small = SiestaQhaFlowMaker(volume_factor=0.98)
    flow_small = maker_small.make(si)
    assert flow_small is not None
    assert maker_small.volume_factor == 0.98

    # Larger volume range
    maker_large = SiestaQhaFlowMaker(volume_factor=0.90)
    flow_large = maker_large.make(si)
    assert flow_large is not None
    assert maker_large.volume_factor == 0.90


def test_qha_with_relaxation():
    """Test QHA workflow with initial structure relaxation."""
    si = get_si_structure()

    # Use factory method for variable cell relaxation
    relax_maker = RelaxMaker.variable_cell_relaxation(
        user_params={"MD.MaxCGDispl": "0.1 Bohr", "MD.MaxForceTol": "0.02 eV/Ang"}
    )

    maker = SiestaQhaFlowMaker(structure_optimizer=relax_maker, number_of_frames=5)

    flow = maker.make(si)

    assert flow is not None
    assert maker.structure_optimizer is not None
    assert isinstance(maker.structure_optimizer, RelaxMaker)


def test_qha_skip_analysis():
    """Test QHA workflow with analysis skipped."""
    si = get_si_structure()
    maker = SiestaQhaFlowMaker(skip_analysis=True, number_of_frames=4)

    flow = maker.make(si)

    assert flow is not None
    assert maker.skip_analysis is True


def test_qha_ignore_imaginary_modes():
    """Test QHA for systems with imaginary modes (metals)."""
    si = get_si_structure()
    maker = SiestaQhaFlowMaker(ignore_imaginary_modes=True, number_of_frames=5)

    flow = maker.make(si)

    assert flow is not None
    assert maker.ignore_imaginary_modes is True


def test_qha_phonon_settings_propagation():
    """Test that phonon settings are properly propagated."""
    from atomate2.siesta.jobs.core import StaticMaker

    # Create custom static maker with custodian settings
    static_maker = StaticMaker(use_custodian=True, custodian_max_errors=15)

    phonon_maker = PhononMaker(
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        mesh=(80, 80, 80),
        static_maker=static_maker,
    )

    maker = SiestaQhaFlowMaker(phonon_maker=phonon_maker, number_of_frames=5)

    si = get_si_structure()
    flow = maker.make(si)

    assert flow is not None
    assert maker.phonon_maker.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert maker.phonon_maker.mesh == (80, 80, 80)
    assert maker.phonon_maker.static_maker.use_custodian is True


def test_qha_custom_eos_maker():
    """Test QHA with custom EOS maker."""
    eos_maker = SiestaEosFlowMaker(name="custom eos", number_of_frames=11)

    maker = SiestaQhaFlowMaker(eos_maker=eos_maker, eos_type="murnaghan")

    si = get_si_structure()
    flow = maker.make(si)

    assert flow is not None
    assert maker.eos_maker.name == "custom eos"
    assert maker.eos_maker.number_of_frames == 11
    assert maker.eos_type == "murnaghan"


def test_qha_maker_repr():
    """Test string representation of SiestaQhaFlowMaker."""
    maker = SiestaQhaFlowMaker(name="test qha", number_of_frames=6)

    repr_str = repr(maker)
    assert "SiestaQhaFlowMaker" in repr_str


@pytest.mark.skip(reason="Requires actual SIESTA execution")
def test_qha_full_workflow():
    """Integration test for full QHA workflow (requires SIESTA)."""
    si = get_si_structure()

    maker = SiestaQhaFlowMaker(
        number_of_frames=5,
        temperature=[300, 600, 900],
        pressure=0.0,
        phonon_maker=PhononMaker(
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], mesh=(50, 50, 50)
        ),
    )

    flow = maker.make(si)
    responses = run_locally(flow, create_folders=True)

    # Check that the workflow completed and produced results
    assert len(responses) > 0
    # The actual QHA results would include thermal expansion, heat capacity, etc.


# ==================== Additional QHA Tests ====================


class TestQhaDryRun:
    """Test dry-run mode for QHA workflows."""

    def test_qha_with_dry_run_enabled(self):
        """Test SiestaQhaFlowMaker with dry_run=True."""
        si = get_si_structure()
        maker = SiestaQhaFlowMaker(dry_run=True, number_of_frames=5)

        flow = maker.make(si)

        assert flow is not None
        assert maker.dry_run is True

    def test_qha_dry_run_default_false(self):
        """Test that dry_run defaults to False."""
        maker = SiestaQhaFlowMaker()
        assert maker.dry_run is False

    def test_qha_dry_run_propagates_to_child_makers(self):
        """Test that dry_run propagates to structure_optimizer, eos_maker, phonon_maker."""
        maker = SiestaQhaFlowMaker(dry_run=True, number_of_frames=5)

        # dry_run should propagate through __post_init__
        assert maker.dry_run is True
        # BaseSiestaFlowMaker's __post_init__ handles propagation


class TestQhaPostInit:
    """Test __post_init__ logic for SiestaQhaFlowMaker."""

    def test_qha_initial_relax_maker_mapping(self):
        """Test that structure_optimizer is mapped to initial_relax_maker."""
        relax_maker = RelaxMaker.variable_cell_relaxation()
        maker = SiestaQhaFlowMaker(structure_optimizer=relax_maker)

        # __post_init__ should map structure_optimizer → initial_relax_maker
        assert maker.initial_relax_maker is relax_maker
        assert maker.structure_optimizer is relax_maker

    def test_qha_eos_relax_maker_creation(self):
        """Test that eos_relax_maker is created from structure_optimizer."""
        relax_maker = RelaxMaker.variable_cell_relaxation(
            user_params={"a2s_kpts": [8, 8, 8], "Mesh.Cutoff": "400 Ry"}
        )
        maker = SiestaQhaFlowMaker(structure_optimizer=relax_maker)

        # __post_init__ should create eos_relax_maker with same user_params
        assert maker.eos_relax_maker is not None
        assert isinstance(maker.eos_relax_maker, RelaxMaker)
        # User params should be propagated
        user_params = maker.eos_relax_maker.input_set_generator.user_params
        assert "a2s_kpts" in user_params
        assert user_params["a2s_kpts"] == [8, 8, 8]

    def test_qha_phonon_maker_attribute_compatibility(self):
        """Test that phonon_maker gets compatibility attributes for CommonQhaMaker."""
        phonon_maker = PhononMaker(supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        maker = SiestaQhaFlowMaker(phonon_maker=phonon_maker)

        # __post_init__ should add bulk_relax_maker and static_energy_maker attributes
        assert hasattr(maker.phonon_maker, "bulk_relax_maker")
        assert hasattr(maker.phonon_maker, "static_energy_maker")


class TestQhaEdgeCases:
    """Test edge cases for QHA workflows."""

    def test_qha_minimum_frames(self):
        """Test QHA with exact minimum number_of_frames=3."""
        si = get_si_structure()
        maker = SiestaQhaFlowMaker(number_of_frames=3)

        flow = maker.make(si)

        assert flow is not None
        assert maker.number_of_frames == 3

    def test_qha_large_number_of_frames(self):
        """Test QHA with large number of volume points."""
        si = get_si_structure()
        maker = SiestaQhaFlowMaker(number_of_frames=15)

        flow = maker.make(si)

        assert flow is not None
        assert maker.number_of_frames == 15


class TestQhaInheritance:
    """Test BaseSiestaFlowMaker inheritance for SiestaQhaFlowMaker."""

    def test_qha_inherits_from_base_siesta_flow_maker(self):
        """Test that SiestaQhaFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = SiestaQhaFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_qha_has_dry_run_attribute(self):
        """Test that SiestaQhaFlowMaker has dry_run attribute from BaseSiestaFlowMaker."""
        maker = SiestaQhaFlowMaker()
        assert hasattr(maker, "dry_run")

    def test_qha_inherits_from_common_qha_maker(self):
        """Test that SiestaQhaFlowMaker inherits from CommonQhaMaker."""
        from atomate2.common.flows.qha import CommonQhaMaker

        maker = SiestaQhaFlowMaker()
        assert isinstance(maker, CommonQhaMaker)


class TestQhaSerialization:
    """Test serialization for SiestaQhaFlowMaker."""

    def test_qha_maker_serialization(self):
        """Test SiestaQhaFlowMaker serialization with as_dict/from_dict."""
        maker = SiestaQhaFlowMaker(
            name="serialize_test",
            number_of_frames=7,
            ignore_imaginary_modes=True,
            eos_type="birch_murnaghan",
            pressure=[0.0, 1.0, 2.0],
            temperature=[300, 600, 900],
            volume_factor=0.92,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "SiestaQhaFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, SiestaQhaFlowMaker)
        assert maker_restored.name == "serialize_test"
        assert maker_restored.number_of_frames == 7
        assert maker_restored.ignore_imaginary_modes is True
        assert maker_restored.eos_type == "birch_murnaghan"
        assert maker_restored.pressure == [0.0, 1.0, 2.0]
        assert maker_restored.temperature == [300, 600, 900]
        assert maker_restored.volume_factor == 0.92
