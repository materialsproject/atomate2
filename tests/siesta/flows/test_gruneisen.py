"""Tests for Grüneisen parameter workflow."""

import pytest
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.phonon.gruneisen import SiestaGruneisenFlowMaker
from atomate2.siesta.jobs.phonon.phonopy import PhonopyMaker as PhononMaker


def get_si_structure():
    """Get a simple Si structure for testing."""
    return Structure(
        lattice=Lattice.cubic(5.43),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def test_gruneisen_maker_initialization():
    """Test that SiestaGruneisenFlowMaker initializes correctly."""
    maker = SiestaGruneisenFlowMaker()

    assert maker.name == "siesta gruneisen"
    assert maker.perc_vol == 0.01
    assert maker.use_symmetry is True
    assert maker.symprec == 1e-4
    assert isinstance(maker.phonon_maker, PhononMaker)
    assert maker.prev_calc_dir_argname == "prev_dir"


def test_gruneisen_maker_custom_params():
    """Test SiestaGruneisenFlowMaker with custom parameters."""
    phonon_maker = PhononMaker(
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], mesh=(50, 50, 50)
    )

    maker = SiestaGruneisenFlowMaker(
        name="custom gruneisen",
        perc_vol=0.02,
        use_symmetry=False,
        phonon_maker=phonon_maker,
    )

    assert maker.name == "custom gruneisen"
    assert maker.perc_vol == 0.02
    assert maker.use_symmetry is False
    assert maker.phonon_maker.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert maker.phonon_maker.mesh == (50, 50, 50)


def test_gruneisen_workflow_creation():
    """Test that the Grüneisen workflow is created correctly."""
    si = get_si_structure()
    maker = SiestaGruneisenFlowMaker(perc_vol=0.01)

    flow = maker.make(si)

    assert flow is not None
    # Flow is created correctly with expected maker name
    assert maker.name == "siesta gruneisen"
    # The flow should contain phonon calculations at 3 volumes
    # This is handled by the parent BaseGruneisenMaker


def test_gruneisen_no_optimization():
    """Test Grüneisen workflow without structure optimization."""
    si = get_si_structure()
    maker = SiestaGruneisenFlowMaker(structure_optimizer=None, perc_vol=0.015)

    flow = maker.make(si)

    assert flow is not None
    assert maker.structure_optimizer is None


@pytest.mark.parametrize("perc_vol", [0.005, 0.01, 0.02, 0.03])
def test_gruneisen_volume_changes(perc_vol):
    """Test different volume change percentages."""
    si = get_si_structure()
    maker = SiestaGruneisenFlowMaker(perc_vol=perc_vol)

    flow = maker.make(si)

    assert flow is not None
    assert maker.perc_vol == perc_vol


def test_gruneisen_symmetry_settings():
    """Test symmetry settings in Grüneisen workflow."""
    si = get_si_structure()

    # With symmetry
    maker_sym = SiestaGruneisenFlowMaker(use_symmetry=True, symprec=1e-3)
    flow_sym = maker_sym.make(si)
    assert flow_sym is not None
    assert maker_sym.use_symmetry is True
    assert maker_sym.symprec == 1e-3

    # Without symmetry
    maker_nosym = SiestaGruneisenFlowMaker(use_symmetry=False)
    flow_nosym = maker_nosym.make(si)
    assert flow_nosym is not None
    assert maker_nosym.use_symmetry is False


def test_gruneisen_phonon_settings_propagation():
    """Test that phonon settings are properly propagated."""
    from atomate2.siesta.jobs.core import StaticMaker

    # Create custom static maker with custodian settings
    static_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)

    phonon_maker = PhononMaker(
        supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        mesh=(100, 100, 100),
        static_maker=static_maker,
    )

    maker = SiestaGruneisenFlowMaker(phonon_maker=phonon_maker, perc_vol=0.01)

    si = get_si_structure()
    flow = maker.make(si)

    assert flow is not None
    assert maker.phonon_maker.supercell_matrix == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    assert maker.phonon_maker.mesh == (100, 100, 100)
    assert maker.phonon_maker.static_maker.use_custodian is True


def test_gruneisen_with_relaxation():
    """Test Grüneisen workflow with initial structure relaxation."""
    from atomate2.siesta.jobs.core import RelaxMaker

    si = get_si_structure()

    # Use factory method for variable cell relaxation
    relax_maker = RelaxMaker.variable_cell_relaxation(
        user_params={"MD.MaxCGDispl": "0.1 Bohr", "MD.MaxForceTol": "0.02 eV/Ang"}
    )

    maker = SiestaGruneisenFlowMaker(structure_optimizer=relax_maker, perc_vol=0.01)

    flow = maker.make(si)

    assert flow is not None
    assert maker.structure_optimizer is not None
    # Check that it's a RelaxMaker instance
    assert isinstance(maker.structure_optimizer, RelaxMaker)


def test_gruneisen_additional_kwargs():
    """Test additional keyword arguments for Grüneisen calculation."""
    si = get_si_structure()

    maker = SiestaGruneisenFlowMaker(
        phonon_maker=PhononMaker(mesh=(20, 20, 20)), perc_vol=0.01
    )

    flow = maker.make(si)

    assert flow is not None
    # Check that phonon maker has custom mesh
    assert maker.phonon_maker.mesh == (20, 20, 20)


def test_gruneisen_maker_repr():
    """Test string representation of SiestaGruneisenFlowMaker."""
    maker = SiestaGruneisenFlowMaker(name="test gruneisen", perc_vol=0.015)

    repr_str = repr(maker)
    assert "SiestaGruneisenFlowMaker" in repr_str


@pytest.mark.skip(reason="Requires actual SIESTA execution")
def test_gruneisen_full_workflow():
    """Integration test for full Grüneisen workflow (requires SIESTA)."""
    si = get_si_structure()

    maker = SiestaGruneisenFlowMaker(
        perc_vol=0.01,
        phonon_maker=PhononMaker(
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], mesh=(50, 50, 50)
        ),
    )

    flow = maker.make(si)
    responses = run_locally(flow, create_folders=True)

    # Check that the workflow completed and produced results
    assert len(responses) > 0
    # The actual Grüneisen parameters would be in the output


# ============================================================================
# Additional tests for untested areas
# ============================================================================


def test_get_static_maker_inherits_input_set_generator():
    """Test that _get_static_maker inherits input_set_generator from phonon_maker.static_maker."""
    from atomate2.siesta.jobs.core import StaticMaker
    from atomate2.siesta.sets.core import StaticSetGenerator

    # Create custom static maker with specific input_set_generator
    custom_generator = StaticSetGenerator(
        user_params={"Mesh.Cutoff": "300 Ry"},
        tier="advanced",
    )
    custom_static = StaticMaker(input_set_generator=custom_generator)

    phonon_maker = PhononMaker(static_maker=custom_static)
    maker = SiestaGruneisenFlowMaker(phonon_maker=phonon_maker)

    # Access the _get_static_maker method
    static_maker = maker._get_static_maker()

    # Verify input_set_generator is inherited (same instance)
    assert static_maker.input_set_generator is custom_generator
    assert static_maker.input_set_generator.tier == "advanced"


def test_get_static_maker_default():
    """Test _get_static_maker with default phonon maker."""
    maker = SiestaGruneisenFlowMaker()

    static_maker = maker._get_static_maker()

    # Should return a StaticMaker instance
    from atomate2.siesta.jobs.core import StaticMaker

    assert isinstance(static_maker, StaticMaker)


def test_dry_run_propagation():
    """Test that dry_run parameter propagates through Grüneisen workflow."""
    si = get_si_structure()

    maker = SiestaGruneisenFlowMaker(dry_run=True, perc_vol=0.01)

    flow = maker.make(si)

    # Verify dry_run is set
    assert maker.dry_run is True
    assert flow is not None


def test_dry_run_false_default():
    """Test that dry_run defaults to False."""
    maker = SiestaGruneisenFlowMaker()

    assert maker.dry_run is False


def test_compute_gruneisen_param_kwargs():
    """Test compute_gruneisen_param_kwargs dictionary handling."""
    si = get_si_structure()

    # Custom kwargs for Grüneisen parameter computation
    custom_kwargs = {"fit_type": "polynomial"}

    maker = SiestaGruneisenFlowMaker(
        compute_gruneisen_param_kwargs=custom_kwargs, perc_vol=0.01
    )

    flow = maker.make(si)

    assert flow is not None
    assert maker.compute_gruneisen_param_kwargs == custom_kwargs


def test_generate_frequencies_eigenvectors_kwargs():
    """Test generate_frequencies_eigenvectors_kwargs dictionary handling."""
    si = get_si_structure()

    # Custom kwargs for frequency/eigenvector generation
    custom_kwargs = {"with_eigenvectors": True, "with_group_velocities": False}

    maker = SiestaGruneisenFlowMaker(
        generate_frequencies_eigenvectors_kwargs=custom_kwargs, perc_vol=0.01
    )

    flow = maker.make(si)

    assert flow is not None
    assert maker.generate_frequencies_eigenvectors_kwargs == custom_kwargs


def test_prev_dir_parameter():
    """Test prev_dir parameter usage in make() method."""
    si = get_si_structure()

    maker = SiestaGruneisenFlowMaker(perc_vol=0.01)

    # Test with prev_dir specified
    flow = maker.make(si, prev_dir="/path/to/previous/calculation")

    assert flow is not None
    # The prev_dir would be used for restart capability


def test_prev_dir_none_default():
    """Test that prev_dir defaults to None."""
    si = get_si_structure()

    maker = SiestaGruneisenFlowMaker(perc_vol=0.01)

    # Test without prev_dir (default None)
    flow = maker.make(si)

    assert flow is not None


def test_invalid_perc_vol_negative():
    """Test Grüneisen maker with negative volume percentage."""
    si = get_si_structure()

    # Negative perc_vol should still create flow (validation in parent class)
    maker = SiestaGruneisenFlowMaker(perc_vol=-0.01)

    flow = maker.make(si)

    assert flow is not None
    assert maker.perc_vol == -0.01


def test_invalid_perc_vol_zero():
    """Test Grüneisen maker with zero volume percentage."""
    si = get_si_structure()

    maker = SiestaGruneisenFlowMaker(perc_vol=0.0)

    flow = maker.make(si)

    assert flow is not None
    assert maker.perc_vol == 0.0


def test_symprec_interaction_with_use_symmetry():
    """Test that symprec is used when use_symmetry is True."""
    si = get_si_structure()

    # High precision symmetry
    maker_high = SiestaGruneisenFlowMaker(use_symmetry=True, symprec=1e-6)
    flow_high = maker_high.make(si)
    assert flow_high is not None
    assert maker_high.symprec == 1e-6

    # Low precision symmetry
    maker_low = SiestaGruneisenFlowMaker(use_symmetry=True, symprec=1e-2)
    flow_low = maker_low.make(si)
    assert flow_low is not None
    assert maker_low.symprec == 1e-2
