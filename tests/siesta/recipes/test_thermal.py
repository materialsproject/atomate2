"""Tests for thermal recipe functions."""

from jobflow import Flow, Job

from atomate2.siesta.recipes.thermal import (
    gruneisen_workflow,
    high_temperature_properties,
    phonon_workflow,
    qha_workflow,
    thermal_expansion_workflow,
    thermal_properties,
    vibrational_stability_check,
)


class TestThermalProperties:
    """Test thermal_properties recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic thermal properties workflow."""
        flow = thermal_properties(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = thermal_properties(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = thermal_properties(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestPhononWorkflow:
    """Test phonon_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic phonon workflow."""
        flow = phonon_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_supercell_matrix(self, si_structure):
        """Test with custom supercell matrix."""
        flow = phonon_workflow(
            si_structure,
            auto_params=False,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        )
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = phonon_workflow(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = phonon_workflow(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestGruneisenWorkflow:
    """Test gruneisen_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic Gruneisen workflow."""
        flow = gruneisen_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = gruneisen_workflow(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = gruneisen_workflow(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestQhaWorkflow:
    """Test qha_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic QHA workflow."""
        flow = qha_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = qha_workflow(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = qha_workflow(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestThermalExpansionWorkflow:
    """Test thermal_expansion_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic thermal expansion workflow."""
        flow = thermal_expansion_workflow(si_structure)
        assert isinstance(flow, Flow)


class TestHighTemperatureProperties:
    """Test high_temperature_properties recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic high temperature properties workflow."""
        flow = high_temperature_properties(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = high_temperature_properties(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = high_temperature_properties(
            si_structure, auto_params=False, dry_run=True
        )
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestVibrationalStabilityCheck:
    """Test vibrational_stability_check recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic vibrational stability check."""
        flow = vibrational_stability_check(si_structure)
        assert isinstance(flow, Flow)
