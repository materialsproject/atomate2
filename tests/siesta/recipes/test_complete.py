"""Tests for complete recipe functions."""

from jobflow import Flow

from atomate2.siesta.recipes.complete import (
    battery_cathode_screening,
    complete_material_study,
    high_temperature_ceramic,
    magnetic_material_study,
    quick_characterization,
    semiconductor_device_study,
    structural_phase_transition,
    thermoelectric_analysis,
)


class TestCompleteMaterialStudy:
    """Test complete_material_study recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic complete material study."""
        flow = complete_material_study(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_properties_list(self, si_structure):
        """Test with specific properties list."""
        flow = complete_material_study(
            si_structure, auto_params=False, properties=["electronic", "mechanical"]
        )
        assert isinstance(flow, Flow)

    def test_with_convergence_testing(self, si_structure):
        """Test with convergence testing enabled."""
        flow = complete_material_study(
            si_structure, auto_params=False, test_convergence=True
        )
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = complete_material_study(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = complete_material_study(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestQuickCharacterization:
    """Test quick_characterization recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic quick characterization."""
        flow = quick_characterization(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = quick_characterization(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = quick_characterization(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestBatteryCathodeScreening:
    """Test battery_cathode_screening recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic battery cathode screening."""
        flow = battery_cathode_screening(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = battery_cathode_screening(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = battery_cathode_screening(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestThermoelectricAnalysis:
    """Test thermoelectric_analysis recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic thermoelectric analysis."""
        flow = thermoelectric_analysis(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = thermoelectric_analysis(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = thermoelectric_analysis(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestHighTemperatureCeramic:
    """Test high_temperature_ceramic recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic high temperature ceramic workflow."""
        flow = high_temperature_ceramic(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_max_temperature(self, si_structure):
        """Test with custom max temperature."""
        flow = high_temperature_ceramic(
            si_structure, auto_params=False, max_temperature=2500.0
        )
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = high_temperature_ceramic(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = high_temperature_ceramic(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestMagneticMaterialStudy:
    """Test magnetic_material_study recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic magnetic material study."""
        flow = magnetic_material_study(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = magnetic_material_study(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = magnetic_material_study(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestSemiconductorDeviceStudy:
    """Test semiconductor_device_study recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic semiconductor device study."""
        flow = semiconductor_device_study(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = semiconductor_device_study(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = semiconductor_device_study(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestStructuralPhaseTransition:
    """Test structural_phase_transition recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic phase transition study."""
        flow = structural_phase_transition(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = structural_phase_transition(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = structural_phase_transition(
            si_structure, auto_params=False, dry_run=True
        )
        assert isinstance(flow, Flow)
