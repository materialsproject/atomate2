"""Tests for mechanical recipe functions."""

from jobflow import Flow, Job

from atomate2.siesta.recipes.mechanical import (
    mechanical_properties,
    elastic_constants_workflow,
    eos_workflow,
    pressure_eos_workflow,
    hardness_estimation,
    anisotropy_analysis,
)


class TestMechanicalProperties:
    """Test mechanical_properties recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic mechanical properties workflow."""
        flow = mechanical_properties(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = mechanical_properties(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = mechanical_properties(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestElasticConstantsWorkflow:
    """Test elastic_constants_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic elastic constants workflow."""
        flow = elastic_constants_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = elastic_constants_workflow(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = elastic_constants_workflow(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestBulkModulusWorkflow:
    """Test eos_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic bulk modulus workflow."""
        flow = eos_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = eos_workflow(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = eos_workflow(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestPressureEosWorkflow:
    """Test pressure_eos_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic pressure EOS workflow."""
        flow = pressure_eos_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = pressure_eos_workflow(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = pressure_eos_workflow(si_structure, auto_params=False, dry_run=True)
        # dry_run returns a single relaxation Job (not a full Flow)
        assert isinstance(flow, (Flow, Job))


class TestHardnessEstimation:
    """Test hardness_estimation recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic hardness estimation."""
        flow = hardness_estimation(si_structure)
        assert isinstance(flow, Flow)


class TestAnisotropyAnalysis:
    """Test anisotropy_analysis recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic anisotropy analysis."""
        flow = anisotropy_analysis(si_structure)
        assert isinstance(flow, Flow)
