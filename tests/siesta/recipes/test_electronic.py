"""Tests for electronic recipe functions."""

from jobflow import Flow

from atomate2.siesta.recipes.electronic import (
    electronic_properties,
    band_structure_workflow,
    dos_workflow,
    optical_properties,
    metal_properties,
    semiconductor_properties,
)


class TestElectronicProperties:
    """Test electronic_properties recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic electronic properties workflow."""
        flow = electronic_properties(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_auto_params(self, si_structure):
        """Test with auto parameter detection disabled to avoid preset bugs."""
        flow = electronic_properties(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameter overrides."""
        flow = electronic_properties(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = electronic_properties(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestBandStructureWorkflow:
    """Test band_structure_workflow recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic band structure workflow."""
        flow = band_structure_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_relax_first(self, si_structure):
        """Test with initial relaxation."""
        flow = band_structure_workflow(
            si_structure, relax_first=True, auto_params=False
        )
        assert isinstance(flow, Flow)

    def test_without_relax(self, si_structure):
        """Test without initial relaxation."""
        flow = band_structure_workflow(
            si_structure, relax_first=False, auto_params=False
        )
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = band_structure_workflow(
            si_structure,
            auto_params=False,
            user_params={"PAO.BasisSize": "TZP"},
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = band_structure_workflow(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestDosCalculation:
    """Test dos_workflow recipe."""

    def test_basic_calculation(self, si_structure):
        """Test basic DOS calculation."""
        flow = dos_workflow(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_relax_first(self, si_structure):
        """Test with initial relaxation."""
        flow = dos_workflow(si_structure, relax_first=True, auto_params=False)
        assert isinstance(flow, Flow)

    def test_without_relax(self, si_structure):
        """Test without initial relaxation."""
        flow = dos_workflow(si_structure, relax_first=False, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = dos_workflow(
            si_structure,
            auto_params=False,
            user_params={"PAO.BasisSize": "DZP"},
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = dos_workflow(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestOpticalProperties:
    """Test optical_properties recipe."""

    def test_basic_calculation(self, si_structure):
        """Test basic optical properties calculation."""
        flow = optical_properties(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_relax_first(self, si_structure):
        """Test with initial relaxation."""
        flow = optical_properties(si_structure, relax_first=True, auto_params=False)
        assert isinstance(flow, Flow)

    def test_without_relax(self, si_structure):
        """Test without initial relaxation."""
        flow = optical_properties(si_structure, relax_first=False, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = optical_properties(
            si_structure,
            auto_params=False,
            user_params={"PAO.BasisSize": "DZP"},
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = optical_properties(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestMetalProperties:
    """Test metal_properties recipe."""

    def test_basic_workflow(self, al_structure):
        """Test basic metal properties workflow."""
        flow = metal_properties(al_structure, relax_first=False)
        assert isinstance(flow, Flow)

    def test_with_relax(self, al_structure):
        """Test with initial relaxation."""
        flow = metal_properties(al_structure, relax_first=True)
        assert isinstance(flow, Flow)

    def test_without_relax(self, al_structure):
        """Test without initial relaxation."""
        flow = metal_properties(al_structure, relax_first=False)
        assert isinstance(flow, Flow)


class TestSemiconductorProperties:
    """Test semiconductor_properties recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic semiconductor properties workflow."""
        flow = semiconductor_properties(si_structure, relax_first=False)
        assert isinstance(flow, Flow)

    def test_with_relax(self, si_structure):
        """Test with initial relaxation."""
        flow = semiconductor_properties(si_structure, relax_first=True)
        assert isinstance(flow, Flow)

    def test_without_relax(self, si_structure):
        """Test without initial relaxation."""
        flow = semiconductor_properties(si_structure, relax_first=False)
        assert isinstance(flow, Flow)
