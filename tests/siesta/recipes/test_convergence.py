"""Tests for convergence recipe functions."""

from jobflow import Flow

from atomate2.siesta.recipes.convergence import (
    basis_convergence,
    complete_convergence,
    convergence_suite,
    kpoints_convergence,
    mesh_cutoff_convergence,
    quick_convergence_check,
)


class TestConvergenceSuite:
    """Test convergence_suite recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic convergence suite."""
        flow = convergence_suite(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = convergence_suite(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = convergence_suite(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestKpointsConvergence:
    """Test kpoints_convergence recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic k-points convergence."""
        flow = kpoints_convergence(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_kpoints_list(self, si_structure):
        """Test with custom k-points list."""
        flow = kpoints_convergence(
            si_structure,
            auto_params=False,
            kpts_range=[2, 4, 6],
        )
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = kpoints_convergence(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = kpoints_convergence(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestMeshCutoffConvergence:
    """Test mesh_cutoff_convergence recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic mesh cutoff convergence."""
        flow = mesh_cutoff_convergence(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_cutoff_list(self, si_structure):
        """Test with custom cutoff list."""
        flow = mesh_cutoff_convergence(
            si_structure, auto_params=False, cutoff_range=[100, 200, 300, 400]
        )
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = mesh_cutoff_convergence(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = mesh_cutoff_convergence(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestBasisConvergence:
    """Test basis_convergence recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic basis convergence."""
        flow = basis_convergence(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = basis_convergence(
            si_structure, auto_params=False, user_params={"a2s_kpts": [6, 6, 6]}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = basis_convergence(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestCompleteConvergence:
    """Test complete_convergence recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic complete convergence."""
        flow = complete_convergence(si_structure, auto_params=False)
        assert isinstance(flow, Flow)

    def test_with_user_params(self, si_structure):
        """Test with user parameters."""
        flow = complete_convergence(
            si_structure, auto_params=False, user_params={"PAO.BasisSize": "DZP"}
        )
        assert isinstance(flow, Flow)

    def test_dry_run_mode(self, si_structure):
        """Test in dry_run mode."""
        flow = complete_convergence(si_structure, auto_params=False, dry_run=True)
        assert isinstance(flow, Flow)


class TestQuickConvergenceCheck:
    """Test quick_convergence_check recipe."""

    def test_basic_workflow(self, si_structure):
        """Test basic quick convergence check."""
        flow = quick_convergence_check(si_structure)
        assert isinstance(flow, Flow)
