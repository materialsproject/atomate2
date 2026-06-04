"""
Tests for Basis Set Convergence workflows.

These tests validate:
- BasisSizeConvergenceFlowMaker (tests different basis sizes)
- CompleteBasisConvergenceFlowMaker (tests basis sizes + parameters)
- @job functions: collect_basis_size_data, plot_basis_size_convergence, write_basis_size_summary
- @job functions: collect_complete_basis_data, plot_complete_basis_convergence, write_complete_basis_summary
"""

import pytest
from jobflow import Flow
from pymatgen.core import Structure, Lattice

from atomate2.siesta.flows.basis import (
    BasisSizeConvergenceFlowMaker,
    CompleteBasisConvergenceFlowMaker,
)
from atomate2.siesta.flows.basis.size import (
    collect_basis_size_data,
    plot_basis_size_convergence,
    write_basis_size_summary,
)
from atomate2.siesta.jobs.core import StaticMaker


@pytest.fixture
def si_structure():
    """Simple Si structure for testing."""
    return Structure(
        lattice=Lattice.cubic(5.43),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


class TestBasisSizeConvergenceMaker:
    """Tests for BasisSizeConvergenceFlowMaker workflow."""

    def test_default_basis_size_maker(self):
        """Test creation of default BasisSizeConvergenceFlowMaker."""
        maker = BasisSizeConvergenceFlowMaker()

        assert maker.name == "Basis Size Convergence"
        assert maker.basis_sizes == [
            "SZ",
            "DZ",
            "DZP",
            "TZP",
        ]  # Default from __post_init__
        assert maker.energy_shift == 0.01
        assert maker.split_norm == 0.15
        assert maker.kpts is None
        assert maker.static_maker is not None

    def test_basis_size_maker_custom_params(self):
        """Test BasisSizeConvergenceFlowMaker with custom parameters."""
        maker = BasisSizeConvergenceFlowMaker(
            name="Custom Basis Test",
            basis_sizes=["DZ", "DZP", "DZDP", "TZP", "TZDP"],
            energy_shift=0.02,
            split_norm=0.20,
            kpts=[6, 6, 6],
        )

        assert maker.name == "Custom Basis Test"
        assert maker.basis_sizes == ["DZ", "DZP", "DZDP", "TZP", "TZDP"]
        assert maker.energy_shift == 0.02
        assert maker.split_norm == 0.20
        assert maker.kpts == [6, 6, 6]

    def test_basis_size_maker_with_custom_static_maker(self):
        """Test BasisSizeConvergenceFlowMaker with custom StaticMaker."""
        custom_static = StaticMaker()
        maker = BasisSizeConvergenceFlowMaker(
            static_maker=custom_static,
            basis_sizes=["DZ", "DZP"],
        )

        assert maker.static_maker is custom_static

    def test_basis_size_maker_make_flow(self, si_structure):
        """Test that BasisSizeConvergenceFlowMaker creates a valid flow."""
        maker = BasisSizeConvergenceFlowMaker(
            basis_sizes=["SZ", "DZ", "DZP"],
            energy_shift=0.01,
            split_norm=0.15,
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "Basis Size Convergence"
        # Should have N basis jobs + collect + plot + summary
        assert len(flow) >= 3 + 3  # 3 basis sizes + 3 analysis jobs

    def test_basis_size_maker_job_naming(self, si_structure):
        """Test that basis size jobs have correct names with counters."""
        maker = BasisSizeConvergenceFlowMaker(
            basis_sizes=["SZ", "DZ"],
            name="Test Basis",
        )

        flow = maker.make(si_structure)

        # Get job names (first N jobs are basis size calculations)
        job_names = [job.name for job in flow]

        # Check for counter format
        assert any("[1_of_2]" in name for name in job_names)
        assert any("[2_of_2]" in name for name in job_names)

    def test_basis_size_maker_serialization(self):
        """Test BasisSizeConvergenceFlowMaker serialization."""
        maker = BasisSizeConvergenceFlowMaker(
            name="serialize_test",
            basis_sizes=["DZ", "DZP"],
            energy_shift=0.015,
            split_norm=0.18,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "BasisSizeConvergenceFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, BasisSizeConvergenceFlowMaker)
        assert maker_restored.name == "serialize_test"
        assert maker_restored.energy_shift == 0.015

    def test_basis_size_maker_post_init_defaults(self):
        """Test that __post_init__ sets default basis_sizes and static_maker."""
        maker = BasisSizeConvergenceFlowMaker(basis_sizes=None, static_maker=None)

        assert maker.basis_sizes == ["SZ", "DZ", "DZP", "TZP"]
        assert isinstance(maker.static_maker, StaticMaker)


class TestCompleteBasisConvergenceMaker:
    """Tests for CompleteBasisConvergenceFlowMaker workflow."""

    def test_default_complete_basis_maker(self):
        """Test creation of default CompleteBasisConvergenceFlowMaker."""
        maker = CompleteBasisConvergenceFlowMaker()

        assert maker.name == "Complete Basis Convergence"
        assert maker.basis_sizes == ["DZ", "DZP", "TZP"]  # Default from __post_init__
        assert maker.energy_shifts == [0.010, 0.015, 0.020]
        assert maker.split_norms == [0.15, 0.20, 0.25]
        assert maker.kpts is None
        assert maker.static_maker is not None

    def test_complete_basis_maker_custom_params(self):
        """Test CompleteBasisConvergenceFlowMaker with custom parameters."""
        maker = CompleteBasisConvergenceFlowMaker(
            name="Custom Complete Test",
            basis_sizes=["DZ", "DZP"],
            energy_shifts=[0.005, 0.010],
            split_norms=[0.15, 0.20],
            kpts=[4, 4, 4],
        )

        assert maker.name == "Custom Complete Test"
        assert maker.basis_sizes == ["DZ", "DZP"]
        assert maker.energy_shifts == [0.005, 0.010]
        assert maker.split_norms == [0.15, 0.20]
        assert maker.kpts == [4, 4, 4]

    def test_complete_basis_maker_job_count(self, si_structure):
        """Test that CompleteBasisConvergenceFlowMaker creates correct number of jobs."""
        maker = CompleteBasisConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP"],
            energy_shifts=[0.010, 0.015],
            split_norms=[0.15, 0.20],
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        # Should have: 2 basis × 2 shifts × 2 norms = 8 SCF jobs + analysis jobs
        # Total = SCF flow + collect + plot + summary + basis_viz + 2 real_basis
        # = 1 (scf_flow) + 1 (collect) + 1 (plot) + 1 (summary) + 1 (basis_viz) + 2 (real_basis) = 7
        assert len(flow) >= 7

    def test_complete_basis_maker_post_init_defaults(self):
        """Test that __post_init__ sets defaults for all lists."""
        maker = CompleteBasisConvergenceFlowMaker(
            basis_sizes=None, energy_shifts=None, split_norms=None
        )

        assert maker.basis_sizes == ["DZ", "DZP", "TZP"]
        assert maker.energy_shifts == [0.010, 0.015, 0.020]
        assert maker.split_norms == [0.15, 0.20, 0.25]

    def test_complete_basis_maker_serialization(self):
        """Test CompleteBasisConvergenceFlowMaker serialization."""
        maker = CompleteBasisConvergenceFlowMaker(
            name="serialize_complete",
            basis_sizes=["DZ"],
            energy_shifts=[0.010],
            split_norms=[0.15],
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "CompleteBasisConvergenceFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, CompleteBasisConvergenceFlowMaker)
        assert maker_restored.name == "serialize_complete"


class TestCollectBasisSizeData:
    """Test collect_basis_size_data @job function."""

    def test_collect_basis_size_data_basic(self):
        """Test collect_basis_size_data with mock outputs."""
        from unittest.mock import MagicMock

        # Create mock job outputs
        mock_output1 = MagicMock()
        mock_output1.output.energy = -10.5
        mock_output1.output.forces = [[0.01, 0.01, 0.01]]
        mock_output1.output.stress = [0.1, 0.1, 0.1, 0, 0, 0]

        mock_output2 = MagicMock()
        mock_output2.output.energy = -10.6
        mock_output2.output.forces = [[0.02, 0.02, 0.02]]
        mock_output2.output.stress = [0.2, 0.2, 0.2, 0, 0, 0]

        job_outputs = [mock_output1, mock_output2]
        job_metadata = [
            {
                "name": "SZ",
                "basis_size": "SZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
            },
            {
                "name": "DZ",
                "basis_size": "DZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
            },
        ]

        result = collect_basis_size_data.original(job_outputs, job_metadata)

        assert isinstance(result, dict)
        assert "basis_sizes" in result
        assert "energies" in result
        assert len(result["energies"]) == 2
        assert result["basis_sizes"][0] == "SZ"
        assert result["basis_sizes"][1] == "DZ"

    def test_collect_basis_size_data_empty_outputs(self):
        """Test collect_basis_size_data with no valid outputs."""
        result = collect_basis_size_data.original([], [])

        assert isinstance(result, dict)
        assert len(result["energies"]) == 0


class TestPlotBasisSizeConvergence:
    """Test plot_basis_size_convergence @job function."""

    def test_plot_basis_size_convergence_with_data(self, tmp_path):
        """Test plot_basis_size_convergence creates plot file."""
        import os

        basis_data = {
            "basis_sizes": ["SZ", "DZ", "DZP"],
            "energies": [-10.5, -10.6, -10.55],
            "max_forces": [0.01, 0.02, 0.015],
            "max_stresses": [0.1, 0.2, 0.15],
            "run_times": [50.0, 75.0, 100.0],
        }

        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = plot_basis_size_convergence.original(
                basis_data, output_file="test_plot.png"
            )

            assert isinstance(result, dict)
            assert "plot" in result
            assert (tmp_path / "test_plot.png").exists()
        finally:
            os.chdir(original_dir)

    def test_plot_basis_size_convergence_empty_data(self, tmp_path):
        """Test plot_basis_size_convergence with no data creates error plot."""
        import os

        basis_data = {
            "basis_sizes": [],
            "energies": [],
            "max_forces": [],
            "max_stresses": [],
            "run_times": [],
        }

        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = plot_basis_size_convergence.original(
                basis_data, output_file="error_plot.png"
            )

            assert isinstance(result, dict)
            assert "plot" in result
            # Should still create a plot (with error message)
            assert (tmp_path / "error_plot.png").exists()
        finally:
            os.chdir(original_dir)


class TestWriteBasisSizeSummary:
    """Test write_basis_size_summary @job function."""

    def test_write_basis_size_summary_with_data(self, tmp_path):
        """Test write_basis_size_summary creates summary file."""
        import os

        basis_data = {
            "basis_sizes": ["SZ", "DZ", "DZP"],
            "energies": [-10.5, -10.6, -10.55],
            "max_forces": [0.01, 0.02, 0.015],
            "max_stresses": [0.1, 0.2, 0.15],
            "run_times": [50.0, 75.0, 100.0],
        }

        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = write_basis_size_summary.original(
                basis_data, output_file="test_summary.txt"
            )

            assert isinstance(result, dict)
            assert "summary" in result
            summary_file = tmp_path / "test_summary.txt"
            assert summary_file.exists()

            # Check content
            content = summary_file.read_text()
            assert "BASIS SIZE CONVERGENCE STUDY" in content
            assert "SZ" in content
            assert "DZ" in content
            assert "DZP" in content
        finally:
            os.chdir(original_dir)

    def test_write_basis_size_summary_empty_data(self, tmp_path):
        """Test write_basis_size_summary with no data creates error summary."""
        import os

        basis_data = {
            "basis_sizes": [],
            "energies": [],
            "max_forces": [],
            "max_stresses": [],
            "run_times": [],
        }

        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = write_basis_size_summary.original(
                basis_data, output_file="error_summary.txt"
            )

            assert isinstance(result, dict)
            assert "summary" in result
            summary_file = tmp_path / "error_summary.txt"
            assert summary_file.exists()

            # Should contain error message
            content = summary_file.read_text()
            assert "ERROR" in content
        finally:
            os.chdir(original_dir)


class TestBasisConvergenceEdgeCases:
    """Test edge cases for basis convergence workflows."""

    def test_basis_size_maker_with_single_basis(self, si_structure):
        """Test BasisSizeConvergenceFlowMaker with only one basis size."""
        maker = BasisSizeConvergenceFlowMaker(basis_sizes=["DZP"])

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_basis_size_maker_with_many_basis_sizes(self, si_structure):
        """Test BasisSizeConvergenceFlowMaker with many basis sizes."""
        maker = BasisSizeConvergenceFlowMaker(
            basis_sizes=["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.basis_sizes == [
            "SZ",
            "DZ",
            "DZP",
            "SZP",
            "DZDP",
            "TZ",
            "TZP",
            "TZDP",
        ]

    def test_complete_basis_maker_single_parameter_set(self, si_structure):
        """Test CompleteBasisConvergenceFlowMaker with single parameter values."""
        maker = CompleteBasisConvergenceFlowMaker(
            basis_sizes=["DZP"],
            energy_shifts=[0.010],
            split_norms=[0.15],
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_complete_basis_maker_large_parameter_space(self, si_structure):
        """Test CompleteBasisConvergenceFlowMaker with large parameter space."""
        maker = CompleteBasisConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP", "TZP"],
            energy_shifts=[0.005, 0.010, 0.015, 0.020],
            split_norms=[0.10, 0.15, 0.20, 0.25],
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        # Should create 3 × 4 × 4 = 48 SCF jobs + analysis
        # Just verify flow was created successfully


class TestBasisConvergenceDryRun:
    """Test dry-run mode for basis convergence workflows."""

    def test_basis_size_maker_with_dry_run(self, si_structure):
        """Test BasisSizeConvergenceFlowMaker with dry_run enabled."""
        maker = BasisSizeConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP"],
            dry_run=True,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.dry_run is True

    def test_complete_basis_maker_with_dry_run(self, si_structure):
        """Test CompleteBasisConvergenceFlowMaker with dry_run enabled."""
        maker = CompleteBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.010],
            split_norms=[0.15],
            dry_run=True,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.dry_run is True


class TestBasisConvergenceInheritance:
    """Test BaseSiestaFlowMaker inheritance."""

    def test_basis_size_maker_inherits_base(self):
        """Test that BasisSizeConvergenceFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = BasisSizeConvergenceFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_complete_basis_maker_inherits_base(self):
        """Test that CompleteBasisConvergenceFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = CompleteBasisConvergenceFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_basis_size_maker_has_dry_run(self):
        """Test that BasisSizeConvergenceFlowMaker has dry_run attribute."""
        maker = BasisSizeConvergenceFlowMaker()
        assert hasattr(maker, "dry_run")

    def test_complete_basis_maker_has_dry_run(self):
        """Test that CompleteBasisConvergenceFlowMaker has dry_run attribute."""
        maker = CompleteBasisConvergenceFlowMaker()
        assert hasattr(maker, "dry_run")
