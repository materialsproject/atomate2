"""Tests for convergence flow module."""

import pytest
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.flows.convergence import (
    KpointsConvergenceFlowMaker,
    MeshCutoffConvergenceFlowMaker,
)

# Note: We don't test collect_convergence_data and plot_convergence directly
# because they are @job decorated functions that cause serialization issues
# when tested with mock objects. The workflow structure tests below verify
# that these jobs are created correctly within the flow.


@pytest.fixture
def si_structure():
    """Create a simple Si structure for testing."""
    lattice = Lattice.cubic(5.43)
    structure = Structure(
        lattice,
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    return structure


@pytest.fixture
def co_molecule():
    """Create a simple CO molecule for testing."""
    molecule = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.128]])
    return molecule


class TestMeshCutoffConvergenceMaker:
    """Test MeshCutoffConvergenceFlowMaker."""

    def test_maker_initialization(self):
        """Test maker initializes with defaults."""
        maker = MeshCutoffConvergenceFlowMaker()

        assert maker.name == "Mesh Cutoff Convergence"
        assert len(maker.mesh_cutoffs) == 9  # Default: 100-500 Ry
        assert maker.mesh_cutoffs == [100, 150, 200, 250, 300, 350, 400, 450, 500]

    def test_maker_with_custom_cutoffs(self):
        """Test maker with custom cutoffs."""
        custom_cutoffs = [100, 200, 300]
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=custom_cutoffs)

        assert maker.mesh_cutoffs == custom_cutoffs

    def test_make_creates_flow(self, si_structure):
        """Test that make() creates a Flow object."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200, 300])

        flow = maker.make(si_structure)

        # Check flow structure
        assert flow is not None
        assert hasattr(flow, "jobs")
        assert len(flow.jobs) == 3  # calc_flow, collect_job, plot_job

    def test_make_with_dry_run(self, si_structure):
        """Test make() with dry_run mode."""
        maker = MeshCutoffConvergenceFlowMaker(
            mesh_cutoffs=[100, 200], dry_run=True, dry_run_output_dir="dry_run_test"
        )

        maker.make(si_structure)

        # Verify dry_run propagated to static_maker
        assert maker.static_maker.dry_run is True
        assert maker.static_maker.dry_run_output_dir == "dry_run_test"

    def test_make_job_names(self, si_structure):
        """Test that jobs have correct names."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200, 300])

        flow = maker.make(si_structure)

        # Get the calculation flow (first job)
        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        assert "Mesh Cutoff Convergence-100Ry" in job_names
        assert "Mesh Cutoff Convergence-200Ry" in job_names
        assert "Mesh Cutoff Convergence-300Ry" in job_names

    def test_make_with_prev_dir(self, si_structure):
        """Test make() with prev_dir parameter."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200])

        flow = maker.make(si_structure, prev_dir="/path/to/prev")

        # Just verify flow was created successfully with prev_dir
        assert flow is not None

    def test_collect_and_plot_jobs_created(self, si_structure):
        """Test that collect and plot jobs are created."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200])

        flow = maker.make(si_structure)

        # Check job names
        job_names = [job.name for job in flow.jobs]
        assert any("collect" in name.lower() for name in job_names)
        assert any("plot" in name.lower() for name in job_names)


class TestKpointsConvergenceMaker:
    """Test KpointsConvergenceFlowMaker."""

    def test_maker_initialization(self):
        """Test maker initializes with defaults."""
        maker = KpointsConvergenceFlowMaker()

        assert maker.name == "K-points Convergence"
        assert len(maker.kpoints_list) == 7  # Default
        assert maker.kpoints_list[0] == [2, 2, 2]
        assert maker.kpoints_list[-1] == [10, 10, 10]

    def test_maker_with_custom_kpoints(self):
        """Test maker with custom k-points."""
        custom_kpts = [[2, 2, 2], [4, 4, 4], [6, 6, 6]]
        maker = KpointsConvergenceFlowMaker(kpoints_list=custom_kpts)

        assert maker.kpoints_list == custom_kpts

    def test_make_creates_flow(self, si_structure):
        """Test that make() creates a Flow object."""
        maker = KpointsConvergenceFlowMaker(
            kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6]]
        )

        flow = maker.make(si_structure)

        # Check flow structure
        assert flow is not None
        assert hasattr(flow, "jobs")
        assert len(flow.jobs) == 3  # calc_flow, collect_job, plot_job

    def test_make_with_dry_run(self, si_structure):
        """Test make() with dry_run mode."""
        maker = KpointsConvergenceFlowMaker(
            kpoints_list=[[2, 2, 2], [4, 4, 4]],
            dry_run=True,
            dry_run_output_dir="dry_run_kpts",
        )

        maker.make(si_structure)

        # Verify dry_run propagated
        assert maker.static_maker.dry_run is True
        assert maker.static_maker.dry_run_output_dir == "dry_run_kpts"

    def test_make_job_names(self, si_structure):
        """Test that jobs have correct names."""
        maker = KpointsConvergenceFlowMaker(
            kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6]]
        )

        flow = maker.make(si_structure)

        # Get the calculation flow (first job)
        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        assert "K-points Convergence-2x2x2" in job_names
        assert "K-points Convergence-4x4x4" in job_names
        assert "K-points Convergence-6x6x6" in job_names

    def test_make_with_prev_dir(self, si_structure):
        """Test make() with prev_dir parameter."""
        maker = KpointsConvergenceFlowMaker(kpoints_list=[[2, 2, 2], [4, 4, 4]])

        flow = maker.make(si_structure, prev_dir="/path/to/prev")

        # Just verify flow was created successfully
        assert flow is not None

    def test_collect_and_plot_jobs_created(self, si_structure):
        """Test that collect and plot jobs are created."""
        maker = KpointsConvergenceFlowMaker(kpoints_list=[[2, 2, 2], [4, 4, 4]])

        flow = maker.make(si_structure)

        # Check job names
        job_names = [job.name for job in flow.jobs]
        assert any("collect" in name.lower() for name in job_names)
        assert any("plot" in name.lower() for name in job_names)

    def test_kpoints_format_conversion(self, si_structure):
        """Test that k-points are properly formatted in job names."""
        maker = KpointsConvergenceFlowMaker(kpoints_list=[[3, 5, 7]])

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        # Should convert [3,5,7] to "3x5x7"
        assert "K-points Convergence-3x5x7" in job_names


class TestConvergenceMakerEdgeCases:
    """Test edge cases for convergence makers."""

    def test_mesh_cutoff_single_value(self, si_structure):
        """Test with single mesh cutoff value."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[200])

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        assert len(calc_flow.jobs) == 1

    def test_kpoints_single_value(self, si_structure):
        """Test with single k-points grid."""
        maker = KpointsConvergenceFlowMaker(kpoints_list=[[4, 4, 4]])

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        assert len(calc_flow.jobs) == 1

    def test_mesh_cutoff_many_values(self, si_structure):
        """Test with many mesh cutoff values."""
        cutoffs = list(range(100, 600, 50))  # 10 values
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=cutoffs)

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        assert len(calc_flow.jobs) == 10

    def test_kpoints_asymmetric(self, si_structure):
        """Test with asymmetric k-points grids."""
        maker = KpointsConvergenceFlowMaker(kpoints_list=[[2, 4, 6], [3, 5, 7]])

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        assert "K-points Convergence-2x4x6" in job_names
        assert "K-points Convergence-3x5x7" in job_names

    def test_flow_output_structure(self, si_structure):
        """Test that flow output is properly structured."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200])

        flow = maker.make(si_structure)

        # Flow should have output set to collect_job.output
        assert hasattr(flow, "output")

    def test_molecule_input(self, co_molecule):
        """Test that convergence works with Molecule input."""
        maker = KpointsConvergenceFlowMaker(kpoints_list=[[1, 1, 1]])

        flow = maker.make(co_molecule)

        # Should work with molecules (though k-points aren't used)
        assert flow is not None
        calc_flow = flow.jobs[0]
        assert len(calc_flow.jobs) == 1


class TestCollectConvergenceData:
    """Test collect_convergence_data function using .original pattern."""

    def test_collect_mesh_cutoff_data(self):
        """Test collecting mesh cutoff convergence data."""
        from atomate2.siesta.flows.convergence.utils import collect_convergence_data

        # Mock flow results
        flow_results = {
            "uuid-1": type(
                "Response", (), {"output": type("Output", (), {"energy": -100.5})}
            )(),
            "uuid-2": type(
                "Response", (), {"output": type("Output", (), {"energy": -100.3})}
            )(),
            "uuid-3": type(
                "Response", (), {"output": type("Output", (), {"energy": -100.1})}
            )(),
        }

        job_metadata = [
            {"name": "Mesh-100Ry", "uuid": "uuid-1"},
            {"name": "Mesh-200Ry", "uuid": "uuid-2"},
            {"name": "Mesh-300Ry", "uuid": "uuid-3"},
        ]

        result = collect_convergence_data.original(
            flow_results, job_metadata, "mesh_cutoff"
        )

        assert len(result["energies"]) == 3
        assert result["energies"] == [-100.5, -100.3, -100.1]
        assert result["parameters"] == ["100Ry", "200Ry", "300Ry"]
        assert result["names"] == ["Mesh-100Ry", "Mesh-200Ry", "Mesh-300Ry"]

    def test_collect_kpoints_data(self):
        """Test collecting k-points convergence data."""
        from atomate2.siesta.flows.convergence.utils import collect_convergence_data

        flow_results = {
            "uuid-1": type(
                "Response", (), {"output": type("Output", (), {"energy": -50.0})}
            )(),
            "uuid-2": type(
                "Response", (), {"output": type("Output", (), {"energy": -50.5})}
            )(),
        }

        job_metadata = [
            {"name": "Kpts-2x2x2", "uuid": "uuid-1"},
            {"name": "Kpts-4x4x4", "uuid": "uuid-2"},
        ]

        result = collect_convergence_data.original(
            flow_results, job_metadata, "kpoints"
        )

        assert len(result["energies"]) == 2
        assert result["parameters"] == ["2x2x2", "4x4x4"]

    def test_collect_with_missing_results(self):
        """Test collection with missing job results."""
        from atomate2.siesta.flows.convergence.utils import collect_convergence_data

        flow_results = {
            "uuid-1": type(
                "Response", (), {"output": type("Output", (), {"energy": -100.0})}
            )(),
            # uuid-2 is missing
        }

        job_metadata = [
            {"name": "Job-1", "uuid": "uuid-1"},
            {"name": "Job-2", "uuid": "uuid-2"},  # Missing
            {"name": "Job-3", "uuid": "uuid-3"},  # Missing
        ]

        result = collect_convergence_data.original(
            flow_results, job_metadata, "mesh_cutoff"
        )

        # Should only have data from uuid-1
        assert len(result["energies"]) == 1
        assert result["energies"] == [-100.0]

    def test_collect_with_dry_run_dict(self):
        """Test collection with dry_run mode (dict results)."""
        from atomate2.siesta.flows.convergence.utils import collect_convergence_data

        # Dry run returns dicts instead of Response objects
        flow_results = {
            "uuid-1": {"dry_run": True, "structure_file": "test.cif"},
            "uuid-2": {"dry_run": True, "structure_file": "test2.cif"},
        }

        job_metadata = [
            {"name": "Mesh-100Ry", "uuid": "uuid-1"},
            {"name": "Mesh-200Ry", "uuid": "uuid-2"},
        ]

        result = collect_convergence_data.original(
            flow_results, job_metadata, "mesh_cutoff"
        )

        # Should skip dry_run results (no energies)
        assert len(result["energies"]) == 0
        assert len(result["parameters"]) == 0

    def test_collect_with_error_in_job(self):
        """Test collection when job data is malformed."""
        from atomate2.siesta.flows.convergence.utils import collect_convergence_data

        flow_results = {
            "uuid-1": type(
                "Response", (), {"output": type("Output", (), {"energy": -100.0})}
            )(),
            "uuid-2": type("Response", (), {"output": None})(),  # No energy attribute
        }

        job_metadata = [
            {"name": "Job-1", "uuid": "uuid-1"},
            {"name": "Job-2", "uuid": "uuid-2"},
        ]

        result = collect_convergence_data.original(
            flow_results, job_metadata, "mesh_cutoff"
        )

        # Should skip uuid-2 (error) and only collect uuid-1
        assert len(result["energies"]) == 1

    def test_collect_empty_job_list(self):
        """Test collection with no jobs."""
        from atomate2.siesta.flows.convergence.utils import collect_convergence_data

        result = collect_convergence_data.original({}, [], "mesh_cutoff")

        assert result["energies"] == []
        assert result["parameters"] == []
        assert result["names"] == []


class TestPlotConvergence:
    """Test plot_convergence function using .original pattern."""

    def test_plot_mesh_cutoff_convergence(self, tmp_path):
        """Test plotting mesh cutoff convergence."""
        from atomate2.siesta.flows.convergence.utils import plot_convergence
        from atomate2.siesta.utils.verbosity import VerbosityLevel

        convergence_data = {
            "parameters": ["100Ry", "200Ry", "300Ry", "400Ry"],
            "energies": [-100.5, -100.3, -100.15, -100.1],
            "names": ["Job-1", "Job-2", "Job-3", "Job-4"],
        }

        output_file = str(tmp_path / "test_mesh_cutoff.png")

        result = plot_convergence.original(
            convergence_data,
            "mesh_cutoff",
            output_file=output_file,
            verbosity=VerbosityLevel.INFO,
        )

        # Check plot files were created (one per metric, suffixed from base name)
        import os

        base_name = output_file.replace(".png", "")
        assert os.path.exists(f"{base_name}_energy.png")
        assert os.path.exists(f"{base_name}_convergence.png")
        assert isinstance(result, dict)
        assert result["energy"] == f"{base_name}_energy.png"

    def test_plot_kpoints_convergence(self, tmp_path):
        """Test plotting k-points convergence."""
        from atomate2.siesta.flows.convergence.utils import plot_convergence

        convergence_data = {
            "parameters": ["2x2x2", "4x4x4", "6x6x6", "8x8x8"],
            "energies": [-50.0, -50.4, -50.45, -50.48],
            "names": ["Kpts-1", "Kpts-2", "Kpts-3", "Kpts-4"],
        }

        output_file = str(tmp_path / "test_kpoints.png")

        _ = plot_convergence.original(
            convergence_data,
            "kpoints",
            output_file=output_file,
            verbosity=0,
        )

        # Check plot file
        import os

        base_name = output_file.replace(".png", "")
        assert os.path.exists(f"{base_name}_energy.png")

    def test_plot_with_default_filename(self, tmp_path):
        """Test plot with auto-generated filename."""
        import os

        from atomate2.siesta.flows.convergence.utils import plot_convergence

        # Change to tmp_path to avoid cluttering
        os.chdir(tmp_path)

        convergence_data = {
            "parameters": ["100Ry", "200Ry"],
            "energies": [-100.0, -100.1],
            "names": ["Job-1", "Job-2"],
        }

        result = plot_convergence.original(
            convergence_data, "mesh_cutoff", output_file=None, verbosity=0
        )

        # Should create default filenames (dict of suffixed plot paths)
        assert isinstance(result, dict)
        assert result["energy"] == "convergence_mesh_cutoff_energy.png"
        assert os.path.exists("convergence_mesh_cutoff_energy.png")

    def test_plot_with_empty_energies(self, tmp_path):
        """Test plot with no energies (dry_run mode)."""
        from atomate2.siesta.flows.convergence.utils import plot_convergence

        convergence_data = {
            "parameters": [],
            "energies": [],
            "names": [],
        }

        result = plot_convergence.original(
            convergence_data,
            "mesh_cutoff",
            output_file=str(tmp_path / "test.png"),
            verbosity=0,
        )

        # Should return dict with message about dry_run
        assert isinstance(result, dict)
        assert result["plot_file"] is None
        assert "dry_run" in result["message"].lower()

    def test_plot_kpoints_list_format(self, tmp_path):
        """Test plotting with k-points as lists."""
        from atomate2.siesta.flows.convergence.utils import plot_convergence

        convergence_data = {
            "parameters": [[2, 2, 2], [4, 4, 4], [6, 6, 6]],  # Lists instead of strings
            "energies": [-50.0, -50.4, -50.45],
            "names": ["Job-1", "Job-2", "Job-3"],
        }

        output_file = str(tmp_path / "test_kpts_list.png")

        _ = plot_convergence.original(
            convergence_data, "kpoints", output_file=output_file, verbosity=0
        )

        import os

        base_name = output_file.replace(".png", "")
        assert os.path.exists(f"{base_name}_energy.png")

    def test_plot_kpoints_bracket_string_format(self, tmp_path):
        """Test plotting with k-points as bracket strings."""
        from atomate2.siesta.flows.convergence.utils import plot_convergence

        convergence_data = {
            "parameters": ["[2, 2, 2]", "[4, 4, 4]"],  # String with brackets
            "energies": [-50.0, -50.4],
            "names": ["Job-1", "Job-2"],
        }

        output_file = str(tmp_path / "test_kpts_bracket.png")

        _ = plot_convergence.original(
            convergence_data, "kpoints", output_file=output_file, verbosity=0
        )

        import os

        base_name = output_file.replace(".png", "")
        assert os.path.exists(f"{base_name}_energy.png")

    def test_plot_with_integer_verbosity(self, tmp_path):
        """Test plot with integer verbosity instead of enum."""
        from atomate2.siesta.flows.convergence.utils import plot_convergence

        convergence_data = {
            "parameters": ["100Ry", "200Ry"],
            "energies": [-100.0, -100.1],
            "names": ["Job-1", "Job-2"],
        }

        output_file = str(tmp_path / "test_verbosity.png")

        # Test with integer verbosity (0 = QUIET, 1 = INFO, etc.)
        _ = plot_convergence.original(
            convergence_data, "mesh_cutoff", output_file=output_file, verbosity=1
        )

        import os

        base_name = output_file.replace(".png", "")
        assert os.path.exists(f"{base_name}_energy.png")


class TestConvergenceMakerCustomization:
    """Test customization options for convergence makers."""

    def test_mesh_cutoff_with_custom_static_maker(self, si_structure):
        """Test MeshCutoffConvergenceFlowMaker with custom StaticMaker."""
        from atomate2.siesta.jobs.core import StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator

        custom_static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )

        maker = MeshCutoffConvergenceFlowMaker(
            mesh_cutoffs=[100, 200], static_maker=custom_static_maker
        )

        maker.make(si_structure)

        # Verify custom static_maker is used
        assert (
            maker.static_maker.input_set_generator.user_params["PAO.BasisSize"] == "DZP"
        )

    def test_kpoints_with_custom_static_maker(self, si_structure):
        """Test KpointsConvergenceFlowMaker with custom StaticMaker."""
        from atomate2.siesta.jobs.core import StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator

        custom_static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params={"Mesh.Cutoff": "300 Ry"}
            )
        )

        maker = KpointsConvergenceFlowMaker(
            kpoints_list=[[2, 2, 2], [4, 4, 4]], static_maker=custom_static_maker
        )

        maker.make(si_structure)

        assert (
            maker.static_maker.input_set_generator.user_params["Mesh.Cutoff"]
            == "300 Ry"
        )

    def test_mesh_cutoff_with_custom_name(self, si_structure):
        """Test MeshCutoffConvergenceFlowMaker with custom workflow name."""
        maker = MeshCutoffConvergenceFlowMaker(
            name="Custom Mesh Study", mesh_cutoffs=[100, 200]
        )

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        assert "Custom Mesh Study-100Ry" in job_names
        assert "Custom Mesh Study-200Ry" in job_names

    def test_kpoints_with_custom_name(self, si_structure):
        """Test KpointsConvergenceFlowMaker with custom workflow name."""
        maker = KpointsConvergenceFlowMaker(
            name="Custom Kpts Study", kpoints_list=[[2, 2, 2]]
        )

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        assert "Custom Kpts Study-2x2x2" in job_names

    def test_mixed_mesh_cutoff_values(self, si_structure):
        """Test mesh cutoff with non-uniform spacing."""
        cutoffs = [50, 100, 250, 500, 1000]  # Non-uniform
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=cutoffs)

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        assert len(calc_flow.jobs) == 5

    def test_kpoints_1d_sampling(self, si_structure):
        """Test k-points with 1D sampling (for wire-like structures)."""
        maker = KpointsConvergenceFlowMaker(
            kpoints_list=[[8, 1, 1], [16, 1, 1], [32, 1, 1]]
        )

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        job_names = [job.name for job in calc_flow.jobs]

        assert "K-points Convergence-8x1x1" in job_names
        assert "K-points Convergence-16x1x1" in job_names

    def test_flow_uuid_uniqueness(self, si_structure):
        """Test that all jobs in flow have unique UUIDs."""
        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200, 300])

        flow = maker.make(si_structure)

        calc_flow = flow.jobs[0]
        uuids = [job.uuid for job in calc_flow.jobs]

        # All UUIDs should be unique
        assert len(uuids) == len(set(uuids))

    def test_console_verbosity_setting(self, si_structure):
        """Test that CONSOLE_VERBOSITY can be set."""
        from atomate2.siesta.utils.verbosity import VerbosityLevel

        maker = MeshCutoffConvergenceFlowMaker(mesh_cutoffs=[100, 200])
        maker.CONSOLE_VERBOSITY = VerbosityLevel.SILENT

        flow = maker.make(si_structure)

        # Just verify flow is created with custom verbosity
        assert flow is not None
        assert maker.CONSOLE_VERBOSITY == VerbosityLevel.SILENT
