"""
Tests for dry-run infrastructure (dry_run.py).

These tests validate:
- Single structure saving with multiple formats
- Multiple structure saving with metadata
- Workflow summary generation
- File I/O operations
- Return value structures
"""

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.dry_run import (
    dry_run_save_multiple_structures,
    dry_run_save_structure,
    dry_run_workflow_summary,
)


@pytest.fixture
def si_structure():
    """Create a simple Si structure for testing."""
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])


@pytest.fixture
def al_structure():
    """Create a simple Al structure for testing."""
    lattice = Lattice.cubic(4.05)
    return Structure(lattice, ["Al"], [[0.0, 0.0, 0.0]])


class TestDryRunSaveStructure:
    """Tests for dry_run_save_structure function."""

    def test_save_structure_cif_format(self, si_structure, tmp_path):
        """Test saving structure in CIF format (default)."""
        result = dry_run_save_structure.original(
            structure=si_structure,
            output_dir=str(tmp_path / "output"),
            output_format="cif",
            label="test_si",
        )

        # Check return dict structure
        assert result["dry_run"] is True
        assert result["label"] == "test_si"
        assert result["formula"] == "Si"
        assert result["num_atoms"] == 2
        assert "lattice" in result
        assert result["metadata"] == {}

        # Check file was created
        expected_file = tmp_path / "output" / "test_si.cif"
        assert expected_file.exists()
        assert result["structure_file"] == str(expected_file)

        # Check lattice info
        assert abs(result["lattice"]["a"] - 5.43) < 0.01
        assert abs(result["lattice"]["volume"] - 5.43**3) < 0.1

    def test_save_structure_xsf_format(self, si_structure, tmp_path):
        """Test saving structure in XSF format."""
        result = dry_run_save_structure.original(
            structure=si_structure,
            output_dir=str(tmp_path),
            output_format="xsf",
            label="xsf_test",
        )

        assert result["dry_run"] is True
        assert result["label"] == "xsf_test"

        # Check XSF file was created
        expected_file = tmp_path / "xsf_test.xsf"
        assert expected_file.exists()

    def test_save_structure_json_format(self, si_structure, tmp_path):
        """Test saving structure in JSON format."""
        _ = dry_run_save_structure.original(
            structure=si_structure,
            output_dir=str(tmp_path),
            output_format="json",
            label="json_test",
        )

        # Check JSON file was created
        expected_file = tmp_path / "json_test.json"
        assert expected_file.exists()

    def test_save_structure_with_metadata(self, si_structure, tmp_path):
        """Test saving structure with custom metadata."""
        metadata = {
            "maker": "RelaxMaker",
            "tier": "intermediate",
            "basis": "DZP",
            "scale_factor": 0.95,
        }

        result = dry_run_save_structure.original(
            structure=si_structure,
            output_dir=str(tmp_path),
            label="metadata_test",
            metadata=metadata,
        )

        assert result["metadata"] == metadata
        assert result["metadata"]["maker"] == "RelaxMaker"
        assert result["metadata"]["scale_factor"] == 0.95

    def test_save_structure_creates_output_dir(self, si_structure, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        _ = dry_run_save_structure.original(
            structure=si_structure, output_dir=str(nested_dir), label="nested_test"
        )

        # Check directory was created
        assert nested_dir.exists()

        # Check file was created inside
        expected_file = nested_dir / "nested_test.cif"
        assert expected_file.exists()

    def test_save_structure_default_label(self, si_structure, tmp_path):
        """Test default label is 'structure'."""
        result = dry_run_save_structure.original(
            structure=si_structure, output_dir=str(tmp_path)
        )

        assert result["label"] == "structure"
        assert (tmp_path / "structure.cif").exists()

    def test_save_structure_lattice_info(self, si_structure, tmp_path):
        """Test lattice information in return dict."""
        result = dry_run_save_structure.original(
            structure=si_structure, output_dir=str(tmp_path), label="lattice_test"
        )

        lattice = result["lattice"]
        assert "a" in lattice
        assert "b" in lattice
        assert "c" in lattice
        assert "alpha" in lattice
        assert "beta" in lattice
        assert "gamma" in lattice
        assert "volume" in lattice

        # For cubic lattice, a=b=c and alpha=beta=gamma=90
        assert abs(lattice["a"] - lattice["b"]) < 0.01
        assert abs(lattice["a"] - lattice["c"]) < 0.01
        assert abs(lattice["alpha"] - 90.0) < 0.01


class TestDryRunSaveMultipleStructures:
    """Tests for dry_run_save_multiple_structures function."""

    def test_save_multiple_structures_basic(self, si_structure, al_structure, tmp_path):
        """Test saving multiple structures."""
        structures = [si_structure, al_structure, si_structure]

        result = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            label_prefix="multi",
        )

        # Check return dict
        assert result["dry_run"] is True
        assert result["num_structures"] == 3
        assert len(result["structure_files"]) == 3
        assert result["output_dir"] == str(tmp_path)

        # Check files were created with correct numbering
        assert (tmp_path / "multi_000.cif").exists()
        assert (tmp_path / "multi_001.cif").exists()
        assert (tmp_path / "multi_002.cif").exists()

    def test_save_multiple_structures_with_metadata(
        self, si_structure, al_structure, tmp_path
    ):
        """Test saving multiple structures with metadata per structure."""
        structures = [si_structure, al_structure]
        metadata_list = [{"scale": 0.95, "step": 0}, {"scale": 1.05, "step": 1}]

        result = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            label_prefix="eos",
            metadata_list=metadata_list,
        )

        # Check metadata was stored correctly
        assert result["structure_files"][0]["metadata"]["scale"] == 0.95
        assert result["structure_files"][0]["metadata"]["step"] == 0
        assert result["structure_files"][1]["metadata"]["scale"] == 1.05
        assert result["structure_files"][1]["metadata"]["step"] == 1

    def test_save_multiple_structures_file_info(self, si_structure, tmp_path):
        """Test structure file info in return dict."""
        structures = [si_structure, si_structure]

        result = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            label_prefix="info_test",
        )

        # Check first structure info
        file_info = result["structure_files"][0]
        assert file_info["index"] == 0
        assert file_info["label"] == "info_test_000"
        assert "file" in file_info
        assert file_info["formula"] == "Si"
        assert file_info["num_atoms"] == 2
        assert "volume" in file_info
        assert file_info["metadata"] == {}

    def test_save_multiple_structures_xsf_format(self, si_structure, tmp_path):
        """Test saving multiple structures in XSF format."""
        structures = [si_structure, si_structure, si_structure]

        _ = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            output_format="xsf",
            label_prefix="xsf_multi",
        )

        # Check XSF files were created
        assert (tmp_path / "xsf_multi_000.xsf").exists()
        assert (tmp_path / "xsf_multi_001.xsf").exists()
        assert (tmp_path / "xsf_multi_002.xsf").exists()

    def test_save_multiple_structures_mismatched_metadata(
        self, si_structure, tmp_path, caplog
    ):
        """Test warning when metadata list length doesn't match structures."""
        structures = [si_structure, si_structure, si_structure]
        metadata_list = [{"a": 1}]  # Only 1 metadata for 3 structures

        result = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            label_prefix="mismatch",
            metadata_list=metadata_list,
        )

        # Should still succeed but use empty metadata
        assert result["num_structures"] == 3
        assert result["structure_files"][0]["metadata"] == {}
        assert result["structure_files"][1]["metadata"] == {}

    def test_save_multiple_structures_single_structure(self, si_structure, tmp_path):
        """Test saving single structure in list."""
        structures = [si_structure]

        result = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            label_prefix="single",
        )

        assert result["num_structures"] == 1
        assert (tmp_path / "single_000.cif").exists()

    def test_save_multiple_structures_many_structures(self, si_structure, tmp_path):
        """Test saving many structures (check numbering with >99 structures)."""
        # Create 105 structures
        structures = [si_structure] * 105

        result = dry_run_save_multiple_structures.original(
            structures=structures,
            output_dir=str(tmp_path),
            label_prefix="many",
        )

        assert result["num_structures"] == 105

        # Check various numbered files
        assert (tmp_path / "many_000.cif").exists()
        assert (tmp_path / "many_050.cif").exists()
        assert (tmp_path / "many_099.cif").exists()
        assert (tmp_path / "many_100.cif").exists()
        assert (tmp_path / "many_104.cif").exists()


class TestDryRunWorkflowSummary:
    """Tests for dry_run_workflow_summary function."""

    def test_create_workflow_summary_basic(self, tmp_path):
        """Test creating basic workflow summary."""
        job_outputs = [
            {
                "dry_run": True,
                "label": "slab",
                "formula": "Al",
                "num_atoms": 32,
            },
            {
                "dry_run": True,
                "label": "adsorbate",
                "formula": "CO",
                "num_atoms": 2,
            },
        ]

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="adsorption_scan",
            output_dir=str(tmp_path),
        )

        # Check return dict
        assert result["dry_run"] is True
        assert result["num_jobs"] == 2
        assert result["workflow_type"] == "adsorption_scan"
        assert "timestamp" in result
        assert "summary_file" in result

        # Check summary file was created
        summary_file = Path(result["summary_file"])
        assert summary_file.exists()
        assert summary_file.name == "dry_run_summary.txt"

    def test_workflow_summary_file_content(self, tmp_path):
        """Test workflow summary file content."""
        job_outputs = [
            {
                "dry_run": True,
                "label": "structure_1",
                "formula": "Si",
                "num_atoms": 8,
            },
        ]

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="eos",
            output_dir=str(tmp_path),
            num_volumes=10,
            volume_range="0.90-1.10",
        )

        # Read and check summary file content
        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        # Check header
        assert "DRY RUN SUMMARY: eos" in content
        assert "Generated:" in content

        # Check workflow parameters section
        assert "Workflow Parameters:" in content
        assert "num_volumes: 10" in content
        assert "volume_range: 0.90-1.10" in content

        # Check job outputs section
        assert "Generated 1 job outputs:" in content
        assert "structure_1" in content
        assert "Si" in content
        assert "8 atoms" in content

        # Check instructions section
        assert "Next Steps:" in content
        assert "VESTA" in content
        assert "dry_run=False" in content

    def test_workflow_summary_multiple_jobs(self, tmp_path):
        """Test summary with multiple job outputs."""
        job_outputs = [
            {"dry_run": True, "label": f"site_{i}", "formula": "Al", "num_atoms": 34}
            for i in range(5)
        ]

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="adsorption_scan",
            output_dir=str(tmp_path),
            grid_size=(5, 5),
            height=2.0,
        )

        assert result["num_jobs"] == 5

        # Check file content
        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        assert "Generated 5 job outputs:" in content
        assert "grid_size: (5, 5)" in content
        assert "height: 2.0" in content

    def test_workflow_summary_with_multiple_structures_output(self, tmp_path):
        """Test summary with multiple structures per job."""
        job_outputs = [
            {
                "dry_run": True,
                "num_structures": 10,
                "structure_files": [{}, {}, {}],  # Simplified
            },
        ]

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="phonon",
            output_dir=str(tmp_path),
        )

        # Check file content
        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        assert "Multiple structures: 10 files" in content

    def test_workflow_summary_no_metadata(self, tmp_path):
        """Test summary without workflow metadata."""
        job_outputs = [
            {"dry_run": True, "label": "test", "formula": "Si", "num_atoms": 2}
        ]

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="simple",
            output_dir=str(tmp_path),
            # No additional metadata
        )

        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        # Should still have summary, just no parameters section
        assert "DRY RUN SUMMARY: simple" in content
        assert "Generated 1 job outputs:" in content

    def test_workflow_summary_non_dry_run_output(self, tmp_path):
        """Test summary with non-dry-run output (should be skipped)."""
        job_outputs = [
            {"dry_run": True, "label": "dry_job", "formula": "Si", "num_atoms": 2},
            {"dry_run": False, "label": "real_job"},  # Not a dry-run
            {"something": "else"},  # No dry_run key
        ]

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="mixed",
            output_dir=str(tmp_path),
        )

        assert result["num_jobs"] == 3

        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        assert "dry_job" in content
        assert "Non-dry-run output (skipped)" in content

    def test_workflow_summary_empty_job_list(self, tmp_path):
        """Test summary with empty job outputs list."""
        job_outputs = []

        result = dry_run_workflow_summary.original(
            job_outputs=job_outputs,
            workflow_type="empty",
            output_dir=str(tmp_path),
        )

        assert result["num_jobs"] == 0

        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        assert "Generated 0 job outputs:" in content

    def test_workflow_summary_creates_directory(self, tmp_path):
        """Test that summary creates output directory if needed."""
        nested_dir = tmp_path / "deep" / "nested" / "path"
        assert not nested_dir.exists()

        result = dry_run_workflow_summary.original(
            job_outputs=[],
            workflow_type="test",
            output_dir=str(nested_dir),
        )

        assert nested_dir.exists()
        assert Path(result["summary_file"]).exists()
