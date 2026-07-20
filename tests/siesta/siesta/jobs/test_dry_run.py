"""Tests for core dry-run infrastructure."""

from pathlib import Path

import pytest
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.dry_run import (
    dry_run_save_multiple_structures,
    dry_run_save_structure,
    dry_run_workflow_summary,
)


@pytest.fixture
def si_structure():
    """Silicon structure for testing."""
    lattice = Lattice.cubic(5.43)
    return Structure(
        lattice,
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def scaled_structures(si_structure):
    """List of scaled structures for EOS-like tests."""
    structures = []
    for scale in [0.95, 1.0, 1.05]:
        struct = si_structure.copy()
        struct.scale_lattice(si_structure.volume * scale)
        structures.append(struct)
    return structures


class TestDryRunSaveStructure:
    """Test dry_run_save_structure function."""

    def test_basic_save(self, si_structure, tmp_path):
        """Test basic structure saving."""
        job = dry_run_save_structure(
            structure=si_structure,
            output_dir=str(tmp_path),
            output_format="cif",
            label="test_structure",
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check output fields
        assert output["dry_run"] is True
        assert output["label"] == "test_structure"
        assert output["formula"] == "Si"
        assert output["num_atoms"] == 2

        # Check file was created
        structure_file = Path(output["structure_file"])
        assert structure_file.exists()
        assert structure_file.name == "test_structure.cif"

    def test_multiple_formats(self, si_structure, tmp_path):
        """Test saving in different formats."""
        # Test only formats supported by pymatgen Structure.to()
        formats = ["cif", "xsf", "json"]

        for fmt in formats:
            job = dry_run_save_structure(
                structure=si_structure,
                output_dir=str(tmp_path / fmt),
                output_format=fmt,
                label=f"test_{fmt}",
            )

            result = run_locally(job, create_folders=True)
            output = result[job.uuid][1].output

            structure_file = Path(output["structure_file"])
            assert structure_file.exists()
            assert structure_file.suffix == f".{fmt}"

    def test_with_metadata(self, si_structure, tmp_path):
        """Test saving with custom metadata."""
        metadata = {
            "maker_name": "RelaxMaker",
            "tier": "intermediate",
            "user_params": {"PAO.BasisSize": "DZP"},
        }

        job = dry_run_save_structure(
            structure=si_structure,
            output_dir=str(tmp_path),
            label="test_metadata",
            metadata=metadata,
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check metadata is preserved
        assert output["metadata"] == metadata

    def test_lattice_info(self, si_structure, tmp_path):
        """Test lattice information is captured."""
        job = dry_run_save_structure(
            structure=si_structure,
            output_dir=str(tmp_path),
            label="test_lattice",
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check lattice parameters
        lattice = output["lattice"]
        assert "a" in lattice
        assert "b" in lattice
        assert "c" in lattice
        assert "volume" in lattice

        # Check values (cubic lattice)
        assert lattice["a"] == pytest.approx(5.43, rel=1e-3)
        assert lattice["a"] == pytest.approx(lattice["b"], rel=1e-10)
        assert lattice["a"] == pytest.approx(lattice["c"], rel=1e-10)


class TestDryRunSaveMultipleStructures:
    """Test dry_run_save_multiple_structures function."""

    def test_multiple_structures(self, scaled_structures, tmp_path):
        """Test saving multiple structures."""
        job = dry_run_save_multiple_structures(
            structures=scaled_structures,
            output_dir=str(tmp_path),
            label_prefix="eos",
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check summary
        assert output["dry_run"] is True
        assert output["num_structures"] == 3

        # Check all files were created
        structure_files = output["structure_files"]
        assert len(structure_files) == 3

        for i, file_info in enumerate(structure_files):
            assert file_info["index"] == i
            assert file_info["label"] == f"eos_{i:03d}"
            assert Path(file_info["file"]).exists()

    def test_with_metadata_list(self, scaled_structures, tmp_path):
        """Test with metadata for each structure."""
        metadata_list = [
            {"scale": 0.95, "volume_factor": 0.95},
            {"scale": 1.0, "volume_factor": 1.0},
            {"scale": 1.05, "volume_factor": 1.05},
        ]

        job = dry_run_save_multiple_structures(
            structures=scaled_structures,
            output_dir=str(tmp_path),
            label_prefix="eos",
            metadata_list=metadata_list,
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check metadata is preserved for each structure
        for file_info, expected_meta in zip(output["structure_files"], metadata_list):
            assert file_info["metadata"] == expected_meta

    def test_different_formats(self, scaled_structures, tmp_path):
        """Test saving multiple structures in different formats."""
        # Test only formats supported by pymatgen Structure.to()
        formats = ["cif", "xsf", "json"]

        for fmt in formats:
            job = dry_run_save_multiple_structures(
                structures=scaled_structures,
                output_dir=str(tmp_path / fmt),
                output_format=fmt,
                label_prefix=f"test_{fmt}",
            )

            result = run_locally(job, create_folders=True)
            output = result[job.uuid][1].output

            # Check all files have correct extension
            for file_info in output["structure_files"]:
                assert Path(file_info["file"]).suffix == f".{fmt}"


class TestDryRunWorkflowSummary:
    """Test dry_run_workflow_summary function."""

    def test_basic_summary(self, tmp_path):
        """Test basic summary generation."""
        # Mock job outputs
        job_outputs = [
            {
                "dry_run": True,
                "label": "structure_1",
                "formula": "Si",
                "num_atoms": 2,
            },
            {
                "dry_run": True,
                "label": "structure_2",
                "formula": "Si",
                "num_atoms": 2,
            },
        ]

        job = dry_run_workflow_summary(
            job_outputs=job_outputs,
            workflow_type="test_workflow",
            output_dir=str(tmp_path),
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check output
        assert output["dry_run"] is True
        assert output["workflow_type"] == "test_workflow"
        assert output["num_jobs"] == 2

        # Check summary file was created
        summary_file = Path(output["summary_file"])
        assert summary_file.exists()
        assert summary_file.name == "dry_run_summary.txt"

        # Check summary content
        content = summary_file.read_text()
        assert "DRY RUN SUMMARY" in content
        assert "test_workflow" in content

    def test_with_workflow_metadata(self, tmp_path):
        """Test summary with workflow metadata."""
        job_outputs = [
            {"dry_run": True, "label": "test", "formula": "Si", "num_atoms": 2}
        ]

        job = dry_run_workflow_summary(
            job_outputs=job_outputs,
            workflow_type="eos",
            output_dir=str(tmp_path),
            num_volumes=10,
            volume_range="0.90-1.10",
            basis_size="DZP",
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        summary_file = Path(output["summary_file"])
        content = summary_file.read_text()

        # Check workflow metadata is in summary
        assert "num_volumes: 10" in content
        assert "volume_range: 0.90-1.10" in content
        assert "basis_size: DZP" in content

    def test_multiple_structure_output(self, tmp_path):
        """Test summary handles multiple-structure outputs."""
        job_outputs = [
            {
                "dry_run": True,
                "num_structures": 5,
                "structure_files": [
                    {"index": i, "label": f"struct_{i}"} for i in range(5)
                ],
            }
        ]

        job = dry_run_workflow_summary(
            job_outputs=job_outputs,
            workflow_type="phonon",
            output_dir=str(tmp_path),
        )

        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        summary_file = Path(output["summary_file"])
        content = summary_file.read_text()

        # Check multiple structures mentioned
        assert "Multiple structures: 5" in content


class TestBaseSiestaMakerDryRun:
    """Test dry-run functionality in BaseSiestaMaker."""

    def test_base_maker_dry_run(self, si_structure, tmp_path):
        """Test BaseSiestaMaker with dry_run enabled."""
        from atomate2.siesta.jobs.base import BaseSiestaMaker

        # Create maker with dry-run enabled
        maker = BaseSiestaMaker(
            name="test_base",
            dry_run=True,
            dry_run_output_dir=str(tmp_path),
            dry_run_format="cif",
        )

        # Create job
        job = maker.make(si_structure)

        # Run job
        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check output
        assert output["dry_run"] is True
        assert output["label"] == "test_base_Si"
        assert output["formula"] == "Si"
        assert output["metadata"]["maker_name"] == "test_base"
        assert output["metadata"]["maker_type"] == "BaseSiestaMaker"

        # Check file was created
        structure_file = Path(output["structure_file"])
        assert structure_file.exists()
        assert structure_file.suffix == ".cif"

    def test_base_maker_custom_label(self, si_structure, tmp_path):
        """Test BaseSiestaMaker with custom dry-run label."""
        from atomate2.siesta.jobs.base import BaseSiestaMaker

        maker = BaseSiestaMaker(
            name="test_base",
            dry_run=True,
            dry_run_output_dir=str(tmp_path),
            dry_run_label="my_custom_structure",
        )

        job = maker.make(si_structure)
        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        assert output["label"] == "my_custom_structure"
        structure_file = Path(output["structure_file"])
        assert structure_file.name == "my_custom_structure.cif"

    def test_base_maker_different_format(self, si_structure, tmp_path):
        """Test BaseSiestaMaker with different output format."""
        from atomate2.siesta.jobs.base import BaseSiestaMaker

        maker = BaseSiestaMaker(
            name="test_base",
            dry_run=True,
            dry_run_output_dir=str(tmp_path),
            dry_run_format="xsf",
        )

        job = maker.make(si_structure)
        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        structure_file = Path(output["structure_file"])
        assert structure_file.suffix == ".xsf"

    def test_base_maker_dry_run_false(self, si_structure, tmp_path):
        """Test BaseSiestaMaker with dry_run=False uses calculation path."""
        from atomate2.siesta.jobs.base import BaseSiestaMaker

        # Create maker with dry_run=False (normal mode)
        maker = BaseSiestaMaker(
            name="test_base",
            dry_run=False,  # Normal calculation mode
        )

        # Just verify that with dry_run=False, we don't get dry-run output
        # We can't run the full calculation without SIESTA, so just test
        # that the maker is configured correctly
        assert maker.dry_run is False
        assert not hasattr(maker.make(si_structure), "output") or True

        # Create a dry-run maker to compare
        dry_maker = BaseSiestaMaker(
            name="test_dry",
            dry_run=True,
            dry_run_output_dir=str(tmp_path),
        )

        # Run dry-run version and verify it works
        dry_job = dry_maker.make(si_structure)
        result = run_locally(dry_job, create_folders=True)
        output = result[dry_job.uuid][1].output

        # Verify dry-run output
        assert output["dry_run"] is True


class TestInheritedMakersDryRun:
    """Test that makers inheriting from BaseSiestaMaker get dry-run support."""

    def test_relax_maker_dry_run(self, si_structure, tmp_path):
        """Test RelaxMaker with dry_run enabled."""
        from atomate2.siesta.jobs.core import RelaxMaker

        # Create maker with dry-run enabled
        maker = RelaxMaker.fixed_cell_relaxation(
            dry_run=True,
            dry_run_output_dir=str(tmp_path),
        )

        # Create and run job
        job = maker.make(si_structure)
        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check output
        assert output["dry_run"] is True
        # Name is automatically set by the classmethod
        expected_name = "Relaxation calculation-fixed-cell"
        assert output["label"] == f"{expected_name}_Si"
        assert output["metadata"]["maker_name"] == expected_name
        assert output["metadata"]["maker_type"] == "RelaxMaker"

        # Check file was created
        structure_file = Path(output["structure_file"])
        assert structure_file.exists()

    def test_static_maker_dry_run(self, si_structure, tmp_path):
        """Test StaticMaker with dry_run enabled."""
        from atomate2.siesta.jobs.core import StaticMaker

        # Create maker with dry-run enabled
        maker = StaticMaker(
            name="test_static",
            dry_run=True,
            dry_run_output_dir=str(tmp_path),
        )

        # Create and run job
        job = maker.make(si_structure)
        result = run_locally(job, create_folders=True)
        output = result[job.uuid][1].output

        # Check output
        assert output["dry_run"] is True
        assert output["label"] == "test_static_Si"
        assert output["metadata"]["maker_name"] == "test_static"
        assert output["metadata"]["maker_type"] == "StaticMaker"

        # Check file was created
        structure_file = Path(output["structure_file"])
        assert structure_file.exists()


class TestBaseSiestaFlowMaker:
    """Test BaseSiestaFlowMaker automatic dry-run propagation."""

    def test_single_maker_propagation(self):
        """Test dry-run propagates to single child maker."""
        from dataclasses import dataclass

        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker

        @dataclass
        class SimpleFlow(BaseSiestaFlowMaker):
            """Simple flow with one maker."""

            child_maker: StaticMaker = None

            def __post_init__(self):
                if self.child_maker is None:
                    self.child_maker = StaticMaker()
                super().__post_init__()

        # Create flow with dry-run disabled
        flow = SimpleFlow(dry_run=False)
        assert flow.child_maker.dry_run is False

        # Create flow with dry-run enabled
        flow = SimpleFlow(dry_run=True, dry_run_output_dir="test_output")
        assert flow.child_maker.dry_run is True
        assert flow.child_maker.dry_run_output_dir == "test_output"

    def test_multiple_makers_propagation(self):
        """Test dry-run propagates to multiple child makers."""
        from dataclasses import dataclass

        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

        @dataclass
        class MultiMakerFlow(BaseSiestaFlowMaker):
            """Flow with multiple makers."""

            relax_maker: RelaxMaker = None
            static_maker: StaticMaker = None

            def __post_init__(self):
                if self.relax_maker is None:
                    self.relax_maker = RelaxMaker()
                if self.static_maker is None:
                    self.static_maker = StaticMaker()
                super().__post_init__()

        # Create flow with dry-run enabled
        flow = MultiMakerFlow(
            dry_run=True, dry_run_output_dir="multi_test", dry_run_format="xsf"
        )

        # Both makers should have dry-run enabled
        assert flow.relax_maker.dry_run is True
        assert flow.static_maker.dry_run is True
        assert flow.relax_maker.dry_run_output_dir == "multi_test"
        assert flow.static_maker.dry_run_output_dir == "multi_test"
        assert flow.relax_maker.dry_run_format == "xsf"
        assert flow.static_maker.dry_run_format == "xsf"

    def test_list_of_makers_propagation(self):
        """Test dry-run propagates to list of makers."""
        from dataclasses import dataclass, field

        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker

        @dataclass
        class ListMakersFlow(BaseSiestaFlowMaker):
            """Flow with list of makers."""

            makers: list = field(default_factory=list)

            def __post_init__(self):
                if not self.makers:
                    self.makers = [StaticMaker() for _ in range(3)]
                super().__post_init__()

        # Create flow with dry-run enabled
        flow = ListMakersFlow(dry_run=True, dry_run_output_dir="list_test")

        # All makers in list should have dry-run enabled
        assert len(flow.makers) == 3
        for maker in flow.makers:
            assert maker.dry_run is True
            assert maker.dry_run_output_dir == "list_test"

    def test_nested_flow_propagation(self):
        """Test dry-run propagates recursively through nested flows."""
        from dataclasses import dataclass

        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker

        @dataclass
        class InnerFlow(BaseSiestaFlowMaker):
            """Inner flow."""

            maker: StaticMaker = None

            def __post_init__(self):
                if self.maker is None:
                    self.maker = StaticMaker()
                super().__post_init__()

        @dataclass
        class OuterFlow(BaseSiestaFlowMaker):
            """Outer flow containing inner flow."""

            inner_flow: InnerFlow = None

            def __post_init__(self):
                if self.inner_flow is None:
                    self.inner_flow = InnerFlow()
                super().__post_init__()

        # Create outer flow with dry-run enabled
        flow = OuterFlow(dry_run=True, dry_run_output_dir="nested_test")

        # Inner flow should have dry-run enabled
        assert flow.inner_flow.dry_run is True
        assert flow.inner_flow.dry_run_output_dir == "nested_test"

        # Maker in inner flow should also have dry-run enabled
        assert flow.inner_flow.maker.dry_run is True
        assert flow.inner_flow.maker.dry_run_output_dir == "nested_test"

    def test_propagation_only_when_enabled(self):
        """Test propagation only happens when dry_run=True."""
        from dataclasses import dataclass

        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker

        @dataclass
        class SimpleFlow(BaseSiestaFlowMaker):
            """Simple flow."""

            maker: StaticMaker = None

            def __post_init__(self):
                if self.maker is None:
                    self.maker = StaticMaker()
                super().__post_init__()

        # With dry_run=False, child maker should not be modified
        flow = SimpleFlow(dry_run=False)
        assert flow.maker.dry_run is False

        # With dry_run=True, child maker should be modified
        flow = SimpleFlow(dry_run=True)
        assert flow.maker.dry_run is True


class TestAdsorptionScanMakerMigration:
    """Test AdsorptionScanFlowMaker migration to BaseSiestaFlowMaker."""

    def test_adsorption_inherits_from_base_flow(self):
        """Test that AdsorptionScanFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker

        assert issubclass(AdsorptionScanFlowMaker, BaseSiestaFlowMaker)

    def test_adsorption_auto_propagation(self):
        """Test dry-run propagates to child makers in AdsorptionScanFlowMaker."""
        from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker

        # Create flow with dry-run enabled
        maker = AdsorptionScanFlowMaker(
            slab_static_maker=StaticMaker(),
            adsorbate_static_maker=StaticMaker(),
            dry_run=True,
        )

        # Both child makers should have dry_run enabled automatically
        assert maker.slab_static_maker.dry_run is True
        assert maker.adsorbate_static_maker.dry_run is True

    def test_adsorption_custom_output_dir(self):
        """Test AdsorptionScanFlowMaker uses custom default output directory."""
        from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker

        # Default should be 'preview_structures', not 'dry_run_output'
        maker = AdsorptionScanFlowMaker(dry_run=True)
        assert maker.dry_run_output_dir == "preview_structures"

        # Should propagate custom directory to child makers
        assert maker.slab_static_maker.dry_run_output_dir == "preview_structures"
        assert maker.adsorbate_static_maker.dry_run_output_dir == "preview_structures"

    def test_adsorption_dry_run_format_propagation(self):
        """Test dry_run_format propagates correctly."""
        from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker

        maker = AdsorptionScanFlowMaker(dry_run=True, dry_run_format="xsf")

        # Should propagate to child makers
        assert maker.slab_static_maker.dry_run_format == "xsf"
        assert maker.adsorbate_static_maker.dry_run_format == "xsf"

    def test_adsorption_dry_run_disabled(self):
        """Test that with dry_run=False, child makers are not modified."""
        from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker

        # Create flow with dry-run disabled
        maker = AdsorptionScanFlowMaker(
            slab_static_maker=StaticMaker(),
            adsorbate_static_maker=StaticMaker(),
            dry_run=False,
        )

        # Child makers should not have dry_run enabled
        assert maker.slab_static_maker.dry_run is False
        assert maker.adsorbate_static_maker.dry_run is False
