"""
Tests for adsorption workflows.

These tests validate:
- AdsorptionScanFlowMaker flow creation
- Adsorption site scanning workflow
- Molecule orientation parameters
- Top/bottom placement
- Plot and summary generation
- AdsorptionOptimizationFlowMaker flow
"""

import pytest
from pymatgen.core import Structure, Molecule, Lattice

from atomate2.siesta.flows.surface.adsorption import (
    AdsorptionScanFlowMaker,
    AdsorptionOptimizationFlowMaker,
)
from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from jobflow import Flow


@pytest.fixture
def simple_slab():
    """Create a simple cubic slab for testing."""
    lattice = Lattice.cubic(5.0)
    return Structure(
        lattice,
        ["Al", "Al", "Al", "Al"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.2],
            [0.5, 0.5, 0.2],
        ],
    )


@pytest.fixture
def co_molecule():
    """Create CO molecule for testing."""
    return Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])


class TestAdsorptionScanMaker:
    """Tests for AdsorptionScanFlowMaker workflow."""

    def test_default_adsorption_scan_maker(self):
        """Test creation of default AdsorptionScanFlowMaker."""
        maker = AdsorptionScanFlowMaker()

        assert maker.name == "adsorption_scan"
        assert maker.grid_size == (4, 4)
        assert maker.height == 2.0
        assert maker.plot_results is True
        assert maker.write_summary is True
        assert maker.placement == "top"
        assert isinstance(maker.slab_static_maker, StaticMaker)
        assert isinstance(maker.adsorbate_static_maker, StaticMaker)

    def test_adsorption_scan_with_custom_grid(self):
        """Test AdsorptionScanFlowMaker with custom grid size."""
        maker = AdsorptionScanFlowMaker(grid_size=(5, 5))

        assert maker.grid_size == (5, 5)

    def test_adsorption_scan_with_custom_height(self):
        """Test AdsorptionScanFlowMaker with custom adsorbate height."""
        maker = AdsorptionScanFlowMaker(height=3.0)

        assert maker.height == 3.0

    def test_adsorption_scan_with_miller_indices(self):
        """Test AdsorptionScanFlowMaker with Miller indices."""
        maker = AdsorptionScanFlowMaker(miller_indices=(1, 1, 1))

        assert maker.miller_indices == (1, 1, 1)

    def test_adsorption_scan_bottom_placement(self):
        """Test AdsorptionScanFlowMaker with bottom placement."""
        maker = AdsorptionScanFlowMaker(placement="bottom")

        assert maker.placement == "bottom"

    def test_adsorption_scan_with_custom_makers(self, simple_slab, co_molecule):
        """Test AdsorptionScanFlowMaker with custom StaticMakers."""
        slab_params = {"PAO.BasisSize": "DZP", "a2s_kpts": [6, 6, 1]}
        ads_params = {"PAO.BasisSize": "TZP"}

        slab_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=slab_params)
        )
        ads_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=ads_params)
        )

        maker = AdsorptionScanFlowMaker(
            slab_static_maker=slab_maker,
            adsorbate_static_maker=ads_maker,
        )

        assert maker.slab_static_maker == slab_maker
        assert maker.adsorbate_static_maker == ads_maker

    def test_adsorption_scan_make_flow(self, simple_slab, co_molecule):
        """Test that AdsorptionScanFlowMaker.make() creates a valid flow."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            height=2.0,
            plot_results=False,
            write_summary=False,
        )

        flow = maker.make(simple_slab, co_molecule)

        # Check flow structure
        assert isinstance(flow, Flow)
        assert flow.name == "adsorption_scan"
        assert len(flow) > 0  # Should have jobs

        # With 2x2 grid: slab(1) + ads(1) + generate(1) + 4 sites + analyze(1) = 8 jobs
        # Note: Each site job creates a nested flow, so len() counts top-level jobs only
        expected_jobs = 1 + 1 + 1 + 4 + 1
        assert len(flow) == expected_jobs

    def test_adsorption_scan_with_plots_and_summary(self, simple_slab, co_molecule):
        """Test flow with plotting and summary enabled."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            plot_results=True,
            write_summary=True,
        )

        flow = maker.make(simple_slab, co_molecule)

        # Plotting and summary are consolidated into the single analysis job,
        # so they do not add top-level jobs to the flow.
        expected_jobs = 1 + 1 + 1 + 4 + 1  # slab + ads + generate + sites + analysis
        assert len(flow) == expected_jobs

    def test_adsorption_scan_only_plots(self, simple_slab, co_molecule):
        """Test flow with only plotting, no summary."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            plot_results=True,
            write_summary=False,
        )

        flow = maker.make(simple_slab, co_molecule)

        # Plotting is consolidated into the analysis job (no extra top-level job).
        expected_jobs = 1 + 1 + 1 + 4 + 1
        assert len(flow) == expected_jobs

    def test_adsorption_scan_only_summary(self, simple_slab, co_molecule):
        """Test flow with only summary, no plotting."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            plot_results=False,
            write_summary=True,
        )

        flow = maker.make(simple_slab, co_molecule)

        # Summary is consolidated into the analysis job (no extra top-level job).
        expected_jobs = 1 + 1 + 1 + 4 + 1
        assert len(flow) == expected_jobs

    def test_adsorption_scan_different_grid_sizes(self, simple_slab, co_molecule):
        """Test flow creation with different grid sizes."""
        grid_sizes = [(2, 2), (3, 3), (4, 4), (2, 5)]

        for grid_size in grid_sizes:
            maker = AdsorptionScanFlowMaker(
                grid_size=grid_size,
                plot_results=False,
                write_summary=False,
            )

            flow = maker.make(simple_slab, co_molecule)

            total_sites = grid_size[0] * grid_size[1]
            expected_jobs = 1 + 1 + 1 + total_sites + 1  # Top-level jobs only
            assert len(flow) == expected_jobs

    def test_adsorption_scan_with_molecule_orientation(self, simple_slab, co_molecule):
        """Test AdsorptionScanFlowMaker with molecule orientation parameters."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            target_vector=[0.0, 0.0, 1.0],
            plane_atoms=[0, 1, 2],
            extra_rotation=45.0,
            rotation_axis=[0.0, 0.0, 1.0],
        )

        flow = maker.make(simple_slab, co_molecule)

        # Flow should still be created successfully
        assert isinstance(flow, Flow)
        assert len(flow) > 0

    def test_adsorption_scan_with_custom_mol_file(
        self, simple_slab, co_molecule, tmp_path
    ):
        """Test AdsorptionScanFlowMaker with custom molecule file."""
        # Create a dummy XYZ file
        mol_file = tmp_path / "test_mol.xyz"
        mol_file.write_text(
            """2
CO molecule
C  0.0  0.0  0.0
O  0.0  0.0  1.13
"""
        )

        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            custom_mol_file=str(mol_file),
        )

        flow = maker.make(simple_slab, co_molecule)

        # Flow should be created
        assert isinstance(flow, Flow)

    def test_adsorption_scan_with_prev_dir(self, simple_slab, co_molecule, tmp_path):
        """Test AdsorptionScanFlowMaker with previous directory."""
        maker = AdsorptionScanFlowMaker(grid_size=(2, 2))

        prev_dir = tmp_path / "previous"
        prev_dir.mkdir()

        flow = maker.make(simple_slab, co_molecule, prev_dir=str(prev_dir))

        assert isinstance(flow, Flow)

    def test_adsorption_scan_flow_has_output(self, simple_slab, co_molecule):
        """Test that flow has expected output."""
        maker = AdsorptionScanFlowMaker(grid_size=(2, 2))

        flow = maker.make(simple_slab, co_molecule)

        # Flow should have output (the analysis job output)
        assert hasattr(flow, "output")
        assert flow.output is not None

    def test_adsorption_scan_serialization(self, simple_slab, co_molecule):
        """Test that AdsorptionScanFlowMaker can be serialized."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(3, 3),
            height=2.5,
            miller_indices=(1, 0, 0),
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "AdsorptionScanFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, AdsorptionScanFlowMaker)
        # grid_size may become list after serialization
        assert list(maker_restored.grid_size) == [3, 3]
        assert maker_restored.height == 2.5


class TestAdsorptionOptimizationMaker:
    """Tests for AdsorptionOptimizationFlowMaker workflow."""

    def test_default_optimization_maker(self):
        """Test creation of default AdsorptionOptimizationFlowMaker."""
        maker = AdsorptionOptimizationFlowMaker()

        assert maker.name == "adsorption_optimization"
        assert maker.relax_adsorbate_only is True
        assert isinstance(maker.relax_maker, RelaxMaker)
        assert isinstance(maker.final_static_maker, StaticMaker)

    def test_optimization_maker_with_custom_makers(self):
        """Test AdsorptionOptimizationFlowMaker with custom makers."""
        relax_maker = RelaxMaker.fixed_cell_relaxation(
            user_params={"PAO.BasisSize": "DZP"}
        )
        static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "TZP"})
        )

        maker = AdsorptionOptimizationFlowMaker(
            relax_maker=relax_maker,
            final_static_maker=static_maker,
        )

        assert maker.relax_maker == relax_maker
        assert maker.final_static_maker == static_maker

    def test_optimization_maker_relax_all_atoms(self):
        """Test optimization with all atoms relaxed."""
        maker = AdsorptionOptimizationFlowMaker(relax_adsorbate_only=False)

        assert maker.relax_adsorbate_only is False

    def test_optimization_make_flow(self, simple_slab, co_molecule):
        """Test that AdsorptionOptimizationFlowMaker.make() creates valid flow."""
        maker = AdsorptionOptimizationFlowMaker()

        flow = maker.make(
            slab=simple_slab,
            adsorbate=co_molecule,
            best_site=(0.5, 0.5),
            height=2.0,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            initial_adsorption_energy=-2.0,
        )

        # Check flow structure
        assert isinstance(flow, Flow)
        assert flow.name == "adsorption_optimization"

        # Expected jobs:
        # 1. create_ads_structure
        # 2. add_constraints
        # 3. relax
        # 4. final_static
        # 5. analyze_optimization
        assert len(flow) == 5

    def test_optimization_flow_has_output(self, simple_slab, co_molecule):
        """Test that optimization flow has output."""
        maker = AdsorptionOptimizationFlowMaker()

        flow = maker.make(
            slab=simple_slab,
            adsorbate=co_molecule,
            best_site=(0.25, 0.75),
            height=2.0,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            initial_adsorption_energy=-2.0,
        )

        assert hasattr(flow, "output")
        assert flow.output is not None

    def test_optimization_different_sites(self, simple_slab, co_molecule):
        """Test optimization flow with different best sites."""
        sites = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]

        maker = AdsorptionOptimizationFlowMaker()

        for site in sites:
            flow = maker.make(
                slab=simple_slab,
                adsorbate=co_molecule,
                best_site=site,
                height=2.0,
                slab_energy=-95.0,
                adsorbate_energy=-3.0,
                initial_adsorption_energy=-2.0,
            )

            assert isinstance(flow, Flow)
            assert len(flow) == 5

    def test_optimization_serialization(self):
        """Test that AdsorptionOptimizationFlowMaker can be serialized."""
        maker = AdsorptionOptimizationFlowMaker(
            relax_adsorbate_only=False,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "AdsorptionOptimizationFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, AdsorptionOptimizationFlowMaker)
        assert maker_restored.relax_adsorbate_only is False


class TestAdsorptionFlowIntegration:
    """Integration tests for complete adsorption workflows."""

    def test_scan_then_optimization(self, simple_slab, co_molecule):
        """Test chaining scan and optimization workflows."""
        # Create scan workflow
        scan_maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            plot_results=False,
            write_summary=False,
        )
        scan_flow = scan_maker.make(simple_slab, co_molecule)

        # Create optimization workflow
        opt_maker = AdsorptionOptimizationFlowMaker()

        # In a real workflow, optimization would use scan results
        # Here we just verify both flows can be created
        opt_flow = opt_maker.make(
            slab=simple_slab,
            adsorbate=co_molecule,
            best_site=(0.5, 0.5),
            height=2.0,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            initial_adsorption_energy=-2.0,
        )

        assert isinstance(scan_flow, Flow)
        assert isinstance(opt_flow, Flow)

    def test_multiple_grid_sizes(self, simple_slab, co_molecule):
        """Test scanning with multiple grid sizes."""
        grid_sizes = [(2, 2), (3, 3), (4, 4)]

        for grid_size in grid_sizes:
            maker = AdsorptionScanFlowMaker(
                grid_size=grid_size,
                plot_results=True,
                write_summary=True,
            )

            flow = maker.make(simple_slab, co_molecule)

            # Verify flow is created correctly
            assert isinstance(flow, Flow)

            # Expected top-level jobs (plot/summary are consolidated into analysis)
            total_sites = grid_size[0] * grid_size[1]
            expected = 1 + 1 + 1 + total_sites + 1
            assert len(flow) == expected

    def test_all_placement_options(self, simple_slab, co_molecule):
        """Test both top and bottom placement."""
        for placement in ["top", "bottom"]:
            maker = AdsorptionScanFlowMaker(
                grid_size=(2, 2),
                placement=placement,
                plot_results=False,
                write_summary=False,
            )

            flow = maker.make(simple_slab, co_molecule)

            assert isinstance(flow, Flow)
            assert maker.placement == placement

    def test_with_all_orientation_parameters(self, simple_slab, co_molecule):
        """Test scan with all molecule orientation parameters."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(2, 2),
            target_vector=[0.0, 0.0, 1.0],
            plane_atoms=[0, 1, 2],
            extra_rotation=90.0,
            rotation_axis=[1.0, 0.0, 0.0],
            placement="top",
        )

        flow = maker.make(simple_slab, co_molecule)

        assert isinstance(flow, Flow)
        assert maker.target_vector == [0.0, 0.0, 1.0]
        assert maker.plane_atoms == [0, 1, 2]
        assert maker.extra_rotation == 90.0
        assert maker.rotation_axis == [1.0, 0.0, 0.0]

    def test_optimization_with_both_relax_modes(self, simple_slab, co_molecule):
        """Test optimization with both relax modes."""
        for relax_adsorbate_only in [True, False]:
            maker = AdsorptionOptimizationFlowMaker(
                relax_adsorbate_only=relax_adsorbate_only
            )

            flow = maker.make(
                slab=simple_slab,
                adsorbate=co_molecule,
                best_site=(0.5, 0.5),
                height=2.0,
                slab_energy=-95.0,
                adsorbate_energy=-3.0,
                initial_adsorption_energy=-2.0,
            )

            assert isinstance(flow, Flow)
            assert len(flow) == 5


# ==================== Additional Adsorption Tests ====================


class TestAdsorptionDryRun:
    """Test dry-run mode for adsorption workflows."""

    def test_adsorption_scan_with_dry_run_enabled(self, simple_slab, co_molecule):
        """Test AdsorptionScanFlowMaker with dry_run=True."""
        maker = AdsorptionScanFlowMaker(
            dry_run=True,
            grid_size=(2, 2),
            plot_results=False,
            write_summary=False,
        )

        flow = maker.make(simple_slab, co_molecule)

        assert isinstance(flow, Flow)
        assert maker.dry_run is True

    def test_adsorption_scan_dry_run_default_false(self):
        """Test that dry_run defaults to False for AdsorptionScanFlowMaker."""
        maker = AdsorptionScanFlowMaker()
        assert maker.dry_run is False

    def test_adsorption_optimization_with_dry_run_enabled(
        self, simple_slab, co_molecule
    ):
        """Test AdsorptionOptimizationFlowMaker with dry_run=True."""
        maker = AdsorptionOptimizationFlowMaker(dry_run=True)

        flow = maker.make(
            slab=simple_slab,
            adsorbate=co_molecule,
            best_site=(0.5, 0.5),
            height=2.0,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            initial_adsorption_energy=-2.0,
        )

        assert isinstance(flow, Flow)
        assert maker.dry_run is True

    def test_adsorption_optimization_dry_run_default_false(self):
        """Test that dry_run defaults to False for AdsorptionOptimizationFlowMaker."""
        maker = AdsorptionOptimizationFlowMaker()
        assert maker.dry_run is False


class TestAdsorptionInheritance:
    """Test BaseSiestaFlowMaker inheritance for adsorption makers."""

    def test_adsorption_scan_inherits_from_base_siesta_flow_maker(self):
        """Test that AdsorptionScanFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = AdsorptionScanFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_adsorption_scan_has_dry_run_attribute(self):
        """Test that AdsorptionScanFlowMaker has dry_run attribute."""
        maker = AdsorptionScanFlowMaker()
        assert hasattr(maker, "dry_run")

    def test_adsorption_optimization_inherits_from_base_siesta_flow_maker(self):
        """Test that AdsorptionOptimizationFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = AdsorptionOptimizationFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_adsorption_optimization_has_dry_run_attribute(self):
        """Test that AdsorptionOptimizationFlowMaker has dry_run attribute."""
        maker = AdsorptionOptimizationFlowMaker()
        assert hasattr(maker, "dry_run")


class TestAdsorptionEdgeCases:
    """Test edge cases for adsorption workflows."""

    def test_adsorption_scan_large_grid(self, simple_slab, co_molecule):
        """Test adsorption scan with large grid size (10×10 = 100 sites)."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(10, 10),
            plot_results=False,
            write_summary=False,
        )

        flow = maker.make(simple_slab, co_molecule)

        assert isinstance(flow, Flow)
        assert maker.grid_size == (10, 10)
        # Should have: slab(1) + ads(1) + generate(1) + 100 sites + analyze(1)
        expected_jobs = 1 + 1 + 1 + 100 + 1
        assert len(flow) == expected_jobs

    def test_adsorption_scan_single_site(self, simple_slab, co_molecule):
        """Test adsorption scan with minimal grid size (1×1)."""
        maker = AdsorptionScanFlowMaker(
            grid_size=(1, 1),
            plot_results=False,
            write_summary=False,
        )

        flow = maker.make(simple_slab, co_molecule)

        assert isinstance(flow, Flow)
        assert maker.grid_size == (1, 1)
        # Should have: slab(1) + ads(1) + generate(1) + 1 site + analyze(1)
        expected_jobs = 1 + 1 + 1 + 1 + 1
        assert len(flow) == expected_jobs
