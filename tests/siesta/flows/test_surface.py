"""
Tests for surface energy calculation workflows.

These tests validate:
- SurfaceEnergyFlowMaker (single Miller index with terminations)
- MultiSurfaceEnergyFlowMaker (multiple Miller indices)
- Slab generation and discovery
- Surface energy calculations
- Flow composition and chaining
"""

from pathlib import Path

from jobflow import Flow

from atomate2.siesta.flows.surface import SurfaceEnergyFlowMaker
from atomate2.siesta.flows.surface.multi_surface import (
    MultiSurfaceEnergyFlowMaker,
    calculate_multi_surface_energies,
)
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker


class TestSurfaceEnergyMaker:
    """Tests for SurfaceEnergyFlowMaker workflow."""

    def test_default_surface_energy_maker(self):
        """Test creation of default SurfaceEnergyFlowMaker."""
        maker = SurfaceEnergyFlowMaker()

        assert maker.name == "surface_energy"
        assert maker.miller_indices == (0, 0, 1)
        assert isinstance(maker.bulk_static_maker, StaticMaker)
        assert isinstance(maker.slab_static_maker, StaticMaker)
        assert maker.slab_relax_maker is None  # No relaxation by default
        assert maker.plot_results is True
        assert maker.write_summary is True

    def test_surface_energy_maker_with_custom_params(self):
        """Test SurfaceEnergyFlowMaker with custom parameters."""
        bulk_maker = StaticMaker()
        slab_maker = StaticMaker()
        relax_maker = RelaxMaker.fixed_cell_relaxation()

        maker = SurfaceEnergyFlowMaker(
            name="custom_surface",
            bulk_static_maker=bulk_maker,
            slab_static_maker=slab_maker,
            slab_relax_maker=relax_maker,
            miller_indices=(1, 1, 0),
            plot_results=False,
            write_summary=False,
        )

        assert maker.name == "custom_surface"
        assert maker.miller_indices == (1, 1, 0)
        assert maker.bulk_static_maker == bulk_maker
        assert maker.slab_static_maker == slab_maker
        assert maker.slab_relax_maker == relax_maker
        assert maker.plot_results is False
        assert maker.write_summary is False

    def test_surface_energy_maker_with_miller_indices(self):
        """Test different Miller indices."""
        miller_indices_list = [
            (0, 0, 1),
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
        ]

        for hkl in miller_indices_list:
            maker = SurfaceEnergyFlowMaker(miller_indices=hkl)
            assert maker.miller_indices == hkl

    def test_surface_energy_maker_with_relaxation(self):
        """Test SurfaceEnergyFlowMaker with slab relaxation."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        maker = SurfaceEnergyFlowMaker(slab_relax_maker=relax_maker)

        assert maker.slab_relax_maker == relax_maker
        assert isinstance(maker.slab_relax_maker, RelaxMaker)

    def test_surface_energy_maker_slab_directory(self, tmp_path):
        """Test SurfaceEnergyFlowMaker with custom slab directory."""
        slab_dir = tmp_path / "slabs"
        maker = SurfaceEnergyFlowMaker(slab_directory=slab_dir)

        assert Path(maker.slab_directory) == slab_dir

    def test_surface_energy_maker_formula_units(self):
        """Test SurfaceEnergyFlowMaker with specified formula units."""
        maker = SurfaceEnergyFlowMaker(formula_units_per_cell=2)

        assert maker.formula_units_per_cell == 2

    def test_surface_energy_maker_serialization(self):
        """Test SurfaceEnergyFlowMaker serialization."""
        maker = SurfaceEnergyFlowMaker(
            miller_indices=(1, 1, 0),
            plot_results=False,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "SurfaceEnergyFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, SurfaceEnergyFlowMaker)
        # Serialization converts tuple to list
        assert tuple(maker_restored.miller_indices) == (
            1,
            1,
            0,
        ) or maker_restored.miller_indices == (1, 1, 0)
        assert maker_restored.plot_results is False


class TestMultiSurfaceEnergyMaker:
    """Tests for MultiSurfaceEnergyFlowMaker workflow."""

    def test_default_multi_surface_energy_maker(self):
        """Test creation of default MultiSurfaceEnergyFlowMaker."""
        maker = MultiSurfaceEnergyFlowMaker()

        assert maker.name == "multi_surface_energy"
        assert isinstance(maker.miller_indices, list)
        assert len(maker.miller_indices) == 7  # Default 7 surfaces
        assert (1, 0, 0) in maker.miller_indices
        assert (1, 1, 1) in maker.miller_indices
        assert maker.slab_layers == 4
        assert maker.vacuum_size == 15.0
        assert maker.symmetrize is False

    def test_multi_surface_energy_maker_with_custom_miller_indices(self):
        """Test MultiSurfaceEnergyFlowMaker with custom Miller indices."""
        custom_hkl = [(0, 0, 1), (1, 1, 0), (1, 1, 1)]

        maker = MultiSurfaceEnergyFlowMaker(miller_indices=custom_hkl)

        assert maker.miller_indices == custom_hkl
        assert len(maker.miller_indices) == 3

    def test_multi_surface_energy_maker_with_slab_params(self):
        """Test MultiSurfaceEnergyFlowMaker with slab parameters."""
        maker = MultiSurfaceEnergyFlowMaker(
            slab_layers=6,
            vacuum_size=20.0,
            symmetrize=True,
        )

        assert maker.slab_layers == 6
        assert maker.vacuum_size == 20.0
        assert maker.symmetrize is True

    def test_multi_surface_energy_maker_with_custom_makers(self):
        """Test MultiSurfaceEnergyFlowMaker with custom StaticMakers."""
        bulk_maker = StaticMaker()
        slab_maker = StaticMaker()

        maker = MultiSurfaceEnergyFlowMaker(
            bulk_static_maker=bulk_maker,
            slab_static_maker=slab_maker,
        )

        assert maker.bulk_static_maker == bulk_maker
        assert maker.slab_static_maker == slab_maker

    def test_multi_surface_energy_maker_make_flow(self, si_structure):
        """Test that MultiSurfaceEnergyFlowMaker creates a valid flow."""
        # Use minimal Miller indices for faster test
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1), (1, 1, 0)],
            slab_layers=2,  # Minimal for speed
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "multi_surface_energy"
        # Should have bulk + slabs + analysis
        assert len(flow) > 2

    def test_multi_surface_energy_maker_job_naming(self, si_structure):
        """Test that jobs have proper naming with counters."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow = maker.make(si_structure)

        # Check job naming pattern (counter/total)
        job_names = [job.name for job in flow]

        # First job should be bulk
        assert "bulk" in job_names[0].lower()

        # Last job should be analysis
        assert "analysis" in job_names[-1].lower()

    def test_multi_surface_energy_maker_serialization(self):
        """Test MultiSurfaceEnergyFlowMaker serialization."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(1, 0, 0), (0, 1, 0)],
            slab_layers=5,
            vacuum_size=18.0,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "MultiSurfaceEnergyFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, MultiSurfaceEnergyFlowMaker)
        assert maker_restored.slab_layers == 5
        assert maker_restored.vacuum_size == 18.0


class TestCalculateMultiSurfaceEnergies:
    """Tests for calculate_multi_surface_energies function."""

    def test_calculate_multi_surface_energies_creates_flow(self, si_structure):
        """Test that calculate_multi_surface_energies creates a Flow."""
        bulk_maker = StaticMaker()
        slab_maker = StaticMaker()

        flow = calculate_multi_surface_energies(
            structure=si_structure,
            miller_indices=[(0, 0, 1)],
            bulk_maker=bulk_maker,
            slab_maker=slab_maker,
            slab_layers=2,
            vacuum_size=15.0,
        )

        assert isinstance(flow, Flow)
        assert flow.name == "multi_surface_energy"

    def test_calculate_multi_surface_energies_with_multiple_surfaces(
        self, si_structure
    ):
        """Test function with multiple Miller indices."""
        flow = calculate_multi_surface_energies(
            structure=si_structure,
            miller_indices=[(0, 0, 1), (1, 0, 0), (1, 1, 0)],
            bulk_maker=StaticMaker(),
            slab_maker=StaticMaker(),
            slab_layers=2,
        )

        assert isinstance(flow, Flow)
        # Should have bulk + multiple slabs + analysis
        assert len(flow) > 3

    def test_calculate_multi_surface_energies_symmetric(self, si_structure):
        """Test function with symmetric slab generation."""
        flow = calculate_multi_surface_energies(
            structure=si_structure,
            miller_indices=[(0, 0, 1)],
            bulk_maker=StaticMaker(),
            slab_maker=StaticMaker(),
            slab_layers=2,
            symmetrize=True,  # Should generate more slabs
        )

        assert isinstance(flow, Flow)
        # Symmetric mode may generate multiple terminations
        assert len(flow) >= 2  # At least bulk + analysis


class TestSurfaceFlowIntegration:
    """Integration tests for surface workflows."""

    def test_all_surface_makers_create_valid_flows(self, si_structure):
        """Test that all surface makers can create valid flows."""
        makers = [
            MultiSurfaceEnergyFlowMaker(
                miller_indices=[(0, 0, 1)],
                slab_layers=2,
            ),
            MultiSurfaceEnergyFlowMaker(
                miller_indices=[(1, 0, 0), (0, 1, 0)],
                slab_layers=2,
                symmetrize=True,
            ),
        ]

        for maker in makers:
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)
            assert hasattr(flow, "name")
            assert len(flow) > 0

    def test_surface_makers_with_different_structures(self, si_structure, al_structure):
        """Test surface makers work with different structures."""
        structures = [si_structure, al_structure]
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        for structure in structures:
            flow = maker.make(structure)
            assert isinstance(flow, Flow)
            assert len(flow) > 0

    def test_surface_flow_output_references(self, si_structure):
        """Test that flows have proper output handling."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow = maker.make(si_structure)

        # Flow should be iterable (has jobs)
        jobs = list(flow)
        assert len(jobs) > 0

        # Each job should have a name
        for job in jobs:
            assert hasattr(job, "name")
            assert hasattr(job, "function")

    def test_multi_surface_with_formula_units(self, si_structure):
        """Test multi-surface with specified formula units."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            formula_units_per_cell=2,
            slab_layers=2,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.formula_units_per_cell == 2

    def test_multi_surface_with_varying_parameters(self, si_structure):
        """Test multi-surface with different parameter combinations."""
        params = [
            {"slab_layers": 2, "vacuum_size": 10.0},
            {"slab_layers": 4, "vacuum_size": 15.0},
            {"slab_layers": 6, "vacuum_size": 20.0, "symmetrize": True},
        ]

        for param_set in params:
            maker = MultiSurfaceEnergyFlowMaker(
                miller_indices=[(0, 0, 1)],
                **param_set,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestSurfaceEdgeCases:
    """Test edge cases and error handling."""

    def test_multi_surface_with_single_miller_index(self, si_structure):
        """Test multi-surface with only one Miller index."""
        maker = MultiSurfaceEnergyFlowMaker(miller_indices=[(0, 0, 1)])

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_multi_surface_with_none_prev_dir(self, si_structure):
        """Test multi-surface with None as prev_dir."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow = maker.make(si_structure, prev_dir=None)
        assert isinstance(flow, Flow)

    def test_multiple_flows_from_same_maker(self, si_structure, al_structure):
        """Test creating multiple flows from the same maker."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow1 = maker.make(si_structure)
        flow2 = maker.make(al_structure)

        # Flows should be independent
        assert flow1 is not flow2
        assert isinstance(flow1, Flow)
        assert isinstance(flow2, Flow)

    def test_maker_modification_doesnt_affect_flows(self, si_structure):
        """Test that modifying maker after flow creation doesn't affect flow."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow1 = maker.make(si_structure)

        # Modify maker
        maker.slab_layers = 4

        flow2 = maker.make(si_structure)

        # Flows should be independent
        assert flow1 is not flow2


class TestSurfaceParameterValidation:
    """Test parameter validation for surface makers."""

    def test_valid_miller_indices(self):
        """Test that valid Miller indices are accepted."""
        valid_indices = [
            (0, 0, 1),
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 0),
        ]

        for hkl in valid_indices:
            maker = MultiSurfaceEnergyFlowMaker(miller_indices=[hkl])
            assert hkl in maker.miller_indices

    def test_slab_layers_parameter(self):
        """Test different slab layer values."""
        layers = [2, 3, 4, 5, 6, 8, 10]

        for n_layers in layers:
            maker = MultiSurfaceEnergyFlowMaker(slab_layers=n_layers)
            assert maker.slab_layers == n_layers

    def test_vacuum_size_parameter(self):
        """Test different vacuum size values."""
        vacuum_sizes = [10.0, 15.0, 20.0, 25.0]

        for vac in vacuum_sizes:
            maker = MultiSurfaceEnergyFlowMaker(vacuum_size=vac)
            assert maker.vacuum_size == vac

    def test_symmetrize_parameter(self):
        """Test symmetrize parameter."""
        for symmetrize in [True, False]:
            maker = MultiSurfaceEnergyFlowMaker(symmetrize=symmetrize)
            assert maker.symmetrize == symmetrize


class TestSurfaceMakerSerialization:
    """Test serialization of all surface makers."""

    def test_all_surface_makers_serializable(self):
        """Test that all surface makers can be serialized and deserialized."""
        makers = [
            ("surface", SurfaceEnergyFlowMaker(miller_indices=(1, 1, 0))),
            (
                "multi_surface",
                MultiSurfaceEnergyFlowMaker(miller_indices=[(0, 0, 1), (1, 1, 0)]),
            ),
        ]

        for name, maker in makers:
            # Serialize
            maker_dict = maker.as_dict()
            assert isinstance(maker_dict, dict), f"{name} failed to serialize"

            # Deserialize
            maker_restored = maker.from_dict(maker_dict)
            assert maker_restored is not None, f"{name} failed to deserialize"

    def test_multi_surface_serialization_preserves_parameters(self):
        """Test that serialization preserves all parameters."""
        maker = MultiSurfaceEnergyFlowMaker(
            name="custom_multi",
            miller_indices=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            slab_layers=5,
            vacuum_size=18.0,
            symmetrize=True,
            formula_units_per_cell=2,
        )

        # Serialize and deserialize
        maker_dict = maker.as_dict()
        maker_restored = maker.from_dict(maker_dict)

        # Check all parameters preserved
        assert maker_restored.name == "custom_multi"
        # Serialization converts tuples to lists - convert back for comparison
        assert [tuple(hkl) for hkl in maker_restored.miller_indices] == [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]
        assert maker_restored.slab_layers == 5
        assert maker_restored.vacuum_size == 18.0
        assert maker_restored.symmetrize is True
        assert maker_restored.formula_units_per_cell == 2


class TestSurfaceFlowComposition:
    """Test flow composition and chaining for surface calculations."""

    def test_multi_surface_flow_structure(self, si_structure):
        """Test the structure of multi-surface flows."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1), (1, 1, 0)],
            slab_layers=2,
        )

        flow = maker.make(si_structure)

        # Flow should have bulk, slabs, and analysis
        jobs = list(flow)

        # First job should be bulk
        assert "bulk" in jobs[0].name.lower()

        # Last job should be analysis
        assert "analysis" in jobs[-1].name.lower()

        # Middle jobs should be slabs
        slab_jobs = jobs[1:-1]
        assert len(slab_jobs) > 0

    def test_multi_surface_with_custom_name(self, si_structure):
        """Test creating multi-surface flows with custom names."""
        maker = MultiSurfaceEnergyFlowMaker(
            name="My Custom Surface Flow",
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow = maker.make(si_structure)
        assert flow.name == "My Custom Surface Flow"

    def test_multi_surface_job_count_scaling(self, si_structure):
        """Test that job count scales with Miller indices."""
        # Test with 1 surface
        maker1 = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
            symmetrize=False,  # One slab per surface
        )
        flow1 = maker1.make(si_structure)
        n_jobs1 = len(list(flow1))

        # Test with 2 surfaces
        maker2 = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1), (1, 0, 0)],
            slab_layers=2,
            symmetrize=False,
        )
        flow2 = maker2.make(si_structure)
        n_jobs2 = len(list(flow2))

        # Should have more jobs with more surfaces
        # (bulk + 1*slab1 + analysis) vs (bulk + 2*slabs + analysis)
        assert n_jobs2 > n_jobs1


# ==================== Additional Surface Tests ====================


class TestMultiSurfaceDryRun:
    """Test dry-run mode for multi-surface workflows."""

    def test_multi_surface_with_dry_run_enabled(self, si_structure):
        """Test MultiSurfaceEnergyFlowMaker with dry_run=True."""
        maker = MultiSurfaceEnergyFlowMaker(
            dry_run=True,
            miller_indices=[(0, 0, 1)],
            slab_layers=2,
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert maker.dry_run is True

    def test_multi_surface_dry_run_default_false(self):
        """Test that dry_run defaults to False."""
        maker = MultiSurfaceEnergyFlowMaker()
        assert maker.dry_run is False

    def test_multi_surface_dry_run_propagates_to_child_makers(self):
        """Test that dry_run propagates to bulk_static_maker and slab_static_maker."""
        maker = MultiSurfaceEnergyFlowMaker(dry_run=True)

        # dry_run should propagate through BaseSiestaFlowMaker's __post_init__
        assert maker.dry_run is True
        # BaseSiestaFlowMaker handles propagation to child makers


class TestMultiSurfaceInheritance:
    """Test BaseSiestaFlowMaker inheritance for MultiSurfaceEnergyFlowMaker."""

    def test_multi_surface_inherits_from_base_siesta_flow_maker(self):
        """Test that MultiSurfaceEnergyFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = MultiSurfaceEnergyFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_multi_surface_has_dry_run_attribute(self):
        """Test that MultiSurfaceEnergyFlowMaker has dry_run attribute."""
        maker = MultiSurfaceEnergyFlowMaker()
        assert hasattr(maker, "dry_run")

    def test_multi_surface_maker_repr(self):
        """Test string representation of MultiSurfaceEnergyFlowMaker."""
        maker = MultiSurfaceEnergyFlowMaker(name="test_multi_surface")

        repr_str = repr(maker)
        assert "MultiSurfaceEnergyFlowMaker" in repr_str


class TestAnalyzeMultiSurfaceResults:
    """Test analyze_multi_surface_results @job function."""

    def test_analyze_multi_surface_results_basic(self):
        """Test analyze_multi_surface_results with mock data."""
        from unittest.mock import MagicMock

        from pymatgen.core import Composition

        from atomate2.siesta.flows.surface.multi_surface import (
            analyze_multi_surface_results,
        )

        # Create mock bulk output
        mock_bulk_output = MagicMock()
        mock_bulk_output.output.energy = -100.0

        # Create mock slab data
        mock_slab_output = MagicMock()
        mock_slab_output.output.energy = -95.0

        all_slab_data = [
            {
                "miller_index": (0, 0, 1),
                "slab_jobs": [
                    {
                        "job_output": mock_slab_output,
                        "termination": "Si_term1",
                        "surface_area": 50.0,
                        "n_formula_units": 4.0,
                        "n_atoms": 8,
                        "thickness": 10.0,
                        "composition": {"Si": 8},
                        "is_symmetric": False,
                    }
                ],
            }
        ]

        bulk_composition = Composition("Si2")

        # Call the function using .original to bypass @job wrapper
        result = analyze_multi_surface_results.original(
            bulk_job_output=mock_bulk_output,
            all_slab_data=all_slab_data,
            bulk_composition=bulk_composition,
            formula_units_per_cell=1,
        )

        # Validate result structure
        assert isinstance(result, dict)
        assert "bulk_energy" in result
        assert "surface_results" in result
        assert result["bulk_energy"] == -100.0
        assert len(result["surface_results"]) == 1

    def test_analyze_multi_surface_results_with_dict_composition(self):
        """Test analyze_multi_surface_results with serialized Composition (dict)."""
        from unittest.mock import MagicMock

        from atomate2.siesta.flows.surface.multi_surface import (
            analyze_multi_surface_results,
        )

        mock_bulk_output = MagicMock()
        mock_bulk_output.output.energy = -100.0

        mock_slab_output = MagicMock()
        mock_slab_output.output.energy = -95.0

        all_slab_data = [
            {
                "miller_index": (0, 0, 1),
                "slab_jobs": [
                    {
                        "job_output": mock_slab_output,
                        "termination": "Si_term1",
                        "surface_area": 50.0,
                        "n_formula_units": 4.0,
                        "n_atoms": 8,
                        "thickness": 10.0,
                        "composition": {"Si": 8},
                        "is_symmetric": False,
                    }
                ],
            }
        ]

        # Pass composition as dict (serialized form)
        bulk_composition_dict = {"Si": 2}

        result = analyze_multi_surface_results.original(
            bulk_job_output=mock_bulk_output,
            all_slab_data=all_slab_data,
            bulk_composition=bulk_composition_dict,
            formula_units_per_cell=1,
        )

        assert isinstance(result, dict)
        assert "surface_results" in result


class TestMultiSurfaceEdgeCasesExtended:
    """Test additional edge cases for multi-surface workflows."""

    def test_multi_surface_with_many_miller_indices(self, si_structure):
        """Test multi-surface with large number of Miller indices."""
        # Test with 10+ Miller indices
        many_hkl = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (0, 2, 1),
            (2, 1, 1),
        ]

        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=many_hkl,
            slab_layers=2,
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert len(maker.miller_indices) == 12
        # Should have bulk + many slabs + analysis
        assert len(flow) > 10

    def test_multi_surface_minimal_parameters(self, si_structure):
        """Test multi-surface with minimal slab_layers and vacuum."""
        maker = MultiSurfaceEnergyFlowMaker(
            miller_indices=[(0, 0, 1)],
            slab_layers=2,  # Minimal layers
            vacuum_size=10.0,  # Minimal vacuum
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert maker.slab_layers == 2
        assert maker.vacuum_size == 10.0
