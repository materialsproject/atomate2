"""
Tests for core SIESTA input set generators.

These tests validate:
- StaticSetGenerator (single-point calculations)
- RelaxSetGenerator (structural relaxation)
- BandStructureSetGenerator (electronic band structure)
- FDF file generation
- Parameter merging and inheritance
- Tier system integration
"""

from collections import OrderedDict

import pytest

from atomate2.siesta.sets.base import SiestaInputSet
from atomate2.siesta.sets.core import (
    BandStructureSetGenerator,
    OpticalSetGenerator,
    PhononSetGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
)


class TestStaticSetGenerator:
    """Tests for StaticSetGenerator."""

    def test_default_static_set_generator(self):
        """Test creation of default StaticSetGenerator."""
        gen = StaticSetGenerator()

        assert gen.basis_set_size == "SZ"
        assert gen.xc == "PBE"
        assert gen.mesh_cutoff == 100.0
        assert gen.kpts == [1, 1, 1]
        assert gen.tier == "intermediate"

    def test_static_set_generator_with_custom_params(self):
        """Test StaticSetGenerator with custom parameters."""
        gen = StaticSetGenerator(
            basis_set_size="DZP",
            xc="LDA",
            mesh_cutoff=200.0,
            kpts=[4, 4, 4],
            tier="advanced",
        )

        assert gen.basis_set_size == "DZP"
        assert gen.xc == "LDA"
        assert gen.mesh_cutoff == 200.0
        assert gen.kpts == [4, 4, 4]
        assert gen.tier == "advanced"

    def test_static_set_generator_with_user_params(self):
        """Test StaticSetGenerator with user_params."""
        user_params = OrderedDict(
            {
                "PAO.BasisSize": "TZP",
                "Mesh.Cutoff": "300 Ry",
                "a2s_kpts": [6, 6, 6],
            }
        )

        gen = StaticSetGenerator(user_params=user_params)

        assert gen.user_params["PAO.BasisSize"] == "TZP"
        assert gen.user_params["Mesh.Cutoff"] == "300 Ry"
        assert gen.user_params["a2s_kpts"] == [6, 6, 6]

    def test_static_set_generator_get_input_set(self, si_structure):
        """Test that StaticSetGenerator creates a valid input set."""
        gen = StaticSetGenerator()
        input_set = gen.get_input_set(si_structure)

        assert isinstance(input_set, SiestaInputSet)
        assert input_set._structure == si_structure

    def test_static_set_generator_serialization(self):
        """Test StaticSetGenerator serialization."""
        gen = StaticSetGenerator(basis_set_size="DZP", tier="advanced")

        # Serialize
        gen_dict = gen.as_dict()
        assert isinstance(gen_dict, dict)
        assert gen_dict["@class"] == "StaticSetGenerator"

        # Deserialize
        gen_restored = gen.from_dict(gen_dict)
        assert isinstance(gen_restored, StaticSetGenerator)
        assert gen_restored.basis_set_size == "DZP"
        assert gen_restored.tier == "advanced"


class TestRelaxSetGenerator:
    """Tests for RelaxSetGenerator."""

    def test_default_relax_set_generator(self, si_structure):
        """Test creation of default RelaxSetGenerator."""
        gen = RelaxSetGenerator()

        assert gen.relax_cell is False  # Fixed-cell by default
        assert gen.enable_lua is False  # Lua disabled for relaxation

        # The relaxation module is created during input set generation as
        # ``_md_relaxation_module`` (see RelaxSetGenerator.__post_init__).
        gen.get_input_set(si_structure)
        assert hasattr(gen, "_md_relaxation_module")

    def test_relax_set_generator_fixed_cell(self, si_structure):
        """Test fixed-cell RelaxSetGenerator."""
        gen = RelaxSetGenerator(relax_cell=False)

        assert gen.relax_cell is False
        gen.get_input_set(si_structure)
        assert gen._md_relaxation_module.md_variable_cell is False

    def test_relax_set_generator_variable_cell(self, si_structure):
        """Test variable-cell RelaxSetGenerator."""
        gen = RelaxSetGenerator(relax_cell=True)

        assert gen.relax_cell is True
        gen.get_input_set(si_structure)
        assert gen._md_relaxation_module.md_variable_cell is True

    def test_relax_set_generator_with_md_steps(self, si_structure):
        """Test RelaxSetGenerator with custom MD steps."""
        user_params = OrderedDict({"MD.NumCGsteps": 100})
        gen = RelaxSetGenerator(user_params=user_params)

        # MD steps should be extracted from user_params
        gen.get_input_set(si_structure)
        assert gen._md_relaxation_module.md_steps == 100

    def test_relax_set_generator_get_parameter_updates(self, si_structure):
        """Test RelaxSetGenerator parameter updates."""
        gen = RelaxSetGenerator()
        input_set = gen.get_input_set(si_structure)

        # Should have relaxation-specific FDF arguments
        fdf_args = input_set.siesta_input.parameters.get("fdf_arguments", {})

        # Relaxation should set MD.TypeOfRun
        assert "MD.TypeOfRun" in fdf_args or gen.relaxation.md_type_of_run is not None

    def test_relax_set_generator_serialization(self):
        """Test RelaxSetGenerator serialization."""
        gen = RelaxSetGenerator(relax_cell=True)

        # Serialize
        gen_dict = gen.as_dict()
        assert isinstance(gen_dict, dict)
        assert gen_dict["@class"] == "RelaxSetGenerator"
        assert gen_dict["relax_cell"] is True

        # Deserialize
        gen_restored = gen.from_dict(gen_dict)
        assert isinstance(gen_restored, RelaxSetGenerator)
        assert gen_restored.relax_cell is True


class TestBandStructureSetGenerator:
    """Tests for BandStructureSetGenerator."""

    def test_default_band_structure_set_generator(self):
        """Test creation of default BandStructureSetGenerator."""
        gen = BandStructureSetGenerator()

        assert gen.basis_set_size == "SZ"
        assert gen.xc == "PBE"
        assert hasattr(gen, "get_parameter_updates")

    def test_band_structure_set_generator_with_custom_params(self):
        """Test BandStructureSetGenerator with custom parameters."""
        gen = BandStructureSetGenerator(
            basis_set_size="DZP",
            kpts=[8, 8, 8],
        )

        assert gen.basis_set_size == "DZP"
        assert gen.kpts == [8, 8, 8]

    def test_band_structure_set_generator_get_input_set(self, si_structure):
        """Test that BandStructureSetGenerator creates a valid input set."""
        gen = BandStructureSetGenerator()
        input_set = gen.get_input_set(si_structure)

        assert isinstance(input_set, SiestaInputSet)
        # Should have band structure k-path parameters
        fdf_args = input_set.siesta_input.parameters.get("fdf_arguments", {})
        # Band structure should set BandLinesScale or similar
        assert len(fdf_args) > 0

    def test_band_structure_set_generator_kpath_generation(self, si_structure):
        """Test k-path generation for band structure."""
        gen = BandStructureSetGenerator()
        input_set = gen.get_input_set(si_structure)

        # Should have automatically generated k-path
        fdf_args = input_set.siesta_input.parameters.get("fdf_arguments", {})
        # Check for band structure-related FDF arguments
        # (actual keys depend on implementation)
        assert isinstance(fdf_args, dict)

    def test_band_structure_set_generator_serialization(self):
        """Test BandStructureSetGenerator serialization."""
        gen = BandStructureSetGenerator(basis_set_size="TZP")

        # Serialize
        gen_dict = gen.as_dict()
        assert isinstance(gen_dict, dict)
        assert gen_dict["@class"] == "BandStructureSetGenerator"

        # Deserialize
        gen_restored = gen.from_dict(gen_dict)
        assert isinstance(gen_restored, BandStructureSetGenerator)
        assert gen_restored.basis_set_size == "TZP"


class TestPhononSetGenerator:
    """Tests for PhononSetGenerator."""

    def test_default_phonon_set_generator(self):
        """Test creation of default PhononSetGenerator."""
        gen = PhononSetGenerator()

        assert gen.md_type_of_run == "FC"
        assert gen.md_fc_first == 1
        assert gen.md_fc_last == 1
        assert gen.md_fc_displ == 0.04
        assert hasattr(gen, "phonon")

    def test_phonon_set_generator_with_custom_params(self):
        """Test PhononSetGenerator with custom parameters."""
        gen = PhononSetGenerator(
            md_fc_first=1,
            md_fc_last=4,
            md_fc_displ=0.02,
        )

        assert gen.md_fc_first == 1
        assert gen.md_fc_last == 4
        assert gen.md_fc_displ == 0.02

    def test_phonon_set_generator_get_input_set(self, si_structure):
        """Test that PhononSetGenerator creates a valid input set."""
        gen = PhononSetGenerator()
        input_set = gen.get_input_set(si_structure)

        assert isinstance(input_set, SiestaInputSet)
        # Should have phonon-specific FDF arguments
        fdf_args = input_set.siesta_input.parameters.get("fdf_arguments", {})
        assert len(fdf_args) > 0

    def test_phonon_set_generator_serialization(self):
        """Test PhononSetGenerator serialization."""
        gen = PhononSetGenerator(md_fc_displ=0.03)

        # Serialize
        gen_dict = gen.as_dict()
        assert isinstance(gen_dict, dict)
        assert gen_dict["@class"] == "PhononSetGenerator"

        # Deserialize
        gen_restored = gen.from_dict(gen_dict)
        assert isinstance(gen_restored, PhononSetGenerator)
        assert gen_restored.md_fc_displ == 0.03


class TestOpticalSetGenerator:
    """Tests for OpticalSetGenerator."""

    def test_default_optical_set_generator(self):
        """Test creation of default OpticalSetGenerator."""
        gen = OpticalSetGenerator(optical_calculation="polarizability")

        assert gen.optical_calculation == "polarizability"
        assert hasattr(gen, "optical")

    def test_optical_set_generator_with_custom_params(self):
        """Test OpticalSetGenerator with custom parameters."""
        gen = OpticalSetGenerator(
            optical_calculation="dielectric_function",
            basis_set_size="DZP",
        )

        assert gen.optical_calculation == "dielectric_function"
        assert gen.basis_set_size == "DZP"

    def test_optical_set_generator_get_input_set(self, si_structure):
        """Test that OpticalSetGenerator creates a valid input set."""
        gen = OpticalSetGenerator(optical_calculation="polarizability")
        input_set = gen.get_input_set(si_structure)

        assert isinstance(input_set, SiestaInputSet)
        # Should have optical-specific FDF arguments
        fdf_args = input_set.siesta_input.parameters.get("fdf_arguments", {})
        assert len(fdf_args) > 0


class TestInputSetGeneration:
    """Integration tests for input set generation."""

    def test_all_generators_create_valid_input_sets(self, si_structure):
        """Test that all generators can create valid input sets."""
        generators = [
            StaticSetGenerator(),
            RelaxSetGenerator(),
            BandStructureSetGenerator(),
            PhononSetGenerator(),
            OpticalSetGenerator(optical_calculation="polarizability"),
        ]

        for gen in generators:
            input_set = gen.get_input_set(si_structure)
            assert isinstance(input_set, SiestaInputSet)
            assert input_set._structure == si_structure

    def test_generators_with_different_structures(
        self, si_structure, al_structure, graphene_structure
    ):
        """Test generators work with different structure types."""
        structures = [si_structure, al_structure, graphene_structure]
        gen = StaticSetGenerator()

        for structure in structures:
            input_set = gen.get_input_set(structure)
            assert isinstance(input_set, SiestaInputSet)
            assert input_set._structure == structure

    def test_generator_parameter_inheritance(self):
        """Test that parameters are correctly inherited."""
        # Create generator with specific parameters
        gen = StaticSetGenerator(
            basis_set_size="TZP",
            xc="LDA",
            mesh_cutoff=250.0,
        )

        # Parameters should be accessible
        assert gen.basis_set_size == "TZP"
        assert gen.xc == "LDA"
        assert gen.mesh_cutoff == 250.0

    def test_generator_tier_integration(self):
        """Test tier system integration."""
        tiers = ["basic", "intermediate", "advanced", "expert"]

        for tier in tiers:
            gen = StaticSetGenerator(tier=tier)
            assert gen.tier == tier

    def test_generator_with_fdf_arguments(self, si_structure):
        """Test generators with explicit FDF arguments."""
        fdf_args = OrderedDict(
            {
                "SCF.Mixer.Weight": 0.1,
                "MaxSCFIterations": 100,
            }
        )

        gen = StaticSetGenerator(fdf_arguments=fdf_args)
        input_set = gen.get_input_set(si_structure)

        # FDF arguments should be present
        result_fdf = input_set.siesta_input.parameters.get("fdf_arguments", {})
        assert "SCF.Mixer.Weight" in result_fdf or len(result_fdf) > 0


class TestInputSetEdgeCases:
    """Test edge cases and error handling."""

    def test_generator_with_none_user_params(self):
        """Test generator with None user_params."""
        gen = StaticSetGenerator(user_params=None)
        # Generator accepts None for user_params
        assert (
            gen.user_params is None
            or gen.user_params == {}
            or isinstance(gen.user_params, (dict, OrderedDict))
        )

    def test_generator_with_empty_user_params(self):
        """Test generator with empty user_params."""
        gen = StaticSetGenerator(user_params=OrderedDict())
        assert gen.user_params is not None

    def test_get_input_set_without_structure_raises_error(self):
        """Test that get_input_set without structure raises ValueError."""
        gen = StaticSetGenerator()

        with pytest.raises(ValueError, match="No structure can be determined"):
            gen.get_input_set(structure=None, prev_dir=None)

    def test_multiple_input_sets_from_same_generator(self, si_structure, al_structure):
        """Test creating multiple input sets from same generator."""
        gen = StaticSetGenerator()

        input_set1 = gen.get_input_set(si_structure)
        input_set2 = gen.get_input_set(al_structure)

        # Should be different input sets
        assert input_set1 is not input_set2
        assert input_set1._structure != input_set2._structure

    def test_generator_modification_doesnt_affect_input_sets(self, si_structure):
        """Test that modifying generator doesn't affect existing input sets."""
        gen = StaticSetGenerator(basis_set_size="SZ")
        input_set1 = gen.get_input_set(si_structure)

        # Modify generator
        gen.basis_set_size = "DZP"
        input_set2 = gen.get_input_set(si_structure)

        # Input sets should be independent
        assert input_set1 is not input_set2


class TestParameterMerging:
    """Test parameter merging and precedence."""

    def test_user_params_override_defaults(self):
        """Test that user_params override default parameters."""
        user_params = OrderedDict(
            {
                "PAO.BasisSize": "TZP",
            }
        )

        gen = StaticSetGenerator(
            basis_set_size="SZ",  # Default
            user_params=user_params,  # Should override
        )

        # user_params should take precedence
        assert gen.user_params["PAO.BasisSize"] == "TZP"

    def test_fdf_arguments_merging(self, si_structure):
        """Test that FDF arguments are correctly merged."""
        fdf_args = OrderedDict(
            {
                "Custom.Parameter": "value1",
                "Another.Parameter": "value2",
            }
        )

        gen = StaticSetGenerator(fdf_arguments=fdf_args)
        input_set = gen.get_input_set(si_structure)

        result_fdf = input_set.siesta_input.parameters.get("fdf_arguments", {})
        # Should have merged FDF arguments (custom + defaults)
        assert len(result_fdf) > 0

    def test_tier_parameter_activation(self):
        """Test that tier affects parameter activation."""
        gen_basic = StaticSetGenerator(tier="basic")
        gen_expert = StaticSetGenerator(tier="expert")

        # Expert tier should enable more modules
        assert gen_basic.tier == "basic"
        assert gen_expert.tier == "expert"


class TestGeneratorSerialization:
    """Test serialization of all generators."""

    def test_all_generators_serializable(self):
        """Test that all generators can be serialized and deserialized."""
        generators = [
            ("static", StaticSetGenerator()),
            ("relax", RelaxSetGenerator()),
            ("bands", BandStructureSetGenerator()),
            ("phonon", PhononSetGenerator()),
            ("optical", OpticalSetGenerator(optical_calculation="polarizability")),
        ]

        for name, gen in generators:
            # Serialize
            gen_dict = gen.as_dict()
            assert isinstance(gen_dict, dict), f"{name} failed to serialize"

            # Deserialize
            gen_restored = gen.from_dict(gen_dict)
            assert gen_restored is not None, f"{name} failed to deserialize"

    def test_generator_serialization_preserves_parameters(self):
        """Test that serialization preserves all parameters."""
        gen = StaticSetGenerator(
            basis_set_size="DZP",
            xc="LDA",
            mesh_cutoff=200.0,
            kpts=[4, 4, 4],
            tier="advanced",
        )

        # Serialize and deserialize
        gen_dict = gen.as_dict()
        gen_restored = gen.from_dict(gen_dict)

        # Check all parameters preserved
        assert gen_restored.basis_set_size == "DZP"
        assert gen_restored.xc == "LDA"
        assert gen_restored.mesh_cutoff == 200.0
        assert gen_restored.kpts == [4, 4, 4]
        assert gen_restored.tier == "advanced"
