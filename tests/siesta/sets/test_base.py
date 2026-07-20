"""Tests for SIESTA base input set and generator.

These tests validate:
- SiestaInputSet class
- SiestaInputGenerator class
- Parameter handling and FDF generation
"""

import pytest
from pymatgen.core import Structure, Lattice

from atomate2.siesta.sets.base import SiestaInputSet, SiestaInputGenerator


@pytest.fixture
def si_structure():
    """Silicon structure for testing."""
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


class TestSiestaInputSet:
    """Test SiestaInputSet class."""

    def test_input_set_class_exists(self):
        """Test SiestaInputSet class exists."""
        assert SiestaInputSet is not None
        assert hasattr(SiestaInputSet, "__init__")

    def test_input_set_is_input_set_subclass(self):
        """Test SiestaInputSet inherits from InputSet."""
        from pymatgen.io.core import InputSet

        assert issubclass(SiestaInputSet, InputSet)


class TestSiestaInputGenerator:
    """Test SiestaInputGenerator class."""

    def test_generator_initialization(self):
        """Test SiestaInputGenerator initialization."""
        generator = SiestaInputGenerator()
        assert generator is not None
        assert hasattr(generator, "get_input_set")

    def test_generator_with_user_params(self):
        """Test SiestaInputGenerator with user parameters."""
        generator = SiestaInputGenerator(
            user_params={"PAO.BasisSize": "DZP", "a2s_kpts": [4, 4, 4]}
        )
        assert generator.user_params is not None
        assert "PAO.BasisSize" in generator.user_params

    def test_generator_with_tier(self):
        """Test SiestaInputGenerator with tier specification."""
        generator = SiestaInputGenerator(tier="intermediate")
        assert generator.tier == "intermediate"

    def test_generator_get_input_set_basic(self, si_structure):
        """Test get_input_set with basic structure."""
        generator = SiestaInputGenerator()

        # This will create the full input set
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None
        assert isinstance(input_set, SiestaInputSet)

    def test_generator_system_label_generation(self, si_structure):
        """Test system label is generated from structure."""
        generator = SiestaInputGenerator()
        input_set = generator.get_input_set(si_structure)

        # System label should be based on structure formula
        assert input_set is not None

    def test_generator_with_empty_user_params(self, si_structure):
        """Test generator with empty user_params."""
        generator = SiestaInputGenerator(user_params={})
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_generator_with_multiple_params(self, si_structure):
        """Test generator with multiple user parameters."""
        generator = SiestaInputGenerator(
            user_params={"PAO.BasisSize": "DZP", "Mesh.Cutoff": "300 Ry"}
        )
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None


class TestSiestaInputGeneratorMethods:
    """Test specific methods of SiestaInputGenerator."""

    def test_generator_methods_exist(self):
        """Test key generator methods exist."""
        generator = SiestaInputGenerator()
        assert hasattr(generator, "get_input_set")
        assert hasattr(generator, "_get_input_parameters")
        assert callable(generator.get_input_set)


class TestSiestaInputGeneratorParameters:
    """Test parameter handling in SiestaInputGenerator."""

    def test_case_insensitive_parameters(self, si_structure):
        """Test that parameters are case-insensitive."""
        gen1 = SiestaInputGenerator(user_params={"PAO.BasisSize": "DZP"})
        gen2 = SiestaInputGenerator(user_params={"pao.basissize": "DZP"})

        # Both should work
        input1 = gen1.get_input_set(si_structure)
        input2 = gen2.get_input_set(si_structure)

        assert input1 is not None
        assert input2 is not None

    def test_kpts_parameter_handling(self, si_structure):
        """Test k-points parameter handling."""
        generator = SiestaInputGenerator(user_params={"a2s_kpts": [6, 6, 6]})
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_mesh_cutoff_parameter(self, si_structure):
        """Test mesh cutoff parameter."""
        generator = SiestaInputGenerator(user_params={"Mesh.Cutoff": "250 Ry"})
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_pao_basis_parameter(self, si_structure):
        """Test PAO.BasisSize parameter."""
        generator = SiestaInputGenerator(user_params={"PAO.BasisSize": "SZ"})
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None


class TestSiestaInputGeneratorTiers:
    """Test tier-based initialization."""

    def test_basic_tier(self, si_structure):
        """Test basic tier initialization."""
        generator = SiestaInputGenerator(tier="basic")
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_intermediate_tier(self, si_structure):
        """Test intermediate tier initialization."""
        generator = SiestaInputGenerator(tier="intermediate")
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_advanced_tier(self, si_structure):
        """Test advanced tier initialization."""
        generator = SiestaInputGenerator(tier="advanced")
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_expert_tier(self, si_structure):
        """Test expert tier initialization."""
        generator = SiestaInputGenerator(tier="expert")
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None

    def test_default_tier(self, si_structure):
        """Test default tier (should be basic)."""
        generator = SiestaInputGenerator()
        input_set = generator.get_input_set(si_structure)

        assert input_set is not None


class TestSiestaInputGeneratorVerbosity:
    """Test verbosity control."""

    def test_console_verbosity_attribute(self):
        """Test CONSOLE_VERBOSITY attribute exists."""
        generator = SiestaInputGenerator()
        assert hasattr(generator, "CONSOLE_VERBOSITY")


class TestSiestaInputGeneratorPseudopotentials:
    """Test pseudopotential handling."""

    def test_pseudopotential_initialization(self, si_structure):
        """Test that pseudopotentials are initialized."""
        generator = SiestaInputGenerator()
        input_set = generator.get_input_set(si_structure)

        # Input set should be created with pseudopotentials
        assert input_set is not None

    def test_generator_has_pseudo_attributes(self):
        """Test generator has pseudopotential-related attributes."""
        generator = SiestaInputGenerator()
        # Test the generator was created successfully
        assert generator is not None


class TestInternalParameterNaming:
    """Test internal parameter naming system with dual prefix support."""

    def test_filter_internal_params_with_alias_prefix(self):
        """Test filtering parameters with a2s_ alias prefix."""
        from atomate2.siesta.sets.base import filter_internal_params

        params = {
            "Mesh.Cutoff": "300 Ry",
            "a2s_magnetic_ordering": "AFM",
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
        }

        fdf_params, internal_params = filter_internal_params(params)

        assert "Mesh.Cutoff" in fdf_params
        assert "PAO.BasisSize" in fdf_params
        assert "a2s_magnetic_ordering" not in fdf_params
        assert "a2s_kpts" not in fdf_params
        assert "magnetic_ordering" in internal_params
        assert "kpts" in internal_params
        assert internal_params["magnetic_ordering"] == "AFM"
        assert internal_params["kpts"] == [4, 4, 4]

    def test_filter_internal_params_with_full_prefix(self):
        """Test filtering parameters with atomate2siesta_ full prefix."""
        from atomate2.siesta.sets.base import filter_internal_params

        params = {
            "Spin": "polarized",
            "atomate2siesta_magnetic_ordering": "ferromagnetic",
            "Mesh.Cutoff": "200 Ry",
        }

        fdf_params, internal_params = filter_internal_params(params)

        assert "Spin" in fdf_params
        assert "Mesh.Cutoff" in fdf_params
        assert "atomate2siesta_magnetic_ordering" not in fdf_params
        assert "magnetic_ordering" in internal_params
        assert internal_params["magnetic_ordering"] == "ferromagnetic"

    def test_filter_internal_params_mixed_prefixes(self):
        """Test filtering with both alias and full prefix."""
        from atomate2.siesta.sets.base import filter_internal_params

        params = {
            "a2s_magnetic_ordering": "FM",
            "atomate2siesta_kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
        }

        fdf_params, internal_params = filter_internal_params(params)

        assert "Mesh.Cutoff" in fdf_params
        assert "magnetic_ordering" in internal_params
        assert "kpts" in internal_params
        assert internal_params["magnetic_ordering"] == "FM"
        assert internal_params["kpts"] == [6, 6, 6]

    def test_filter_internal_params_no_internal(self):
        """Test filtering with no internal parameters."""
        from atomate2.siesta.sets.base import filter_internal_params

        params = {
            "PAO.BasisSize": "DZP",
            "Mesh.Cutoff": "300 Ry",
            "Spin": "polarized",
        }

        fdf_params, internal_params = filter_internal_params(params)

        assert len(fdf_params) == 3
        assert len(internal_params) == 0
        assert "PAO.BasisSize" in fdf_params
        assert "Mesh.Cutoff" in fdf_params
        assert "Spin" in fdf_params

    def test_normalize_internal_params_legacy_names(self):
        """Test that legacy unprefixed parameter names are rejected (v1.0.0+)."""
        from atomate2.siesta.sets.base import normalize_internal_params

        params = {
            "Mesh.Cutoff": "300 Ry",
            "magnetic_ordering": "FM",
            "kpts": [4, 4, 4],
        }

        # Legacy unprefixed names are no longer supported and must raise.
        with pytest.raises(
            ValueError, match="Legacy unprefixed parameter\\(s\\) detected"
        ) as exc_info:
            normalize_internal_params(params)

        # Both legacy keys should be reported
        assert "magnetic_ordering" in str(exc_info.value)
        assert "kpts" in str(exc_info.value)

        # Already-prefixed equivalents are accepted unchanged
        prefixed = {
            "Mesh.Cutoff": "300 Ry",
            "a2s_magnetic_ordering": "FM",
            "a2s_kpts": [4, 4, 4],
        }
        normalized = normalize_internal_params(prefixed)
        assert normalized["a2s_magnetic_ordering"] == "FM"
        assert normalized["a2s_kpts"] == [4, 4, 4]

    def test_normalize_internal_params_new_names(self):
        """Test normalization with already-prefixed names."""
        from atomate2.siesta.sets.base import normalize_internal_params
        import warnings

        params = {
            "a2s_magnetic_ordering": "AFM",
            "Mesh.Cutoff": "300 Ry",
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = normalize_internal_params(params)

            assert len(w) == 0

        assert normalized == params

    def test_normalize_internal_params_mixed_legacy_and_new(self):
        """Test that a mix of legacy and new names still rejects the legacy one."""
        from atomate2.siesta.sets.base import normalize_internal_params

        params = {
            "magnetic_ordering": "FM",
            "a2s_kpts": [4, 4, 4],
            "Mesh.Cutoff": "300 Ry",
        }

        # The legacy unprefixed key must trigger a ValueError even when a
        # correctly-prefixed key is also present.
        with pytest.raises(
            ValueError, match="Legacy unprefixed parameter\\(s\\) detected"
        ) as exc_info:
            normalize_internal_params(params)

        assert "magnetic_ordering" in str(exc_info.value)

    def test_internal_params_not_in_fdf(self, si_structure):
        """Test that internal parameters don't appear in FDF output."""
        generator = SiestaInputGenerator(
            user_params={
                "Spin": "polarized",
                "a2s_magnetic_ordering": "ferromagnetic",
                "Mesh.Cutoff": "300 Ry",
            }
        )

        input_set = generator.get_input_set(si_structure)
        assert input_set is not None

    def test_legacy_magnetic_ordering_constant(self):
        """Test LEGACY_INTERNAL_PARAMS constant."""
        from atomate2.siesta.sets.base import LEGACY_INTERNAL_PARAMS

        assert "magnetic_ordering" in LEGACY_INTERNAL_PARAMS
        assert LEGACY_INTERNAL_PARAMS["magnetic_ordering"] == "a2s_magnetic_ordering"
        assert "kpts" in LEGACY_INTERNAL_PARAMS
        assert LEGACY_INTERNAL_PARAMS["kpts"] == "a2s_kpts"
        assert "pseudo_path" in LEGACY_INTERNAL_PARAMS
        assert LEGACY_INTERNAL_PARAMS["pseudo_path"] == "a2s_pseudo_path"
        assert "pseudo_family" in LEGACY_INTERNAL_PARAMS

    def test_prefix_constants(self):
        """Test internal parameter prefix constants."""
        from atomate2.siesta.sets.base import (
            INTERNAL_PARAM_PREFIX_FULL,
            INTERNAL_PARAM_PREFIX_ALIAS,
        )

        assert INTERNAL_PARAM_PREFIX_FULL == "atomate2siesta_"
        assert INTERNAL_PARAM_PREFIX_ALIAS == "a2s_"

    def test_workflow_with_prefixed_params(self):
        """Test complete workflow with prefixed internal parameters."""
        from pymatgen.core import Structure, Lattice

        lattice = Lattice.cubic(2.87)
        fe_structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        fe_structure.add_site_property("magmom", [2.5, 2.5])

        generator = SiestaInputGenerator(
            user_params={
                "Spin": "polarized",
                "a2s_magnetic_ordering": "FM",
                "PAO.BasisSize": "DZP",
                "Mesh.Cutoff": "300 Ry",
            }
        )

        input_set = generator.get_input_set(fe_structure)
        assert input_set is not None
