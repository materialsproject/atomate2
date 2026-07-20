"""
Unit tests for the module registry system.

Tests the registry-based auto-initialization system including:
- Module registration
- Tier filtering (hierarchical)
- Priority-based sorting
- Module metadata validation
"""

import pytest

from atomate2.siesta.dataclass.registry import (
    MODULE_REGISTRY,
    DataclassModule,
    get_modules_by_category,
    get_modules_for_tier,
    get_sorted_modules,
)


class TestDataclassModule:
    """Test DataclassModule dataclass."""

    def test_valid_module_creation(self):
        """Test creating a valid DataclassModule."""
        module = DataclassModule(
            name="test_module",
            module_path="atomate2.siesta.dataclass.test",
            class_name="TestClass",
            setup_method="setup_test",
            fdf_attribute="test_fdf_arguments",
            tier="intermediate",
            category="electronic",
            priority=50,
            description="Test module",
        )
        assert module.name == "test_module"
        assert module.tier == "intermediate"
        assert module.priority == 50

    def test_invalid_tier_raises_error(self):
        """Test that invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid module tier"):
            DataclassModule(
                name="test_module",
                module_path="atomate2.siesta.dataclass.test",
                class_name="TestClass",
                setup_method="setup_test",
                fdf_attribute="test_fdf_arguments",
                tier="invalid_tier",  # Invalid
            )

    def test_default_values(self):
        """Test default values for optional fields."""
        module = DataclassModule(
            name="test_module",
            module_path="atomate2.siesta.dataclass.test",
            class_name="TestClass",
            setup_method="setup_test",
            fdf_attribute="test_fdf_arguments",
        )
        assert module.tier == "intermediate"
        assert module.category == "general"
        assert module.priority == 50
        assert module.description == ""


class TestModuleRegistry:
    """Test module registry functions."""

    def test_registry_populated(self):
        """Test that MODULE_REGISTRY is populated with expected modules."""
        # Should have 24 registered modules
        assert len(MODULE_REGISTRY) >= 24

        # Check some expected modules
        expected_modules = [
            "pseudopotentials",
            "basis_sets",
            "xc_functional",
            "kpoints",
            "spin",
            "scf_loop",
        ]
        for module_name in expected_modules:
            assert module_name in MODULE_REGISTRY

    def test_module_has_required_attributes(self):
        """Test that registered modules have all required attributes."""
        for name, module in MODULE_REGISTRY.items():
            assert isinstance(module, DataclassModule)
            assert module.name == name
            assert module.module_path.startswith("atomate2.siesta.dataclass")
            assert module.class_name
            assert module.setup_method
            assert module.fdf_attribute
            assert module.tier in {"basic", "intermediate", "advanced", "expert"}
            assert isinstance(module.priority, int)

    def test_priority_ordering(self):
        """Test that priorities make sense (no duplicates in critical range)."""
        priorities = [module.priority for module in MODULE_REGISTRY.values()]
        # All priorities should be positive
        assert all(p > 0 for p in priorities)
        # Priorities should be reasonable (1-100 range)
        assert all(1 <= p <= 100 for p in priorities)


class TestTierFiltering:
    """Test tier-based module filtering."""

    def test_basic_tier(self):
        """Test that basic tier returns only basic modules."""
        modules = get_modules_for_tier("basic")

        # Basic tier should have 6 modules
        assert len(modules) == 6

        # All should be basic tier
        assert all(m.tier == "basic" for m in modules.values())

        # Check expected basic modules
        expected = {
            "pseudopotentials",
            "basis_sets",
            "xc_functional",
            "kpoints",
            "mesh_cutoff",
            "general_system",
        }
        assert set(modules.keys()) == expected

    def test_intermediate_tier(self):
        """Test that intermediate tier returns basic + intermediate modules."""
        modules = get_modules_for_tier("intermediate")

        # Intermediate tier should have 13 modules (6 basic + 7 intermediate)
        assert len(modules) == 13

        # Should include both basic and intermediate
        tiers = {m.tier for m in modules.values()}
        assert tiers == {"basic", "intermediate"}

        # Should include all basic modules
        basic_modules = get_modules_for_tier("basic")
        assert all(name in modules for name in basic_modules.keys())

    def test_advanced_tier(self):
        """Test that advanced tier returns basic + intermediate + advanced modules."""
        modules = get_modules_for_tier("advanced")

        # Advanced tier should have 22 modules (6 basic + 7 intermediate + 9 advanced)
        assert len(modules) == 22

        # Should include basic, intermediate, and advanced
        tiers = {m.tier for m in modules.values()}
        assert tiers == {"basic", "intermediate", "advanced"}

    def test_expert_tier(self):
        """Test that expert tier returns all modules."""
        modules = get_modules_for_tier("expert")

        # Expert tier should have all 24 modules
        assert len(modules) >= 24

        # Should include all tiers
        tiers = {m.tier for m in modules.values()}
        assert tiers == {"basic", "intermediate", "advanced", "expert"}

    def test_all_tier_alias(self):
        """Test that 'all' is an alias for 'expert'."""
        all_modules = get_modules_for_tier("all")
        expert_modules = get_modules_for_tier("expert")

        assert len(all_modules) == len(expert_modules)
        assert set(all_modules.keys()) == set(expert_modules.keys())

    def test_invalid_tier_raises_error(self):
        """Test that invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tier"):
            get_modules_for_tier("invalid_tier")

    def test_tier_hierarchy(self):
        """Test that tier hierarchy is cumulative."""
        basic = get_modules_for_tier("basic")
        intermediate = get_modules_for_tier("intermediate")
        advanced = get_modules_for_tier("advanced")
        expert = get_modules_for_tier("expert")

        # Each higher tier should include all lower tiers
        assert len(basic) < len(intermediate) < len(advanced) < len(expert)

        # Basic modules should be in all tiers
        for name in basic.keys():
            assert name in intermediate
            assert name in advanced
            assert name in expert


class TestCategoryFiltering:
    """Test category-based module filtering."""

    def test_get_modules_by_category(self):
        """Test filtering modules by category."""
        electronic_modules = get_modules_by_category("electronic")

        # Should have multiple electronic modules
        assert len(electronic_modules) > 0

        # All should be electronic category
        assert all(m.category == "electronic" for m in electronic_modules.values())

        # Should include expected modules
        expected_in_electronic = {"basis_sets", "xc_functional", "spin"}
        assert expected_in_electronic.issubset(set(electronic_modules.keys()))

    def test_numerical_category(self):
        """Test numerical category modules."""
        numerical_modules = get_modules_by_category("numerical")

        # Should include mesh_cutoff
        assert "mesh_cutoff" in numerical_modules
        assert all(m.category == "numerical" for m in numerical_modules.values())

    def test_convergence_category(self):
        """Test convergence category modules."""
        convergence_modules = get_modules_by_category("convergence")

        # Should include scf_loop
        assert "scf_loop" in convergence_modules
        assert all(m.category == "convergence" for m in convergence_modules.values())


class TestPrioritySorting:
    """Test priority-based module sorting."""

    def test_get_sorted_modules(self):
        """Test that modules are sorted by priority."""
        modules = get_modules_for_tier("intermediate")
        sorted_modules = get_sorted_modules(modules)

        # Should return a list
        assert isinstance(sorted_modules, list)

        # Should have same length as input
        assert len(sorted_modules) == len(modules)

        # Should be sorted by priority (ascending)
        priorities = [m.priority for m in sorted_modules]
        assert priorities == sorted(priorities)

    def test_priority_ordering_makes_sense(self):
        """Test that priority ordering follows dependency logic."""
        sorted_modules = get_sorted_modules(MODULE_REGISTRY)

        # Pseudopotentials should be early (low priority number)
        pseudo_index = next(
            i for i, m in enumerate(sorted_modules) if m.name == "pseudopotentials"
        )
        assert pseudo_index < 10, "Pseudopotentials should initialize early"

        # Basis sets should be early
        basis_index = next(
            i for i, m in enumerate(sorted_modules) if m.name == "basis_sets"
        )
        assert basis_index < 10, "Basis sets should initialize early"

        # XC functional should be early
        xc_index = next(
            i for i, m in enumerate(sorted_modules) if m.name == "xc_functional"
        )
        assert xc_index < 10, "XC functional should initialize early"

    def test_empty_dict_returns_empty_list(self):
        """Test sorting empty dict returns empty list."""
        sorted_modules = get_sorted_modules({})
        assert sorted_modules == []


class TestModuleMetadata:
    """Test specific module metadata is correct."""

    def test_pseudopotentials_module(self):
        """Test pseudopotentials module metadata."""
        module = MODULE_REGISTRY["pseudopotentials"]
        assert module.tier == "basic"
        assert module.category == "electronic"
        assert module.priority == 5
        assert module.class_name == "Pseudopotentials"
        assert module.setup_method == "setup_pseudos"
        assert module.fdf_attribute == "pseudo_path"

    def test_basis_sets_module(self):
        """Test basis sets module metadata."""
        module = MODULE_REGISTRY["basis_sets"]
        assert module.tier == "basic"
        assert module.category == "electronic"
        assert module.priority == 10
        assert module.class_name == "BasisSetsAndProjectors"
        assert module.setup_method == "setup_basis_sets_and_projectors"
        assert module.fdf_attribute == "basis_set_fdf_arguments"

    def test_scf_loop_module(self):
        """Test SCF loop module metadata."""
        module = MODULE_REGISTRY["scf_loop"]
        assert module.tier == "intermediate"
        assert module.category == "convergence"
        assert module.priority == 30
        assert module.class_name == "SCFLoopParameters"
        assert module.setup_method == "setup_scf_settings"
        assert module.fdf_attribute == "scf_fdf_arguments"

    def test_phonons_module(self):
        """Test phonons module metadata."""
        module = MODULE_REGISTRY["phonons"]
        assert module.tier == "advanced"
        assert module.category == "phonons"
        assert module.priority == 60
        assert module.class_name == "PhononCalculations"

    def test_parallel_module(self):
        """Test parallel module metadata."""
        module = MODULE_REGISTRY["parallel"]
        assert module.tier == "expert"
        assert module.category == "performance"
        assert module.priority == 80


class TestRegistryImmutability:
    """Test that registry operations don't modify the original."""

    def test_get_modules_returns_copy(self):
        """Test that tier filtering returns independent dict."""
        modules1 = get_modules_for_tier("intermediate")
        modules2 = get_modules_for_tier("intermediate")

        # Should be equal but not the same object
        assert modules1 == modules2
        assert modules1 is not modules2

        # Modifying one shouldn't affect the other
        modules1.pop("basis_sets")
        assert "basis_sets" in modules2

    def test_get_sorted_modules_returns_new_list(self):
        """Test that sorting returns new list."""
        modules = get_modules_for_tier("intermediate")
        sorted1 = get_sorted_modules(modules)
        sorted2 = get_sorted_modules(modules)

        assert sorted1 == sorted2
        assert sorted1 is not sorted2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
