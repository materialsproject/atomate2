"""
Unit tests for tier presets system.

Tests the material-specific tier preset system including:
- Preset definitions and structure
- Preset retrieval and validation
- Preset application to Makers
- Parameter merging logic
- Category organization
"""

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import (
    TIER_CATEGORIES,
    TIER_PRESETS,
    apply_tier_preset,
    get_presets_by_category,
    get_tier_preset,
    list_tier_presets,
)


class TestPresetDefinitions:
    """Test that preset definitions are valid."""

    def test_all_presets_have_required_fields(self):
        """Test that all presets have required fields."""
        required_fields = {
            "description",
            "tier",
            "enabled_modules",
            "disabled_modules",
            "recommended_params",
        }

        for name, preset in TIER_PRESETS.items():
            assert isinstance(preset, dict), f"{name} is not a dict"
            for field in required_fields:
                assert field in preset, f"{name} missing {field}"

    def test_preset_tiers_are_valid(self):
        """Test that all preset tiers are valid."""
        valid_tiers = {"basic", "intermediate", "advanced", "expert", "dirty", "ultra"}

        for name, preset in TIER_PRESETS.items():
            assert preset["tier"] in valid_tiers, (
                f"{name} has invalid tier: {preset['tier']}"
            )

    def test_enabled_modules_are_lists(self):
        """Test that enabled_modules are lists."""
        for name, preset in TIER_PRESETS.items():
            assert isinstance(preset["enabled_modules"], list), (
                f"{name} enabled_modules not a list"
            )

    def test_disabled_modules_are_lists(self):
        """Test that disabled_modules are lists."""
        for name, preset in TIER_PRESETS.items():
            assert isinstance(preset["disabled_modules"], list), (
                f"{name} disabled_modules not a list"
            )

    def test_recommended_params_are_dicts(self):
        """Test that recommended_params are dicts."""
        for name, preset in TIER_PRESETS.items():
            assert isinstance(preset["recommended_params"], dict), (
                f"{name} recommended_params not a dict"
            )

    def test_descriptions_are_non_empty(self):
        """Test that all presets have non-empty descriptions."""
        for name, preset in TIER_PRESETS.items():
            assert preset["description"], f"{name} has empty description"
            assert isinstance(preset["description"], str)


class TestPresetCounts:
    """Test preset counts and organization."""

    def test_expected_number_of_presets(self):
        """Test that we have the expected number of presets."""
        # Should have 39 presets
        assert len(TIER_PRESETS) == 39

    def test_category_organization(self):
        """Every preset is categorized exactly once, with no stale names."""
        from atomate2.siesta.sets.tiers import TIER_PRESETS

        categorized = [p for presets in TIER_CATEGORIES.values() for p in presets]

        # No preset appears in more than one category
        assert len(categorized) == len(set(categorized))
        # Categories cover exactly the real presets (no stale/missing names)
        assert set(categorized) == set(TIER_PRESETS)

    def test_expected_categories(self):
        """Test that we have the expected categories."""
        expected_categories = {
            "Structural",
            "Surface",
            "2D Materials",
            "Magnetic",
            "Phonon",
            "Optical",
            "Defects",
            "Electrocatalysis",
            "Performance",
            "Testing",
        }
        assert set(TIER_CATEGORIES.keys()) == expected_categories


class TestGetTierPreset:
    """Test get_tier_preset function."""

    def test_get_valid_preset(self):
        """Test getting a valid preset."""
        preset = get_tier_preset("surface_metal")

        assert preset["tier"] == "intermediate"
        assert "description" in preset
        assert "OccupationFunction" in preset["recommended_params"]

    def test_get_invalid_preset_raises_error(self):
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tier preset"):
            get_tier_preset("nonexistent_preset")

    def test_preset_is_copy(self):
        """Test that returned preset is a copy."""
        preset1 = get_tier_preset("surface_metal")
        preset2 = get_tier_preset("surface_metal")

        # Should be equal but not same object
        assert preset1 == preset2
        assert preset1 is not preset2

        # Modifying one shouldn't affect the other
        preset1["tier"] = "expert"
        assert preset2["tier"] == "intermediate"


class TestListTierPresets:
    """Test list_tier_presets function."""

    def test_list_presets_returns_dict(self):
        """Test that list_tier_presets returns a dict."""
        presets = list_tier_presets()
        assert isinstance(presets, dict)

    def test_list_presets_has_all_presets(self):
        """Test that list includes all presets."""
        presets = list_tier_presets()
        assert len(presets) == len(TIER_PRESETS)

    def test_list_presets_maps_to_descriptions(self):
        """Test that list maps names to descriptions."""
        presets = list_tier_presets()

        for name, description in presets.items():
            assert isinstance(description, str)
            assert description == TIER_PRESETS[name]["description"]


class TestGetPresetsByCategory:
    """Test get_presets_by_category function."""

    def test_get_structural_presets(self):
        """Test getting structural presets."""
        presets = get_presets_by_category("Structural")

        assert isinstance(presets, list)
        assert len(presets) == 7
        assert set(presets) == {
            "relax_dirty",
            "relax_standard",
            "relax_high_accuracy",
            "relax_bulk_metal",
            "relax_bulk_semiconductor",
            "molecule_gas_phase",
            "adsorbate_screening",
        }

    def test_get_surface_presets(self):
        """Test getting surface presets."""
        presets = get_presets_by_category("Surface")

        assert len(presets) == 4
        assert set(presets) == {
            "surface_basic",
            "surface_dirty",
            "surface_metal",
            "surface_semiconductor",
        }

    def test_get_phonon_presets(self):
        """Test getting phonon presets."""
        presets = get_presets_by_category("Phonon")

        assert len(presets) == 3
        assert set(presets) == {
            "phonon_dirty",
            "phonon_standard",
            "phonon_high_accuracy",
        }

    def test_invalid_category_raises_error(self):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown category"):
            get_presets_by_category("InvalidCategory")


class TestApplyTierPreset:
    """Test apply_tier_preset function."""

    def test_apply_preset_to_maker(self):
        """Test applying a preset to a Maker."""
        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply preset
        maker = apply_tier_preset(maker, "surface_metal")

        # Check that preset was applied
        assert maker.input_set_generator.tier == "intermediate"
        assert "OccupationFunction" in maker.input_set_generator.user_params
        assert maker.input_set_generator.user_params["OccupationFunction"] == "MP"

    def test_preset_parameter_merging(self):
        """Test that preset parameters are merged correctly."""
        # Create maker with an existing (valid FDF) param
        maker = RelaxMaker.fixed_cell_relaxation(user_params={"MaxSCFIterations": 50})

        # Apply preset
        maker = apply_tier_preset(maker, "surface_metal")

        # Should have both existing and preset params
        assert "MaxSCFIterations" in maker.input_set_generator.user_params
        assert "OccupationFunction" in maker.input_set_generator.user_params

    def test_override_params(self):
        """Test parameter override functionality."""
        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply preset with overrides
        maker = apply_tier_preset(
            maker,
            "phonon_high_accuracy",
            override_params={"kpts": [10, 10, 10], "custom": "test"},
        )

        # Override should take precedence
        assert maker.input_set_generator.user_params["kpts"] == [10, 10, 10]
        # Custom param should be added
        assert maker.input_set_generator.user_params["custom"] == "test"
        # Preset param should still be there
        assert "PAO.BasisSize" in maker.input_set_generator.user_params

    def test_enabled_modules_merging(self):
        """Test that enabled_modules are merged correctly."""
        maker = RelaxMaker.fixed_cell_relaxation(enabled_modules=["existing_module"])

        # Apply preset with enabled modules
        maker = apply_tier_preset(maker, "phonon_high_accuracy")

        # Should have both existing and preset modules
        assert "existing_module" in maker.input_set_generator.enabled_modules
        assert "phonons" in maker.input_set_generator.enabled_modules
        assert "dos_bands" in maker.input_set_generator.enabled_modules

    def test_disabled_modules_merging(self):
        """Test that disabled_modules are merged correctly."""
        maker = RelaxMaker.fixed_cell_relaxation(disabled_modules=["module1"])

        # Apply preset (relax_dirty has no disabled modules)
        maker = apply_tier_preset(maker, "relax_dirty")

        # Should keep existing disabled modules
        assert "module1" in maker.input_set_generator.disabled_modules


class TestSpecificPresets:
    """Test specific preset configurations."""

    def test_basic_relax_preset(self):
        """Test relax_dirty (formerly basic_relax) preset configuration."""
        preset = get_tier_preset("relax_dirty")

        assert preset["tier"] == "basic"
        assert preset["recommended_params"]["PAO.BasisSize"] == "SZ"
        assert preset["recommended_params"]["a2s_kpts"] == [1, 1, 1]
        assert (
            preset["recommended_params"]["Mesh.Cutoff"] == "50 Ry"
        )  # Updated from 100 Ry

    def test_surface_metal_preset(self):
        """Test surface_metal preset configuration."""
        preset = get_tier_preset("surface_metal")

        assert preset["tier"] == "intermediate"
        assert preset["recommended_params"]["OccupationFunction"] == "MP"
        assert preset["recommended_params"]["OccupationMPOrder"] == 1
        assert preset["recommended_params"]["SCF.Mixer.Method"] == "Pulay"
        assert preset["recommended_params"]["a2s_kpts"] == [6, 6, 1]

    def test_phonon_high_accuracy_preset(self):
        """Test phonon_high_accuracy preset configuration."""
        preset = get_tier_preset("phonon_high_accuracy")

        assert preset["tier"] == "advanced"
        assert "phonons" in preset["enabled_modules"]
        assert "dos_bands" in preset["enabled_modules"]
        assert preset["recommended_params"]["PAO.BasisSize"] == "TZP"
        assert preset["recommended_params"]["a2s_kpts"] == [8, 8, 8]

    def test_magnetic_correlated_preset(self):
        """Test magnetic_correlated preset configuration."""
        preset = get_tier_preset("magnetic_correlated")

        assert preset["tier"] == "advanced"
        assert "dftu" in preset["enabled_modules"]
        assert preset["recommended_params"]["spin"] == "polarized"

    def test_large_system_preset(self):
        """Test large_system preset configuration."""
        preset = get_tier_preset("large_system")

        assert preset["tier"] == "expert"
        assert "parallel" in preset["enabled_modules"]
        assert "solvers" in preset["enabled_modules"]
        assert "efficiency" in preset["enabled_modules"]
        assert preset["recommended_params"]["SolutionMethod"] == "OrderN"


class TestPresetIntegration:
    """Integration tests for preset system."""

    def test_all_presets_apply_without_error(self):
        """Test that all presets can be applied without error."""
        for preset_name in TIER_PRESETS.keys():
            maker = RelaxMaker.fixed_cell_relaxation()
            # Should not raise any error
            maker = apply_tier_preset(maker, preset_name)
            assert maker.input_set_generator.tier in {
                "basic",
                "intermediate",
                "advanced",
                "expert",
                "dirty",
                "ultra",
            }

    def test_preset_with_structure(self):
        """Test that preset works with actual structure."""
        # Create simple structure
        si_lattice = Lattice.cubic(5.43)
        structure = Structure(
            si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
        )

        # Apply preset and create job
        maker = RelaxMaker.fixed_cell_relaxation()
        maker = apply_tier_preset(maker, "relax_standard")

        # Should be able to create input set
        job = maker.make(structure)
        assert job is not None

    def test_multiple_preset_applications(self):
        """Test applying multiple presets in sequence."""
        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply first preset
        maker = apply_tier_preset(maker, "relax_dirty")
        assert maker.input_set_generator.user_params["PAO.BasisSize"] == "SZ"

        # Apply second preset
        # Note: Existing params have precedence over preset params,
        # so SZ from first preset will remain unless overridden explicitly
        maker = apply_tier_preset(
            maker,
            "relax_high_accuracy",
            override_params={"PAO.BasisSize": "TZP"},  # Must override explicitly
        )
        assert maker.input_set_generator.user_params["PAO.BasisSize"] == "TZP"


class TestParameterMergingPrecedence:
    """Test parameter merging precedence rules."""

    def test_precedence_order(self):
        """Test that precedence is: preset < existing < override."""
        # Create maker with existing params (valid FDF / prefixed internal keys)
        maker = RelaxMaker.fixed_cell_relaxation(
            user_params={"a2s_kpts": [2, 2, 2], "MaxSCFIterations": 50}
        )

        # Apply preset with override
        maker = apply_tier_preset(
            maker,
            "relax_standard",  # Has a2s_kpts: [4,4,4]
            override_params={"a2s_kpts": [8, 8, 8], "DM.MixingWeight": 0.3},
        )

        params = maker.input_set_generator.user_params

        # Override should win for a2s_kpts
        assert params["a2s_kpts"] == [8, 8, 8]
        # Existing param should be preserved
        assert params["MaxSCFIterations"] == 50
        # Override param should be added
        assert params["DM.MixingWeight"] == 0.3
        # Preset param (not overridden) should be there
        assert "PAO.BasisSize" in params
        assert params["PAO.BasisSize"] == "DZP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
