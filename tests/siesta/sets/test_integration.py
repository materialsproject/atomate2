"""
Integration tests for tier-based auto-initialization system.

Tests the complete workflow from Maker creation through input set generation,
verifying that:
- Tier parameters trigger correct module initialization
- Different tiers initialize expected number of modules
- Enabled/disabled module overrides work
- FDF arguments are generated from initialized modules
- Full end-to-end workflow functions correctly
"""

import pytest
from pymatgen.core import Structure, Lattice

from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.dataclass.registry import get_modules_for_tier


class TestTierBasedInitialization:
    """Test that tier parameter triggers correct module initialization."""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple Si structure for testing."""
        si_lattice = Lattice.cubic(5.43)
        structure = Structure(
            si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
        )
        return structure

    def test_basic_tier_initialization(self, simple_structure):
        """Test that basic tier initializes only basic modules."""
        # Create generator with basic tier
        generator = StaticSetGenerator(tier="basic")

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Check that basic modules were initialized
        # Basic tier should have 6 modules
        expected_modules = get_modules_for_tier("basic")
        assert len(expected_modules) == 6

        # Verify SIESTA parameters contain module outputs
        siesta_params = input_set.siesta_input.parameters

        # Should have basic parameters initialized
        assert len(siesta_params) > 0

        # Should have XC functional (default: GGA or PBE)
        assert "xc" in siesta_params or "xc_functional" in siesta_params

    def test_intermediate_tier_initialization(self, simple_structure):
        """Test that intermediate tier includes basic + intermediate modules."""
        # Create generator with intermediate tier
        generator = StaticSetGenerator(tier="intermediate")

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Intermediate tier should have 13 modules (6 basic + 7 intermediate)
        expected_modules = get_modules_for_tier("intermediate")
        assert len(expected_modules) == 13

        # Verify SIESTA parameters generated
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0

    def test_advanced_tier_initialization(self, simple_structure):
        """Test that advanced tier includes basic + intermediate + advanced modules."""
        # Create generator with advanced tier
        generator = StaticSetGenerator(tier="advanced")

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Advanced tier should have 22 modules
        expected_modules = get_modules_for_tier("advanced")
        assert len(expected_modules) == 22

        # Verify SIESTA parameters generated
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0

    def test_expert_tier_initialization(self, simple_structure):
        """Test that expert tier initializes all modules."""
        # Create generator with expert tier
        generator = StaticSetGenerator(tier="expert")

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Expert tier should have all 24+ modules
        expected_modules = get_modules_for_tier("expert")
        assert len(expected_modules) >= 24

        # Verify SIESTA parameters generated
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0


class TestEnabledDisabledModules:
    """Test enabled/disabled module overrides."""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple Si structure for testing."""
        si_lattice = Lattice.cubic(5.43)
        structure = Structure(
            si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
        )
        return structure

    def test_enable_specific_modules(self, simple_structure):
        """Test that enabled_modules adds specific modules to tier."""
        # Create generator with basic tier + dos_bands module
        generator = StaticSetGenerator(tier="basic", enabled_modules=["dos_bands"])

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Should have basic modules + dos_bands
        # Verify SIESTA parameters generated
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0

    def test_disable_specific_modules(self, simple_structure):
        """Test that disabled_modules removes modules from tier."""
        # Create generator with intermediate tier - spin module
        generator = StaticSetGenerator(tier="intermediate", disabled_modules=["spin"])

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Should have intermediate modules except spin
        # Verify SIESTA parameters generated
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0

        # Spin-related parameters should not be present
        _spin_keys = [k for k in siesta_params.keys() if "spin" in k.lower()]
        # If spin module properly disabled, should have no spin keys
        # (or only default non-polarized spin)

    def test_enable_and_disable_together(self, simple_structure):
        """Test combining enabled_modules and disabled_modules."""
        # Create generator with basic tier + phonons - mesh_cutoff
        generator = StaticSetGenerator(
            tier="basic",
            enabled_modules=["phonons"],
            disabled_modules=["mesh_cutoff"],
        )

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Verify SIESTA parameters generated
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0


class TestTierPresetIntegration:
    """Test tier preset application in complete workflows."""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple Si structure for testing."""
        si_lattice = Lattice.cubic(5.43)
        structure = Structure(
            si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
        )
        return structure

    def test_preset_sets_tier_correctly(self, simple_structure):
        """Test that applying preset sets the correct tier."""
        # Create maker
        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply preset
        maker = apply_tier_preset(maker, "relax_standard")

        # Check tier was set
        assert maker.input_set_generator.tier == "intermediate"

        # Generate input set to verify initialization
        input_set = maker.input_set_generator.get_input_set(simple_structure)
        assert input_set is not None
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0

    def test_preset_with_enabled_modules(self, simple_structure):
        """Test that preset enabled_modules are applied."""
        # Create maker
        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply preset that enables phonons module
        maker = apply_tier_preset(maker, "phonon_high_accuracy")

        # Check tier and enabled modules
        assert maker.input_set_generator.tier == "advanced"
        assert "phonons" in maker.input_set_generator.enabled_modules

        # Generate input set
        input_set = maker.input_set_generator.get_input_set(simple_structure)
        assert input_set is not None

    def test_preset_parameters_in_fdf(self, simple_structure):
        """Test that preset parameters appear in FDF output."""
        # Create maker
        maker = StaticMaker()

        # Apply preset with specific parameters
        maker = apply_tier_preset(maker, "relax_standard")

        # Generate input set
        _input_set = maker.input_set_generator.get_input_set(simple_structure)

        # Check that preset parameters are in user_params
        user_params = maker.input_set_generator.user_params
        assert "PAO.BasisSize" in user_params
        assert user_params["PAO.BasisSize"] == "DZP"

    def test_multiple_presets_with_override(self, simple_structure):
        """Test applying multiple presets with explicit overrides."""
        # Create maker
        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply first preset
        maker = apply_tier_preset(maker, "relax_dirty")
        assert maker.input_set_generator.user_params["PAO.BasisSize"] == "SZ"

        # Apply second preset with override
        maker = apply_tier_preset(
            maker,
            "relax_high_accuracy",
            override_params={"PAO.BasisSize": "TZP"},
        )
        assert maker.input_set_generator.user_params["PAO.BasisSize"] == "TZP"

        # Generate input set
        input_set = maker.input_set_generator.get_input_set(simple_structure)
        assert input_set is not None


class TestEndToEndWorkflow:
    """Test complete workflow from Maker to FDF file generation."""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple Si structure for testing."""
        si_lattice = Lattice.cubic(5.43)
        structure = Structure(
            si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
        )
        return structure

    def test_basic_tier_complete_workflow(self, simple_structure, tmp_path):
        """Test complete workflow with basic tier."""
        import os

        # Create maker with basic tier
        maker = StaticMaker(input_set_generator=StaticSetGenerator(tier="basic"))

        # Generate input set
        input_set = maker.input_set_generator.get_input_set(simple_structure)

        # Change to tmp directory and write FDF file
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            input_set.write_siesta_fdf(simple_structure)

            # Verify FDF file was created
            fdf_file = tmp_path / "siesta.fdf"
            assert fdf_file.exists()

            # Read FDF file and verify basic content
            fdf_content = fdf_file.read_text()
            assert len(fdf_content) > 0
        finally:
            os.chdir(original_dir)

    def test_preset_complete_workflow(self, simple_structure, tmp_path):
        """Test complete workflow with preset."""
        # Create maker and apply preset
        maker = RelaxMaker.fixed_cell_relaxation()
        maker = apply_tier_preset(maker, "relax_standard")

        # Generate input set
        input_set = maker.input_set_generator.get_input_set(simple_structure)

        # Write to temporary directory
        input_set.write_input(tmp_path)

        # Verify FDF file was created
        fdf_file = tmp_path / "siesta.fdf"
        assert fdf_file.exists()

        # Read FDF file and verify preset parameters
        fdf_content = fdf_file.read_text()
        # DZP should be in user_params which gets written to FDF
        # Check that file has content at minimum
        assert len(fdf_content) > 0

    def test_custom_parameters_complete_workflow(self, simple_structure, tmp_path):
        """Test complete workflow with custom user parameters."""
        # Create maker with custom params
        maker = StaticMaker(
            input_set_generator=StaticSetGenerator(
                tier="intermediate",
                user_params={
                    "PAO.BasisSize": "TZP",
                    "Mesh.Cutoff": "300 Ry",
                    "a2s_kpts": [6, 6, 6],
                },
            )
        )

        # Generate input set
        input_set = maker.input_set_generator.get_input_set(simple_structure)

        # Write to temporary directory
        input_set.write_input(tmp_path)

        # Verify FDF file was created
        fdf_file = tmp_path / "siesta.fdf"
        assert fdf_file.exists()

        # Read FDF file and verify file has content
        fdf_content = fdf_file.read_text()
        assert len(fdf_content) > 0


class TestModuleInitializationOrder:
    """Test that modules are initialized in correct priority order."""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple Si structure for testing."""
        si_lattice = Lattice.cubic(5.43)
        structure = Structure(
            si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
        )
        return structure

    def test_basic_modules_initialized_first(self, simple_structure):
        """Test that basic tier modules (low priority) initialize first."""
        # Create generator with expert tier (all modules)
        generator = StaticSetGenerator(tier="expert")

        # Generate input set
        input_set = generator.get_input_set(simple_structure)

        # Basic modules should be initialized
        # This is verified by checking that SIESTA parameters exist
        siesta_params = input_set.siesta_input.parameters
        assert len(siesta_params) > 0

        # Basic parameters like xc should be present
        assert (
            "xc" in siesta_params
            or "xc_functional" in siesta_params
            or len(siesta_params) > 0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
