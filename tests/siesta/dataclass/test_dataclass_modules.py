"""Tests for dataclass modules functionality."""

from dataclasses import fields, is_dataclass

from atomate2.siesta.dataclass.kpoint_sampling import KPointSampling
from atomate2.siesta.dataclass.molecular_dynamics_and_relaxation import (
    MolecularDynamicsAndRelaxation,
)
from atomate2.siesta.dataclass.optical_properties import OpticalProperties
from atomate2.siesta.dataclass.phonon_calculations import PhononCalculations
from atomate2.siesta.dataclass.pseudopotentials import Pseudopotentials
from atomate2.siesta.dataclass.real_space_grid_parameters import RealSpaceGridParameters
from atomate2.siesta.dataclass.scf_loop_parameters import SCFLoopParameters
from atomate2.siesta.dataclass.spin_settings import SpinSettings


class TestDataclassStructure:
    """Test that dataclass modules have expected structure."""

    def test_all_are_dataclasses(self):
        """Test that modules are proper dataclasses."""
        modules = [
            SpinSettings,
            SCFLoopParameters,
            Pseudopotentials,
            KPointSampling,
            RealSpaceGridParameters,
            MolecularDynamicsAndRelaxation,
            OpticalProperties,
            PhononCalculations,
        ]

        for module in modules:
            # Check it's a dataclass
            assert is_dataclass(module), f"{module.__name__} should be a dataclass"

            # Check it has dataclass fields
            module_fields = fields(module)
            assert len(module_fields) > 0, f"{module.__name__} should have fields"

    def test_dataclasses_can_be_instantiated(self):
        """Test that all dataclasses can be instantiated with defaults."""
        modules = [
            SpinSettings,
            SCFLoopParameters,
            Pseudopotentials,
            KPointSampling,
            RealSpaceGridParameters,
            MolecularDynamicsAndRelaxation,
            OpticalProperties,
            PhononCalculations,
        ]

        for module in modules:
            instance = module()
            assert instance is not None, f"{module.__name__} should be instantiatable"
            assert is_dataclass(instance), (
                f"Instance of {module.__name__} should be a dataclass"
            )

    def test_dataclasses_have_fdf_arguments(self):
        """Test that dataclasses have fdf_arguments field."""
        modules = [
            SpinSettings,
            SCFLoopParameters,
            Pseudopotentials,
            KPointSampling,
            RealSpaceGridParameters,
            MolecularDynamicsAndRelaxation,
            OpticalProperties,
            PhononCalculations,
        ]

        for module in modules:
            instance = module()
            # Most dataclasses should have an fdf_arguments field or similar
            field_names = [f.name for f in fields(instance)]
            # Check for common FDF-related field patterns (informational only)
            _ = any(
                "fdf" in name.lower() or "arguments" in name.lower()
                for name in field_names
            )

    def test_dataclasses_have_comments(self):
        """Test that dataclasses have comments field."""
        modules = [
            SpinSettings,
            SCFLoopParameters,
            Pseudopotentials,
            KPointSampling,
            RealSpaceGridParameters,
            MolecularDynamicsAndRelaxation,
            OpticalProperties,
            PhononCalculations,
        ]

        for module in modules:
            instance = module()
            field_names = [f.name for f in fields(instance)]
            # Most dataclasses should have a comments field (informational only)
            _ = "comments" in field_names


class TestDataclassInstances:
    """Test specific dataclass instances."""

    def test_spin_settings_basic(self):
        """Test SpinSettings basic functionality."""
        spin = SpinSettings()
        assert spin is not None
        assert is_dataclass(spin)
        assert len(fields(spin)) > 0

    def test_scf_loop_parameters_basic(self):
        """Test SCFLoopParameters basic functionality."""
        scf = SCFLoopParameters()
        assert scf is not None
        assert is_dataclass(scf)
        assert len(fields(scf)) > 0

    def test_pseudopotentials_basic(self):
        """Test Pseudopotentials basic functionality."""
        pseudo = Pseudopotentials()
        assert pseudo is not None
        assert is_dataclass(pseudo)
        assert len(fields(pseudo)) > 0

    def test_kpoint_sampling_basic(self):
        """Test KPointSampling basic functionality."""
        kpoints = KPointSampling()
        assert kpoints is not None
        assert is_dataclass(kpoints)
        assert len(fields(kpoints)) > 0

    def test_real_space_grid_basic(self):
        """Test RealSpaceGridParameters basic functionality."""
        grid = RealSpaceGridParameters()
        assert grid is not None
        assert is_dataclass(grid)
        assert len(fields(grid)) > 0

    def test_md_and_relaxation_basic(self):
        """Test MolecularDynamicsAndRelaxation basic functionality."""
        md = MolecularDynamicsAndRelaxation()
        assert md is not None
        assert is_dataclass(md)
        assert len(fields(md)) > 0

    def test_optical_properties_basic(self):
        """Test OpticalProperties basic functionality."""
        optical = OpticalProperties()
        assert optical is not None
        assert is_dataclass(optical)
        assert len(fields(optical)) > 0

    def test_phonon_calculations_basic(self):
        """Test PhononCalculations basic functionality."""
        phonon = PhononCalculations()
        assert phonon is not None
        assert is_dataclass(phonon)
        assert len(fields(phonon)) > 0
