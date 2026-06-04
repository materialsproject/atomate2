"""Tests for SIESTA sets utility functions.

These tests validate:
- pymatgen ↔ ASE conversion functions
- pymatgen → sisl conversion
- YAML file reading (read_outvars)
- FDF to JSON conversion
"""

import pytest
import json
import yaml
from pymatgen.core import Structure, Lattice
import numpy as np

from atomate2.siesta.sets.utils import (
    pymatgen_to_ase,
    pymatgen_to_ase_v2,
    ase_v2_to_pymatgen,
    pymatgen_to_sisl,
    read_outvars,
    siesta_fdf_to_json,
)


@pytest.fixture
def si_structure():
    """Silicon structure for testing."""
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.fixture
def al_structure():
    """Aluminum structure for testing."""
    lattice = Lattice.cubic(4.05)
    return Structure(lattice, ["Al"], [[0, 0, 0]])


class TestPymatgenToAse:
    """Test pymatgen_to_ase conversion function."""

    def test_basic_conversion(self, si_structure):
        """Test basic pymatgen to ASE conversion."""
        ase_atoms = pymatgen_to_ase(si_structure)

        # Check that conversion produces ASE Atoms object
        from ase import Atoms

        assert isinstance(ase_atoms, Atoms)

        # Check number of atoms
        assert len(ase_atoms) == 2

        # Check atomic numbers (Si = 14)
        assert all(ase_atoms.get_atomic_numbers() == 14)

        # Check cell
        assert ase_atoms.cell is not None
        assert ase_atoms.pbc.all()  # All periodic

    def test_conversion_with_different_structure(self, al_structure):
        """Test conversion with aluminum structure."""
        ase_atoms = pymatgen_to_ase(al_structure)

        from ase import Atoms

        assert isinstance(ase_atoms, Atoms)
        assert len(ase_atoms) == 1
        assert ase_atoms.get_atomic_numbers()[0] == 13  # Al

    def test_conversion_with_ghost_tags(self, si_structure):
        """Test conversion with ghost atoms."""
        ghost_tags = [False, True]  # Second atom is ghost

        ase_atoms = pymatgen_to_ase(si_structure, ghost_tags=ghost_tags)

        # Check atomic numbers (ghost atoms have negative numbers)
        atomic_numbers = ase_atoms.get_atomic_numbers()
        assert atomic_numbers[0] == 14  # Normal Si
        assert atomic_numbers[1] == -14  # Ghost Si (negative)

    def test_conversion_preserves_cell(self, si_structure):
        """Test that conversion preserves cell parameters."""
        ase_atoms = pymatgen_to_ase(si_structure)

        # Check cell volume
        cell_volume = ase_atoms.get_volume()
        assert cell_volume == pytest.approx(si_structure.volume, rel=1e-5)

    def test_conversion_without_ghost_tags(self, si_structure):
        """Test conversion without ghost_tags (should log info)."""
        # Structure without ghost_tags should work fine
        ase_atoms = pymatgen_to_ase(si_structure)

        assert len(ase_atoms) == 2
        # All atoms should be normal (positive atomic numbers)
        assert all(ase_atoms.get_atomic_numbers() > 0)


class TestPymatgenToAseV2:
    """Test pymatgen_to_ase_v2 simplified conversion."""

    def test_v2_basic_conversion(self, si_structure):
        """Test v2 conversion."""
        ase_atoms = pymatgen_to_ase_v2(si_structure)

        from ase import Atoms

        assert isinstance(ase_atoms, Atoms)
        assert len(ase_atoms) == 2

    def test_v2_conversion_different_structure(self, al_structure):
        """Test v2 with different structure."""
        ase_atoms = pymatgen_to_ase_v2(al_structure)

        assert len(ase_atoms) == 1
        assert ase_atoms.get_atomic_numbers()[0] == 13

    def test_v2_preserves_cell(self, si_structure):
        """Test that v2 preserves cell."""
        ase_atoms = pymatgen_to_ase_v2(si_structure)

        cell_volume = ase_atoms.get_volume()
        assert cell_volume == pytest.approx(si_structure.volume, rel=1e-5)

    def test_v2_vs_v1_consistency(self, si_structure):
        """Test that v1 and v2 produce similar results (without ghost atoms)."""
        ase_v1 = pymatgen_to_ase(si_structure)
        ase_v2 = pymatgen_to_ase_v2(si_structure)

        # Both should have same number of atoms
        assert len(ase_v1) == len(ase_v2)

        # Same atomic numbers
        assert np.array_equal(ase_v1.get_atomic_numbers(), ase_v2.get_atomic_numbers())


class TestAseV2ToPymatgen:
    """Test ase_v2_to_pymatgen reverse conversion."""

    def test_reverse_conversion(self, si_structure):
        """Test ASE to pymatgen conversion."""
        # Convert pymatgen -> ASE
        ase_atoms = pymatgen_to_ase_v2(si_structure)

        # Convert back ASE -> pymatgen
        structure_back = ase_v2_to_pymatgen(ase_atoms)

        # Check it's a Structure
        assert isinstance(structure_back, Structure)

        # Check number of sites
        assert len(structure_back) == len(si_structure)

    def test_reverse_conversion_different_structure(self, al_structure):
        """Test reverse conversion with different structure."""
        ase_atoms = pymatgen_to_ase_v2(al_structure)
        structure_back = ase_v2_to_pymatgen(ase_atoms)

        assert isinstance(structure_back, Structure)
        assert len(structure_back) == 1

    def test_roundtrip_conversion(self, si_structure):
        """Test roundtrip: pymatgen -> ASE -> pymatgen."""
        ase_atoms = pymatgen_to_ase_v2(si_structure)
        structure_back = ase_v2_to_pymatgen(ase_atoms)

        # Should have same composition
        assert structure_back.composition == si_structure.composition

        # Should have similar volume
        assert structure_back.volume == pytest.approx(si_structure.volume, rel=1e-5)


class TestPymatgenToSisl:
    """Test pymatgen_to_sisl conversion."""

    def test_basic_sisl_conversion(self, si_structure):
        """Test basic pymatgen to sisl conversion."""
        sisl_geom = pymatgen_to_sisl(si_structure)

        # Check that it's a sisl Geometry object
        import sisl

        assert isinstance(sisl_geom, sisl.Geometry)

        # Check number of atoms
        assert len(sisl_geom) == 2

    def test_sisl_conversion_different_structure(self, al_structure):
        """Test sisl conversion with aluminum."""
        sisl_geom = pymatgen_to_sisl(al_structure)

        import sisl

        assert isinstance(sisl_geom, sisl.Geometry)
        assert len(sisl_geom) == 1

    def test_sisl_conversion_with_ghost_tags(self, si_structure):
        """Test sisl conversion with ghost atoms."""
        ghost_tags = [False, True]

        sisl_geom = pymatgen_to_sisl(si_structure, ghost_tags=ghost_tags)

        import sisl

        assert isinstance(sisl_geom, sisl.Geometry)
        assert len(sisl_geom) == 2

    def test_sisl_conversion_without_ghost_tags(self, si_structure):
        """Test sisl conversion without ghost_tags."""
        sisl_geom = pymatgen_to_sisl(si_structure)

        assert len(sisl_geom) == 2

    def test_sisl_preserves_cell(self, si_structure):
        """Test that sisl conversion preserves cell."""
        sisl_geom = pymatgen_to_sisl(si_structure)

        # Check that cell exists
        assert sisl_geom.cell is not None

        # Check volume approximately preserved
        sisl_volume = sisl_geom.volume
        assert sisl_volume == pytest.approx(si_structure.volume, rel=1e-2)


class TestReadOutvars:
    """Test read_outvars YAML reading function."""

    def test_read_valid_yaml(self, tmp_path):
        """Test reading valid YAML file."""
        yaml_file = tmp_path / "test.yml"

        # Create test YAML file
        test_data = {
            "energy": -123.45,
            "forces": [[0.1, 0.2, 0.3]],
            "stress": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }

        with open(yaml_file, "w") as f:
            yaml.dump(test_data, f)

        # Read the file
        data = read_outvars(str(yaml_file))

        assert data is not None
        assert data["energy"] == -123.45
        assert "forces" in data
        assert "stress" in data

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        result = read_outvars("/nonexistent/path/file.yml")

        # Should return None for nonexistent file
        assert result is None

    def test_read_empty_yaml(self, tmp_path):
        """Test reading empty YAML file."""
        yaml_file = tmp_path / "empty.yml"
        yaml_file.write_text("")

        data = read_outvars(str(yaml_file))

        # Empty YAML returns None
        assert data is None

    def test_read_yaml_with_nested_data(self, tmp_path):
        """Test reading YAML with nested structures."""
        yaml_file = tmp_path / "nested.yml"

        test_data = {
            "calculation": {
                "type": "relax",
                "parameters": {"basis": "DZP", "kpts": [4, 4, 4]},
            }
        }

        with open(yaml_file, "w") as f:
            yaml.dump(test_data, f)

        data = read_outvars(str(yaml_file))

        assert data is not None
        assert "calculation" in data
        assert data["calculation"]["type"] == "relax"
        assert data["calculation"]["parameters"]["basis"] == "DZP"


class TestSiestaFdfToJson:
    """Test siesta_fdf_to_json conversion function."""

    def test_fdf_to_json_with_provided_data(self, tmp_path):
        """Test FDF to JSON with pre-provided data."""
        json_output = tmp_path / "output.json"

        # Provide FDF data directly
        fdf_data = {
            "SCFMustConverge": True,
            "Spin": "polarized",
            "XC.functional": "GGA",
            "MeshCutoff": "300 Ry",
        }

        # Note: siesta_fdf_path is required but won't be read if fdf_data provided
        # Create a dummy FDF file
        fdf_file = tmp_path / "dummy.fdf"
        fdf_file.write_text("SystemName Test\n")

        siesta_fdf_to_json(str(fdf_file), str(json_output), fdf_data=fdf_data)

        # Check JSON file was created
        assert json_output.exists()

        # Read and verify JSON content
        with open(json_output, "r") as f:
            data = json.load(f)

        assert data["SCFMustConverge"] is True
        assert data["Spin"] == "polarized"
        assert data["XC.functional"] == "GGA"

    def test_json_output_format(self, tmp_path):
        """Test that JSON output is properly formatted."""
        json_output = tmp_path / "formatted.json"
        fdf_file = tmp_path / "dummy.fdf"
        fdf_file.write_text("SystemName Test\n")

        fdf_data = {"test_key": "test_value"}

        siesta_fdf_to_json(str(fdf_file), str(json_output), fdf_data=fdf_data)

        # Check file is valid JSON
        with open(json_output, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "test_key" in data

    def test_json_with_complex_data(self, tmp_path):
        """Test JSON conversion with complex nested data."""
        json_output = tmp_path / "complex.json"
        fdf_file = tmp_path / "dummy.fdf"
        fdf_file.write_text("SystemName Test\n")

        fdf_data = {
            "simple": "value",
            "array": [1, 2, 3],
            "nested": {"key1": "val1", "key2": [4, 5, 6]},
        }

        siesta_fdf_to_json(str(fdf_file), str(json_output), fdf_data=fdf_data)

        with open(json_output, "r") as f:
            data = json.load(f)

        assert data["simple"] == "value"
        assert data["array"] == [1, 2, 3]
        assert data["nested"]["key1"] == "val1"


class TestUtilsFunctionExistence:
    """Test that all utility functions exist and are callable."""

    def test_all_functions_exist(self):
        """Test that all expected utility functions exist."""
        functions = [
            pymatgen_to_ase,
            pymatgen_to_ase_v2,
            ase_v2_to_pymatgen,
            pymatgen_to_sisl,
            read_outvars,
            siesta_fdf_to_json,
        ]

        for func in functions:
            assert callable(func)


class TestConversionEdgeCases:
    """Test edge cases and error handling in conversion functions."""

    def test_pymatgen_to_ase_single_atom(self):
        """Test conversion with single atom."""
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, ["H"], [[0, 0, 0]])

        ase_atoms = pymatgen_to_ase(structure)

        assert len(ase_atoms) == 1
        assert ase_atoms.get_atomic_numbers()[0] == 1  # Hydrogen

    def test_ghost_tags_list_length_matches(self, si_structure):
        """Test that ghost_tags must match number of atoms."""
        ghost_tags = [False, True]  # Correct length

        ase_atoms = pymatgen_to_ase(si_structure, ghost_tags=ghost_tags)

        assert len(ase_atoms) == len(ghost_tags)

    def test_conversion_preserves_periodicity(self, si_structure):
        """Test that periodic boundary conditions are preserved."""
        ase_atoms = pymatgen_to_ase(si_structure)

        # All directions should be periodic (use == for numpy bool comparison)
        assert ase_atoms.pbc[0] == True
        assert ase_atoms.pbc[1] == True
        assert ase_atoms.pbc[2] == True
        # Or simply check all at once
        assert ase_atoms.pbc.all()
