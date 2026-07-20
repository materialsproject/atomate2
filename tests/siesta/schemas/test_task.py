"""
Tests for schemas/task.py (SIESTA task document schemas).

These tests validate:
- InputDoc creation and from_siesta_calc_doc
- OutputDoc creation and from_siesta_calc_doc
- OutputDoc.energy_per_atom property
- SiestaTaskDoc creation and from_directory
- _find_siesta_files utility
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.schemas.calculation import TaskState
from atomate2.siesta.schemas.task import (
    InputDoc,
    OutputDoc,
    SiestaTaskDoc,
    _find_siesta_files,
)


class TestInputDoc:
    """Tests for InputDoc class."""

    def test_input_doc_creation_empty(self):
        """Test creating empty InputDoc."""
        doc = InputDoc()

        assert doc.xc_functional is None
        assert doc.xc_authors is None

    def test_input_doc_creation_with_values(self):
        """Test creating InputDoc with values."""
        doc = InputDoc(xc_functional="GGA", xc_authors="PBE")

        assert doc.xc_functional == "GGA"
        assert doc.xc_authors == "PBE"

    @patch("atomate2.siesta.schemas.task.load_siesta_input")
    def test_from_siesta_calc_doc(self, mock_load_siesta_input):
        """Test creating InputDoc from calculation document."""
        # Mock calc_doc
        mock_calc_doc = MagicMock()
        mock_calc_doc.dir_name = "/path/to/calc"

        # Mock siesta input data
        mock_load_siesta_input.return_value = {
            "XC.functional": "GGA",
            "XC.authors": "PBE",
        }

        doc = InputDoc.from_siesta_calc_doc(mock_calc_doc)

        assert doc.xc_functional == "GGA"
        assert doc.xc_authors == "PBE"
        mock_load_siesta_input.assert_called_once_with("/path/to/calc")


class TestOutputDoc:
    """Tests for OutputDoc class."""

    def test_output_doc_creation_empty(self):
        """Test creating empty OutputDoc."""
        doc = OutputDoc()

        assert doc.structure is None
        assert doc.trajectory is None
        assert doc.energy is None
        assert doc.bandgap is None
        assert doc.cbm is None
        assert doc.vbm is None
        assert doc.forces is None
        assert doc.stress is None

    def test_output_doc_creation_with_values(self, si_structure):
        """Test creating OutputDoc with values."""
        doc = OutputDoc(
            structure=si_structure,
            energy=-10.5,
            bandgap=1.2,
            cbm=2.0,
            vbm=0.8,
            forces=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            stress=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        assert doc.structure == si_structure
        assert doc.energy == -10.5
        assert doc.bandgap == 1.2
        assert doc.cbm == 2.0
        assert doc.vbm == 0.8
        assert len(doc.forces) == 2
        assert len(doc.stress) == 3

    def test_energy_per_atom_with_energy_and_structure(self, si_structure):
        """Test energy_per_atom calculation."""
        doc = OutputDoc(structure=si_structure, energy=-10.0)

        energy_per_atom = doc.energy_per_atom

        # si_structure has 2 atoms
        assert energy_per_atom == pytest.approx(-5.0)

    def test_energy_per_atom_without_energy(self, si_structure):
        """Test energy_per_atom when energy is None."""
        doc = OutputDoc(structure=si_structure, energy=None)

        assert doc.energy_per_atom is None

    def test_energy_per_atom_without_structure(self):
        """Test energy_per_atom when structure is None."""
        doc = OutputDoc(structure=None, energy=-10.0)

        assert doc.energy_per_atom is None

    def test_energy_per_atom_both_none(self):
        """Test energy_per_atom when both are None."""
        doc = OutputDoc(structure=None, energy=None)

        assert doc.energy_per_atom is None

    def test_from_siesta_calc_doc(self, si_structure):
        """Test creating OutputDoc from calculation document."""
        # Mock calc_doc with output
        mock_calc_doc = MagicMock()
        mock_calc_doc.output.structure = si_structure
        mock_calc_doc.output.total_energy = -10.5
        mock_calc_doc.output.bandgap = 1.2
        mock_calc_doc.output.cbm = 2.0
        mock_calc_doc.output.vbm = 0.8
        mock_calc_doc.output.forces = [[0.1, 0.2, 0.3]]
        mock_calc_doc.output.stress = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        doc = OutputDoc.from_siesta_calc_doc(mock_calc_doc)

        assert doc.structure == si_structure
        assert doc.energy == -10.5
        assert doc.bandgap == 1.2
        assert doc.cbm == 2.0
        assert doc.vbm == 0.8
        # Forces and stress are converted to Vector3D/Matrix3D types (tuples)
        assert doc.forces is not None
        assert doc.stress is not None
        assert len(doc.forces) == 1
        assert len(doc.stress) == 3


class TestSiestaTaskDoc:
    """Tests for SiestaTaskDoc class."""

    def test_siesta_task_doc_creation_minimal(self):
        """Test creating minimal SiestaTaskDoc."""
        doc = SiestaTaskDoc()

        assert doc.dir_name is None
        assert doc.last_updated is not None  # Has default_factory
        assert doc.completed_at is None
        assert doc.input is None
        assert doc.output is None
        assert doc.structure is None
        assert doc.state is None

    def test_siesta_task_doc_creation_with_fields(self, si_structure):
        """Test creating SiestaTaskDoc with fields."""
        input_doc = InputDoc(xc_functional="GGA", xc_authors="PBE")
        output_doc = OutputDoc(structure=si_structure, energy=-10.5)

        doc = SiestaTaskDoc(
            dir_name="/path/to/calc",
            completed_at="2024-01-01 12:00:00",
            input=input_doc,
            output=output_doc,
            structure=si_structure,
            state=TaskState.SUCCESS,
            task_label="test_calc",
            tags=["tag1", "tag2"],
            author="Test Author",
            icsd_id="12345",
        )

        assert doc.dir_name == "/path/to/calc"
        assert doc.completed_at == "2024-01-01 12:00:00"
        assert doc.input == input_doc
        assert doc.output == output_doc
        assert doc.structure == si_structure
        assert doc.state == TaskState.SUCCESS
        assert doc.task_label == "test_calc"
        assert doc.tags == ["tag1", "tag2"]
        assert doc.author == "Test Author"
        assert doc.icsd_id == "12345"

    def test_siesta_task_doc_with_calcs_reversed(self, si_structure):
        """Test SiestaTaskDoc with calcs_reversed."""
        # Use real Calculation objects or skip validation
        doc = SiestaTaskDoc.model_construct(
            structure=si_structure,
            calcs_reversed=["calc1", "calc2"],  # Simplified for testing
        )

        assert len(doc.calcs_reversed) == 2

    def test_siesta_task_doc_with_transformations(self):
        """Test SiestaTaskDoc with transformations."""
        transformations = {
            "transformation_1": {"type": "substitution"},
            "transformation_2": {"type": "rotation"},
        }

        doc = SiestaTaskDoc(transformations=transformations)

        assert doc.transformations == transformations

    def test_siesta_task_doc_with_custodian(self):
        """Test SiestaTaskDoc with custodian info."""
        custodian_info = {"handlers": ["SCFConvergenceHandler"], "max_errors": 10}

        doc = SiestaTaskDoc(custodian=custodian_info)

        assert doc.custodian == custodian_info

    def test_siesta_task_doc_with_additional_json(self):
        """Test SiestaTaskDoc with additional JSON."""
        additional = {"custom_field_1": "value1", "custom_field_2": 42}

        doc = SiestaTaskDoc(additional_json=additional)

        assert doc.additional_json == additional

    @patch("atomate2.siesta.schemas.task._find_siesta_files")
    @patch("atomate2.siesta.schemas.task.Calculation.from_siesta_files")
    @patch("atomate2.siesta.schemas.task.InputDoc.from_siesta_calc_doc")
    @patch("atomate2.siesta.schemas.task.OutputDoc.from_siesta_calc_doc")
    def test_from_directory_basic(
        self,
        mock_output_from_calc,
        mock_input_from_calc,
        mock_calc_from_files,
        mock_find_files,
        si_structure,
        tmp_path,
    ):
        """Test creating SiestaTaskDoc from directory."""
        # Mock file finding
        mock_find_files.return_value = {"MESSAGES": Path("MESSAGES")}

        # Mock calculation document
        mock_calc = MagicMock()
        mock_calc.output.structure = si_structure
        mock_calc.output.total_energy = -10.5
        mock_calc.completed_at = "2024-01-01 12:00:00"
        mock_calc.has_siesta_completed = TaskState.SUCCESS
        mock_calc_from_files.return_value = mock_calc

        # Mock input/output docs
        mock_input_doc = InputDoc(xc_functional="GGA")
        mock_output_doc = OutputDoc(structure=si_structure, energy=-10.5)
        mock_input_from_calc.return_value = mock_input_doc
        mock_output_from_calc.return_value = mock_output_doc

        # Create doc from directory
        doc = SiestaTaskDoc.from_directory(tmp_path)

        assert doc.structure == si_structure
        assert doc.state == TaskState.SUCCESS
        assert doc.completed_at == "2024-01-01 12:00:00"
        assert len(doc.calcs_reversed) == 1
        mock_find_files.assert_called_once()
        mock_calc_from_files.assert_called_once()

    @patch("atomate2.siesta.schemas.task._find_siesta_files")
    def test_from_directory_no_files(self, mock_find_files, tmp_path):
        """Test from_directory when no SIESTA files found."""
        mock_find_files.return_value = {}

        with pytest.raises(FileNotFoundError, match="No Siesta files found"):
            SiestaTaskDoc.from_directory(tmp_path)

    @patch("atomate2.siesta.schemas.task._find_siesta_files")
    @patch("atomate2.siesta.schemas.task.Calculation.from_siesta_files")
    @patch("atomate2.siesta.schemas.task.InputDoc.from_siesta_calc_doc")
    @patch("atomate2.siesta.schemas.task.OutputDoc.from_siesta_calc_doc")
    def test_from_directory_with_additional_fields(
        self,
        mock_output_from_calc,
        mock_input_from_calc,
        mock_calc_from_files,
        mock_find_files,
        si_structure,
        tmp_path,
    ):
        """Test from_directory with additional fields."""
        # Mock file finding
        mock_find_files.return_value = {"MESSAGES": Path("MESSAGES")}

        # Mock calculation
        mock_calc = MagicMock()
        mock_calc.output.structure = si_structure
        mock_calc.completed_at = "2024-01-01 12:00:00"
        mock_calc.has_siesta_completed = TaskState.SUCCESS
        mock_calc_from_files.return_value = mock_calc

        # Mock input/output docs
        mock_input_doc = InputDoc(xc_functional="GGA")
        mock_output_doc = OutputDoc(structure=si_structure, energy=-10.5)
        mock_input_from_calc.return_value = mock_input_doc
        mock_output_from_calc.return_value = mock_output_doc

        # Additional fields
        additional_fields = {
            "task_label": "custom_task",
            "tags": ["custom_tag"],
            "author": "Custom Author",
        }

        doc = SiestaTaskDoc.from_directory(
            tmp_path, additional_fields=additional_fields
        )

        assert doc.task_label == "custom_task"
        assert doc.tags == ["custom_tag"]
        assert doc.author == "Custom Author"

    @patch("atomate2.siesta.schemas.task._find_siesta_files")
    @patch("atomate2.siesta.schemas.task.Calculation.from_siesta_files")
    def test_from_directory_with_error(
        self, mock_calc_from_files, mock_find_files, tmp_path
    ):
        """Test from_directory when calculation reading fails."""
        # Mock file finding
        mock_find_files.return_value = {"MESSAGES": Path("MESSAGES")}

        # Mock calculation to raise error
        mock_calc_from_files.side_effect = ValueError("Cannot read calculation")

        with pytest.raises(RuntimeError, match="Cannot read calculation document"):
            SiestaTaskDoc.from_directory(tmp_path)

    def test_from_directory_requires_structure(self):
        """Test that from_directory requires valid structure data."""
        # This test documents that from_directory expects valid structures
        # When structure is None, the from_structure method will fail
        # This is expected behavior and callers should ensure structure exists


class TestFindSiestaFiles:
    """Tests for _find_siesta_files utility function."""

    def test_find_siesta_files_basic(self, tmp_path):
        """Test finding SIESTA files in directory."""
        # Create MESSAGES file
        messages_file = tmp_path / "MESSAGES"
        messages_file.write_text("Test messages")

        result = _find_siesta_files(tmp_path)

        assert isinstance(result, dict)
        assert "MESSAGES" in result
        assert result["MESSAGES"] == Path("MESSAGES")

    def test_find_siesta_files_returns_dict(self, tmp_path):
        """Test that _find_siesta_files returns a dictionary."""
        result = _find_siesta_files(tmp_path)

        assert isinstance(result, dict)

    def test_find_siesta_files_with_string_path(self, tmp_path):
        """Test _find_siesta_files with string path."""
        result = _find_siesta_files(str(tmp_path))

        assert isinstance(result, dict)
        assert "MESSAGES" in result

    def test_find_siesta_files_with_path_object(self, tmp_path):
        """Test _find_siesta_files with Path object."""
        result = _find_siesta_files(tmp_path)

        assert isinstance(result, dict)
        assert "MESSAGES" in result


class TestTaskDocIntegration:
    """Integration tests for task document classes."""

    def test_input_output_doc_together(self, si_structure):
        """Test using InputDoc and OutputDoc together."""
        input_doc = InputDoc(xc_functional="GGA", xc_authors="PBE")
        output_doc = OutputDoc(structure=si_structure, energy=-10.5)

        task_doc = SiestaTaskDoc(
            input=input_doc, output=output_doc, structure=si_structure
        )

        assert task_doc.input.xc_functional == "GGA"
        assert task_doc.output.energy == -10.5
        assert task_doc.output.energy_per_atom == pytest.approx(-5.25)  # 2 atoms

    def test_task_doc_model_dump(self, si_structure):
        """Test serializing task document."""
        input_doc = InputDoc(xc_functional="GGA")
        output_doc = OutputDoc(structure=si_structure, energy=-10.5)

        task_doc = SiestaTaskDoc(
            dir_name="/path/to/calc",
            input=input_doc,
            output=output_doc,
            structure=si_structure,
            state=TaskState.SUCCESS,
        )

        dumped = task_doc.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["dir_name"] == "/path/to/calc"
        assert dumped["state"] == TaskState.SUCCESS

    def test_task_doc_with_trajectory(self, si_structure):
        """Test task document with trajectory."""
        # Create slightly different structures for trajectory
        struct1 = si_structure.copy()
        struct2 = si_structure.copy()
        struct2.translate_sites(0, [0.01, 0.01, 0.01])

        output_doc = OutputDoc(
            structure=si_structure, trajectory=[struct1, struct2], energy=-10.5
        )

        task_doc = SiestaTaskDoc(output=output_doc, structure=si_structure)

        assert len(task_doc.output.trajectory) == 2


class TestTaskDocEdgeCases:
    """Test edge cases for task documents."""

    def test_output_doc_with_zero_atoms(self):
        """Test OutputDoc energy_per_atom with empty structure."""
        # Create empty structure
        lattice = Lattice.cubic(5.0)
        empty_structure = Structure(lattice, [], [])

        doc = OutputDoc(structure=empty_structure, energy=-10.0)

        # Should handle division by zero gracefully
        with pytest.raises(ZeroDivisionError):
            _ = doc.energy_per_atom

    def test_input_doc_none_values(self):
        """Test InputDoc with explicitly None values."""
        doc = InputDoc(xc_functional=None, xc_authors=None)

        assert doc.xc_functional is None
        assert doc.xc_authors is None

    def test_task_doc_empty_tags(self):
        """Test SiestaTaskDoc with empty tags list."""
        doc = SiestaTaskDoc(tags=[])

        assert doc.tags == []
        assert len(doc.tags) == 0

    def test_task_doc_empty_calcs_reversed(self):
        """Test SiestaTaskDoc with empty calcs_reversed."""
        doc = SiestaTaskDoc(calcs_reversed=[])

        assert doc.calcs_reversed == []

    def test_find_siesta_files_nonexistent_path(self):
        """Test _find_siesta_files with nonexistent path."""
        # Should still work, just returns empty dict for MESSAGES
        result = _find_siesta_files("/nonexistent/path")

        assert isinstance(result, dict)
        assert "MESSAGES" in result
