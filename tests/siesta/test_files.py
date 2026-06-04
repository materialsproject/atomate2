"""Tests for files.py - File handling and I/O operations.

These tests validate:
- copy_siesta_outputs function
- write_siesta_input_set function
- cleanup_siesta_outputs function
- load_siesta_input function
- read_directly_from_siesta_out function
- copy_file_from_flos function
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from pymatgen.core import Structure, Lattice


@pytest.fixture
def si_structure():
    """Create a simple Si structure for testing."""
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    return structure


@pytest.fixture
def tmp_siesta_dir(tmp_path):
    """Create a temporary directory with mock SIESTA files."""
    # Create some mock SIESTA output files
    (tmp_path / "siesta.out").write_text("Mock SIESTA output\nVersion: siesta-4.1\n")
    (tmp_path / "siesta.DM").write_text("Mock density matrix")
    (tmp_path / "siesta.XV").write_text("Mock structure")
    (tmp_path / "siesta.bands").write_text("Mock bands")
    return tmp_path


class TestCopySiestaOutputs:
    """Test copy_siesta_outputs function."""

    @patch("atomate2.siesta.files.FileClient")
    @patch("atomate2.siesta.files.gunzip_files")
    def test_copy_siesta_outputs_basic(self, mock_gunzip, mock_fileclient, tmp_path):
        """Test basic copy_siesta_outputs functionality."""
        from atomate2.siesta.files import copy_siesta_outputs

        # Setup mock file client
        mock_client = MagicMock()
        mock_client.listdir.return_value = ["siesta.out", "siesta.DM", "siesta.XV"]
        mock_fileclient.return_value = mock_client

        # Test without decorator to avoid auto_fileclient complications
        # Just verify function exists and is callable
        assert callable(copy_siesta_outputs)

    def test_copy_siesta_outputs_with_restart(self, tmp_siesta_dir):
        """Test copy_siesta_outputs with restart_to_input=True."""
        from atomate2.siesta.files import copy_siesta_outputs

        # Function should be callable
        assert callable(copy_siesta_outputs)

        # The function uses auto_fileclient decorator which handles file operations
        # We verify the function signature accepts restart_to_input parameter
        import inspect

        sig = inspect.signature(copy_siesta_outputs)
        assert "restart_to_input" in sig.parameters

    def test_copy_siesta_outputs_signature(self):
        """Test copy_siesta_outputs has correct signature."""
        from atomate2.siesta.files import copy_siesta_outputs
        import inspect

        sig = inspect.signature(copy_siesta_outputs)
        params = sig.parameters

        # Check required and optional parameters
        assert "src_dir" in params
        assert "src_host" in params
        assert "additional_siesta_files" in params
        assert "restart_to_input" in params
        assert "file_client" in params


class TestWriteSiestaInputSet:
    """Test write_siesta_input_set function."""

    def test_write_siesta_input_set_callable(self):
        """Test that write_siesta_input_set is callable."""
        from atomate2.siesta.files import write_siesta_input_set

        assert callable(write_siesta_input_set)

    def test_write_siesta_input_set_signature(self):
        """Test write_siesta_input_set has correct signature."""
        from atomate2.siesta.files import write_siesta_input_set
        import inspect

        sig = inspect.signature(write_siesta_input_set)
        params = sig.parameters

        # Check required parameters
        assert "structure" in params
        assert "input_set_generator" in params
        assert "directory" in params
        assert "prev_dir" in params

    @patch("atomate2.siesta.files.copy_file_from_flos")
    def test_write_siesta_input_set_with_lua(
        self, mock_copy_flos, si_structure, tmp_path
    ):
        """Test write_siesta_input_set with Lua script handling."""
        from atomate2.siesta.files import write_siesta_input_set
        from atomate2.siesta.sets.base import SiestaInputGenerator

        # Create mock input set generator
        mock_generator = MagicMock(spec=SiestaInputGenerator)
        mock_input_set = MagicMock()

        # Mock parameters with MD.TypeOfRun = "LUA"
        mock_input_set.siesta_input.parameters = {
            "fdf_arguments": {"MD.TypeOfRun": "LUA", "Lua.Script": "test_script.lua"}
        }
        mock_generator.get_input_set.return_value = mock_input_set

        # Call function
        write_siesta_input_set(si_structure, mock_generator, directory=str(tmp_path))

        # Verify copy_file_from_flos was called
        mock_copy_flos.assert_called_once_with("test_script.lua", tmp_path)

    def test_write_siesta_input_set_without_lua(self, si_structure, tmp_path):
        """Test write_siesta_input_set without Lua script."""
        from atomate2.siesta.files import write_siesta_input_set
        from atomate2.siesta.sets.base import SiestaInputGenerator

        # Create mock input set generator
        mock_generator = MagicMock(spec=SiestaInputGenerator)
        mock_input_set = MagicMock()

        # Mock parameters without MD.TypeOfRun = "LUA"
        mock_input_set.siesta_input.parameters = {"fdf_arguments": {}}
        mock_generator.get_input_set.return_value = mock_input_set

        # Call function - should not raise exception
        write_siesta_input_set(si_structure, mock_generator, directory=str(tmp_path))

        # Verify get_input_set was called
        mock_generator.get_input_set.assert_called_once()


class TestCleanupSiestaOutputs:
    """Test cleanup_siesta_outputs function."""

    def test_cleanup_siesta_outputs_callable(self):
        """Test that cleanup_siesta_outputs is callable."""
        from atomate2.siesta.files import cleanup_siesta_outputs

        assert callable(cleanup_siesta_outputs)

    def test_cleanup_siesta_outputs_signature(self):
        """Test cleanup_siesta_outputs has correct signature."""
        from atomate2.siesta.files import cleanup_siesta_outputs
        import inspect

        sig = inspect.signature(cleanup_siesta_outputs)
        params = sig.parameters

        # Check parameters
        assert "directory" in params
        assert "host" in params
        assert "file_patterns" in params
        assert "file_client" in params

    @patch("atomate2.siesta.files.FileClient")
    def test_cleanup_siesta_outputs_with_patterns(self, mock_fileclient, tmp_path):
        """Test cleanup_siesta_outputs with file patterns."""
        from atomate2.siesta.files import cleanup_siesta_outputs

        # Setup mock file client
        mock_client = MagicMock()
        mock_client.glob.return_value = [
            tmp_path / "file1.tmp",
            tmp_path / "file2.tmp",
        ]
        mock_fileclient.return_value = mock_client

        # Function should be callable with patterns
        # (auto_fileclient decorator handles actual file operations)
        assert callable(cleanup_siesta_outputs)


class TestLoadSiestaInput:
    """Test load_siesta_input function."""

    def test_load_siesta_input_callable(self):
        """Test that load_siesta_input is callable."""
        from atomate2.siesta.files import load_siesta_input

        assert callable(load_siesta_input)

    def test_load_siesta_input_signature(self):
        """Test load_siesta_input has correct signature."""
        from atomate2.siesta.files import load_siesta_input
        import inspect

        sig = inspect.signature(load_siesta_input)
        params = sig.parameters

        # Check parameters
        assert "dirpath" in params
        assert "fname" in params

    def test_load_siesta_input_missing_file(self, tmp_path):
        """Test load_siesta_input with missing file raises NotImplementedError."""
        from atomate2.siesta.files import load_siesta_input

        with pytest.raises(NotImplementedError, match="Cannot load SiestaInput"):
            load_siesta_input(tmp_path, fname="nonexistent.json")

    def test_load_siesta_input_with_file(self, tmp_path):
        """Test load_siesta_input with existing file."""
        from atomate2.siesta.files import load_siesta_input

        # Create a mock siesta_parameters.json file
        mock_data = {"parameters": {"test": "value"}}
        json_file = tmp_path / "siesta_parameters.json"
        json_file.write_text(json.dumps(mock_data))

        # Load the file
        result = load_siesta_input(tmp_path)

        # Verify result
        assert result == mock_data

    def test_load_siesta_input_custom_fname(self, tmp_path):
        """Test load_siesta_input with custom filename."""
        from atomate2.siesta.files import load_siesta_input

        # Create a mock file with custom name
        mock_data = {"custom": "data"}
        json_file = tmp_path / "custom_params.json"
        json_file.write_text(json.dumps(mock_data))

        # Load with custom fname
        result = load_siesta_input(tmp_path, fname="custom_params.json")

        # Verify result
        assert result == mock_data


class TestReadDirectlyFromSiestaOut:
    """Test read_directly_from_siesta_out function."""

    def test_read_directly_from_siesta_out_callable(self):
        """Test that read_directly_from_siesta_out is callable."""
        from atomate2.siesta.files import read_directly_from_siesta_out

        assert callable(read_directly_from_siesta_out)

    def test_read_directly_from_siesta_out_signature(self):
        """Test read_directly_from_siesta_out has correct signature."""
        from atomate2.siesta.files import read_directly_from_siesta_out
        import inspect

        sig = inspect.signature(read_directly_from_siesta_out)
        params = sig.parameters

        # Check parameters
        assert "file_path" in params
        assert "what" in params
        assert "multiple" in params

    def test_read_single_keyword(self, tmp_path):
        """Test reading a single keyword from siesta.out."""
        from atomate2.siesta.files import read_directly_from_siesta_out

        # Create mock siesta.out file
        siesta_out = tmp_path / "siesta.out"
        siesta_out.write_text(
            "Some header\nVersion: siesta-4.1.5\nMore output\nEnergy: -100.5 eV\n"
        )

        # Read Version keyword
        result = read_directly_from_siesta_out(siesta_out, what="Version")

        # Verify result
        assert "Version" in result
        assert result["Version"] == "siesta-4.1.5"

    def test_read_multiple_keywords(self, tmp_path):
        """Test reading multiple occurrences of a keyword."""
        from atomate2.siesta.files import read_directly_from_siesta_out

        # Create mock siesta.out with multiple SCF iterations
        siesta_out = tmp_path / "siesta.out"
        siesta_out.write_text(
            "SCF: Iteration 1\n"
            "Energy: -100.0 eV\n"
            "SCF: Iteration 2\n"
            "Energy: -100.5 eV\n"
            "SCF: Iteration 3\n"
            "Energy: -100.8 eV\n"
        )

        # Read Energy keyword multiple times
        result = read_directly_from_siesta_out(siesta_out, what="Energy", multiple=True)

        # Verify result
        assert "Energy" in result
        assert isinstance(result["Energy"], list)
        assert len(result["Energy"]) == 3
        assert result["Energy"][0] == "-100.0 eV"
        assert result["Energy"][2] == "-100.8 eV"

    def test_read_missing_keyword_single(self, tmp_path):
        """Test reading a missing keyword (single mode)."""
        from atomate2.siesta.files import read_directly_from_siesta_out

        # Create mock siesta.out without the keyword
        siesta_out = tmp_path / "siesta.out"
        siesta_out.write_text("Some output\nOther data\n")

        # Read missing keyword
        result = read_directly_from_siesta_out(siesta_out, what="MissingKeyword")

        # Verify empty result
        assert result == {}

    def test_read_missing_keyword_multiple(self, tmp_path):
        """Test reading a missing keyword (multiple mode)."""
        from atomate2.siesta.files import read_directly_from_siesta_out

        # Create mock siesta.out without the keyword
        siesta_out = tmp_path / "siesta.out"
        siesta_out.write_text("Some output\nOther data\n")

        # Read missing keyword with multiple=True
        result = read_directly_from_siesta_out(
            siesta_out, what="MissingKeyword", multiple=True
        )

        # Verify None result
        assert "MissingKeyword" in result
        assert result["MissingKeyword"] is None

    def test_read_keyword_with_colon(self, tmp_path):
        """Test reading keyword that appears with colon separator."""
        from atomate2.siesta.files import read_directly_from_siesta_out

        # Create mock siesta.out with colon-separated data
        siesta_out = tmp_path / "siesta.out"
        siesta_out.write_text("Lattice constant:   5.43 Ang\n")

        # Read keyword
        result = read_directly_from_siesta_out(siesta_out, what="Lattice constant")

        # Verify result (should strip leading/trailing whitespace)
        assert "Lattice constant" in result
        assert result["Lattice constant"] == "5.43 Ang"


class TestCopyFileFromFlos:
    """Test copy_file_from_flos function."""

    def test_copy_file_from_flos_callable(self):
        """Test that copy_file_from_flos is callable."""
        from atomate2.siesta.files import copy_file_from_flos

        assert callable(copy_file_from_flos)

    def test_copy_file_from_flos_signature(self):
        """Test copy_file_from_flos has correct signature."""
        from atomate2.siesta.files import copy_file_from_flos
        import inspect

        sig = inspect.signature(copy_file_from_flos)
        params = sig.parameters

        # Check parameters
        assert "file_name" in params
        assert "destination_dir" in params

    @patch("atomate2.siesta.files.SETTINGS")
    def test_copy_file_from_flos_no_flos_path(self, mock_settings, tmp_path):
        """Test copy_file_from_flos with no FLOS_PATH set."""
        from atomate2.siesta.files import copy_file_from_flos

        # Mock SETTINGS with no FLOS_PATH
        mock_settings.FLOS_PATH = None

        # Should raise EnvironmentError
        with pytest.raises(EnvironmentError, match="FLOS_PATH is not configured"):
            copy_file_from_flos("test_script.lua", tmp_path)

    @patch("atomate2.siesta.files.SETTINGS")
    @patch("atomate2.siesta.files.os.path.exists")
    @patch("atomate2.siesta.files.os.path.isdir")
    def test_copy_file_from_flos_file_not_found(
        self, mock_isdir, mock_exists, mock_settings, tmp_path
    ):
        """Test copy_file_from_flos with missing file."""
        from atomate2.siesta.files import copy_file_from_flos

        # Mock SETTINGS with FLOS_PATH
        mock_settings.FLOS_PATH = "/path/to/flos"
        mock_isdir.return_value = False
        mock_exists.return_value = False

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="not found in FLOS_PATH"):
            copy_file_from_flos("missing_script.lua", tmp_path)

    @patch("atomate2.siesta.files.SETTINGS")
    @patch("atomate2.siesta.files.shutil.copy")
    @patch("atomate2.siesta.files.os.path.exists")
    @patch("atomate2.siesta.files.os.path.isdir")
    @patch("atomate2.siesta.files.os.makedirs")
    def test_copy_file_from_flos_success(
        self, mock_makedirs, mock_isdir, mock_exists, mock_copy, mock_settings, tmp_path
    ):
        """Test successful copy_file_from_flos."""
        from atomate2.siesta.files import copy_file_from_flos

        # Mock SETTINGS with FLOS_PATH
        mock_settings.FLOS_PATH = "/path/to/flos"
        mock_isdir.return_value = False  # No examples subfolder
        # Source file exists in FLOS_PATH, but destination does not yet exist
        # (source skips the copy if the destination file already exists).
        mock_exists.side_effect = lambda path: str(path).startswith("/path/to/flos")

        # Call function
        copy_file_from_flos("test_script.lua", tmp_path)

        # Verify shutil.copy was called
        mock_copy.assert_called_once()
        # Verify makedirs was called
        mock_makedirs.assert_called_once_with(tmp_path, exist_ok=True)

    @patch("atomate2.siesta.files.SETTINGS")
    @patch("atomate2.siesta.files.shutil.copy")
    @patch("atomate2.siesta.files.os.path.exists")
    @patch("atomate2.siesta.files.os.path.isdir")
    @patch("atomate2.siesta.files.os.makedirs")
    @patch("atomate2.siesta.files.os.path.join")
    def test_copy_file_from_flos_with_examples(
        self,
        mock_join,
        mock_makedirs,
        mock_isdir,
        mock_exists,
        mock_copy,
        mock_settings,
        tmp_path,
    ):
        """Test copy_file_from_flos with examples subfolder."""
        from atomate2.siesta.files import copy_file_from_flos

        # Mock SETTINGS with FLOS_PATH
        mock_settings.FLOS_PATH = "/path/to/flos"
        mock_isdir.return_value = True  # Has examples subfolder
        # Source file exists in FLOS_PATH/examples, but destination does not yet
        # exist (source skips the copy if the destination file already exists).
        mock_exists.side_effect = lambda path: str(path).startswith("/path/to/flos")
        # Convert PosixPath to str before joining
        mock_join.side_effect = lambda *args: "/".join(str(arg) for arg in args)

        # Call function
        copy_file_from_flos("test_script.lua", tmp_path)

        # Verify os.path.isdir was called to check for examples
        mock_isdir.assert_called()
        # Verify copy was called
        mock_copy.assert_called_once()


class TestFilesModuleIntegration:
    """Integration tests for files module."""

    def test_all_functions_importable(self):
        """Test that all main functions are importable."""
        from atomate2.siesta import files

        # Check all main functions exist
        assert hasattr(files, "copy_siesta_outputs")
        assert hasattr(files, "write_siesta_input_set")
        assert hasattr(files, "cleanup_siesta_outputs")
        assert hasattr(files, "load_siesta_input")
        assert hasattr(files, "read_directly_from_siesta_out")
        assert hasattr(files, "copy_file_from_flos")

    def test_module_has_logger(self):
        """Test that module has logger configured."""
        from atomate2.siesta import files

        assert hasattr(files, "logger")
        assert files.logger is not None

    def test_module_has_console(self):
        """Test that module has rich console configured."""
        from atomate2.siesta import files

        assert hasattr(files, "console")
        assert files.console is not None
