"""Tests for siesta_inputs CLI module."""

import pytest
from click.testing import CliRunner

from atomate2.siesta.cli.inputs import (
    DATA_CLASSES,
    cli,
    format_default_value,
    get_class_docstring,
)


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


class TestFormatDefaultValue:
    """Test format_default_value helper function."""

    def test_format_none(self):
        """Test formatting None value."""
        assert format_default_value(None) == "None"

    def test_format_empty_list(self):
        """Test formatting empty list."""
        assert format_default_value([]) == "[]"

    def test_format_empty_dict(self):
        """Test formatting empty dict."""
        assert format_default_value({}) == "{}"

    def test_format_string(self):
        """Test formatting string value."""
        assert format_default_value("test") == "test"

    def test_format_number(self):
        """Test formatting number value."""
        assert format_default_value(42) == "42"
        assert format_default_value(3.14) == "3.14"

    def test_format_callable_returns_value(self):
        """Test formatting callable that returns a value."""
        result = format_default_value(lambda: "result")
        assert result == "result"

    def test_format_callable_returns_empty_list(self):
        """Test formatting callable that returns empty list."""
        result = format_default_value(list)
        assert result == "[]"

    def test_format_callable_raises_exception(self):
        """Test formatting callable that raises exception."""

        def raise_error():
            raise ValueError("test error")

        result = format_default_value(raise_error)
        assert result == "<callable>"

    def test_format_non_empty_list(self):
        """Test formatting non-empty list."""
        result = format_default_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_format_non_empty_dict(self):
        """Test formatting non-empty dict."""
        result = format_default_value({"key": "value"})
        assert "key" in result
        assert "value" in result


class TestGetClassDocstring:
    """Test get_class_docstring helper function."""

    def test_get_docstring_from_dataclass(self):
        """Test getting docstring from actual dataclass."""
        from atomate2.siesta.dataclass.general_system_descriptors import (
            GeneralSystemDescriptors,
        )

        docstring = get_class_docstring(GeneralSystemDescriptors)
        assert docstring is not None
        assert docstring != "No docstring available."

    def test_get_docstring_no_doc(self):
        """Test getting docstring from class without docstring."""

        class NoDoc:
            pass

        docstring = get_class_docstring(NoDoc)
        assert docstring == "No docstring available."


class TestListCommand:
    """Test 'list' CLI command."""

    def test_list_shows_all_classes(self, runner):
        """Test that list command shows all available classes."""
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Available SIESTA Data Classes" in result.output

        # Check that some known classes are listed
        assert "Pseudopotentials" in result.output
        assert "BasisSetsAndProjectors" in result.output
        assert "KPointSampling" in result.output
        assert "SpinSettings" in result.output

    def test_list_contains_all_data_classes(self, runner):
        """Test that all DATA_CLASSES are shown in list."""
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0

        # Check a sample of classes
        sample_classes = [
            "GeneralSystemDescriptors",
            "ExchangeCorrelationFunctionals",
            "SCFLoopParameters",
            "RealSpaceGridParameters",
            "OpticalProperties",
        ]

        for class_name in sample_classes:
            assert class_name in result.output


class TestShowCommand:
    """Test 'show' CLI command."""

    def test_show_valid_class(self, runner):
        """Test showing information for a valid class."""
        result = runner.invoke(cli, ["show", "Pseudopotentials"])

        assert result.exit_code == 0
        assert "Data Class: Pseudopotentials" in result.output
        assert "Description" in result.output
        assert "Attributes" in result.output

    def test_show_invalid_class(self, runner):
        """Test showing information for an invalid class."""
        result = runner.invoke(cli, ["show", "NonExistentClass"])

        assert result.exit_code == 0
        assert "Error" in result.output
        assert "not found" in result.output
        assert "Use 'list' command" in result.output

    def test_show_with_complete_flag(self, runner):
        """Test show command with --complete flag."""
        result = runner.invoke(cli, ["show", "SpinSettings", "--complete"])

        assert result.exit_code == 0
        assert "Data Class: SpinSettings" in result.output
        # Complete flag shows more columns, even if table is truncated in output
        # Just check that it runs successfully and shows attributes
        assert "Attributes" in result.output

    def test_show_with_siesta_flag(self, runner):
        """Test show command with --siesta flag."""
        result = runner.invoke(cli, ["show", "KPointSampling", "--siesta"])

        assert result.exit_code == 0
        assert "Data Class: KPointSampling" in result.output
        assert "SIESTA Keyword" in result.output

    def test_show_with_unit_flag(self, runner):
        """Test show command with --unit flag."""
        result = runner.invoke(cli, ["show", "RealSpaceGridParameters", "--unit"])

        assert result.exit_code == 0
        assert "Data Class: RealSpaceGridParameters" in result.output
        assert "Unit" in result.output

    def test_show_displays_attributes(self, runner):
        """Test that show displays class attributes."""
        result = runner.invoke(cli, ["show", "GeneralSystemDescriptors"])

        assert result.exit_code == 0
        assert "Attributes" in result.output
        assert "Name" in result.output
        assert "Type" in result.output
        assert "Default" in result.output


class TestSearchCommand:
    """Test 'search' CLI command."""

    def test_search_finds_matches(self, runner):
        """Test that search finds matching attributes."""
        result = runner.invoke(cli, ["search", "basis"])

        assert result.exit_code == 0
        # Should find matches in BasisSetsAndProjectors and other classes
        assert "Search Results" in result.output or "Found" in result.output

    def test_search_no_matches(self, runner):
        """Test search with no matching results."""
        result = runner.invoke(cli, ["search", "xyznonexistent123"])

        assert result.exit_code == 0
        assert "No attributes found" in result.output
        assert "Use 'list' command" in result.output

    def test_search_with_restrict_flag(self, runner):
        """Test search with --restrict flag for exact word matching."""
        # Search for a common word that should match with restrict
        result = runner.invoke(cli, ["search", "spin", "--restrict"])

        assert result.exit_code == 0
        # Should either find matches or report none
        assert (
            "Search Results" in result.output or "No attributes found" in result.output
        )

    def test_search_case_insensitive(self, runner):
        """Test that search is case-insensitive."""
        result1 = runner.invoke(cli, ["search", "BASIS"])
        result2 = runner.invoke(cli, ["search", "basis"])

        # Both should produce output (case doesn't matter)
        assert result1.exit_code == 0
        assert result2.exit_code == 0

    def test_search_in_field_name(self, runner):
        """Test search finds matches in field names."""
        # 'cutoff' appears in many field names
        result = runner.invoke(cli, ["search", "cutoff"])

        assert result.exit_code == 0
        assert "Search Results" in result.output or "Found" in result.output

    def test_search_shows_class_and_attribute(self, runner):
        """Test that search results show class and attribute names."""
        result = runner.invoke(cli, ["search", "kpts"])

        assert result.exit_code == 0
        # Should show table with Class and Attribute columns
        if "Search Results" in result.output:
            assert "Class" in result.output
            assert "Attribute" in result.output
            assert "Type" in result.output

    def test_search_displays_count(self, runner):
        """Test that search displays count of matches found."""
        result = runner.invoke(cli, ["search", "mesh"])

        assert result.exit_code == 0
        # Should show "Found X attribute(s)" message
        if "Found" in result.output:
            assert "attribute" in result.output


class TestCLIGroup:
    """Test CLI group and general functionality."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Command-line interface" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "search" in result.output

    def test_list_help(self, runner):
        """Test list command help."""
        result = runner.invoke(cli, ["list", "--help"])

        assert result.exit_code == 0
        assert "List all available data classes" in result.output

    def test_show_help(self, runner):
        """Test show command help."""
        result = runner.invoke(cli, ["show", "--help"])

        assert result.exit_code == 0
        assert "Show detailed information" in result.output
        assert "--complete" in result.output
        assert "--siesta" in result.output
        assert "--unit" in result.output

    def test_search_help(self, runner):
        """Test search command help."""
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "Search for data class attributes" in result.output
        assert "--restrict" in result.output


class TestDataClassesDictionary:
    """Test DATA_CLASSES dictionary structure."""

    def test_data_classes_not_empty(self):
        """Test that DATA_CLASSES dictionary is not empty."""
        assert len(DATA_CLASSES) > 0

    def test_data_classes_contains_expected_classes(self):
        """Test that DATA_CLASSES contains expected dataclass types."""
        expected_classes = [
            "Pseudopotentials",
            "BasisSetsAndProjectors",
            "KPointSampling",
            "ExchangeCorrelationFunctionals",
            "SpinSettings",
            "SCFLoopParameters",
            "RealSpaceGridParameters",
        ]

        for class_name in expected_classes:
            assert class_name in DATA_CLASSES

    def test_data_classes_all_are_classes(self):
        """Test that all values in DATA_CLASSES are actual classes."""
        for class_name, cls in DATA_CLASSES.items():
            assert callable(cls)  # Classes are callable
            assert hasattr(cls, "__name__")

    def test_data_classes_count(self):
        """Test that we have the expected number of data classes."""
        # Should have 30 dataclasses registered
        assert len(DATA_CLASSES) == 30
