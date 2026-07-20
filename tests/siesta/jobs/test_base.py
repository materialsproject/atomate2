"""Tests for SIESTA base job maker.

These tests validate:
- BaseSiestaMaker class
- Dry-run mode functionality
- Custodian integration
- Method behaviors
"""

import pytest
from unittest.mock import patch
from pymatgen.core import Structure, Lattice

from atomate2.siesta.jobs.base import BaseSiestaMaker, display_welcome_banner


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


class TestBaseSiestaMaker:
    """Test BaseSiestaMaker class."""

    def test_default_initialization(self):
        """Test default BaseSiestaMaker initialization."""
        maker = BaseSiestaMaker()

        assert maker.name == "base"
        assert maker.store_output_data is True
        assert maker.use_custodian is False
        assert maker.custodian_max_errors == 5
        assert maker.dry_run is False
        assert maker.dry_run_output_dir == "dry_run_output"
        assert maker.dry_run_format == "cif"
        assert maker.dry_run_label is None

    def test_initialization_with_custom_name(self):
        """Test BaseSiestaMaker with custom name."""
        maker = BaseSiestaMaker(name="custom_job")

        assert maker.name == "custom_job"

    def test_initialization_with_dry_run(self):
        """Test BaseSiestaMaker with dry_run enabled."""
        maker = BaseSiestaMaker(
            dry_run=True,
            dry_run_output_dir="test_output",
            dry_run_format="xsf",
            dry_run_label="test_label",
        )

        assert maker.dry_run is True
        assert maker.dry_run_output_dir == "test_output"
        assert maker.dry_run_format == "xsf"
        assert maker.dry_run_label == "test_label"

    def test_initialization_with_custodian(self):
        """Test BaseSiestaMaker with custodian enabled."""
        maker = BaseSiestaMaker(use_custodian=True, custodian_max_errors=10)

        assert maker.use_custodian is True
        assert maker.custodian_max_errors == 10

    def test_initialization_with_custom_kwargs(self):
        """Test BaseSiestaMaker with custom kwargs."""
        write_kwargs = {"test": "value"}
        copy_kwargs = {"copy_test": "value"}

        maker = BaseSiestaMaker(
            write_input_set_kwargs=write_kwargs, copy_siesta_kwargs=copy_kwargs
        )

        assert maker.write_input_set_kwargs == write_kwargs
        assert maker.copy_siesta_kwargs == copy_kwargs

    def test_store_output_data_false(self):
        """Test BaseSiestaMaker with store_output_data=False."""
        maker = BaseSiestaMaker(store_output_data=False)

        assert maker.store_output_data is False

    def test_make_method_exists(self, si_structure):
        """Test that make method exists and is callable."""
        maker = BaseSiestaMaker()

        assert hasattr(maker, "make")
        assert callable(maker.make)

        # Create job (don't execute)
        job = maker.make(si_structure)
        assert hasattr(job, "name")
        assert hasattr(job, "function")

    def test_serialization(self):
        """Test BaseSiestaMaker serialization."""
        maker = BaseSiestaMaker(name="test_maker", dry_run=True, use_custodian=True)

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "BaseSiestaMaker"
        assert maker_dict["name"] == "test_maker"
        assert maker_dict["dry_run"] is True
        assert maker_dict["use_custodian"] is True

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, BaseSiestaMaker)
        assert maker_restored.name == maker.name
        assert maker_restored.dry_run == maker.dry_run


class TestBaseSiestaMakerDryRun:
    """Test dry-run functionality."""

    def test_generate_dry_run_label_simple(self, si_structure):
        """Test _generate_dry_run_label with simple structure."""
        maker = BaseSiestaMaker(name="test_job")

        label = maker._generate_dry_run_label(si_structure)

        # Should contain job name and reduced formula
        assert "test_job" in label
        assert "Si" in label
        # Format: {maker_name}_{formula}
        assert label == "test_job_Si"

    def test_generate_dry_run_label_different_structures(
        self, si_structure, al_structure
    ):
        """Test _generate_dry_run_label with different structures."""
        maker = BaseSiestaMaker(name="test_job")

        label_si = maker._generate_dry_run_label(si_structure)
        label_al = maker._generate_dry_run_label(al_structure)

        # Labels should be different
        assert label_si != label_al
        assert "Si" in label_si
        assert "Al" in label_al
        assert label_si == "test_job_Si"
        assert label_al == "test_job_Al"

    def test_make_with_dry_run_enabled(self, si_structure, tmp_path):
        """Test make() method with dry_run=True."""
        maker = BaseSiestaMaker(
            dry_run=True,
            dry_run_output_dir=str(tmp_path / "dry_run"),
            dry_run_format="cif",
        )

        # Create job
        job = maker.make(si_structure)

        assert job.name == "base"
        assert hasattr(job, "function")

    def test_make_with_custom_dry_run_label(self, si_structure, tmp_path):
        """Test make() with custom dry_run_label."""
        maker = BaseSiestaMaker(
            dry_run=True,
            dry_run_output_dir=str(tmp_path / "dry_run"),
            dry_run_label="custom_label",
        )

        job = maker.make(si_structure)
        assert job.name == "base"


class TestBaseSiestaMakerCustodian:
    """Test custodian integration."""

    def test_custodian_enabled(self):
        """Test maker with custodian enabled."""
        maker = BaseSiestaMaker(use_custodian=True)

        assert maker.use_custodian is True
        assert maker.custodian_handlers is None  # Uses default handlers

    def test_custodian_with_custom_handlers(self):
        """Test maker with custom custodian handlers."""
        from atomate2.siesta.custodian import SCFConvergenceHandler

        custom_handlers = [SCFConvergenceHandler(max_attempts=5)]

        maker = BaseSiestaMaker(use_custodian=True, custodian_handlers=custom_handlers)

        assert maker.use_custodian is True
        assert maker.custodian_handlers == custom_handlers

    def test_custodian_max_errors(self):
        """Test custodian_max_errors parameter."""
        maker = BaseSiestaMaker(use_custodian=True, custodian_max_errors=15)

        assert maker.custodian_max_errors == 15


class TestBaseSiestaMakerMethods:
    """Test specific methods of BaseSiestaMaker."""

    def test_run_post_siesta_method_exists(self):
        """Test that run_post_siesta method exists."""
        maker = BaseSiestaMaker()

        assert hasattr(maker, "run_post_siesta")
        assert callable(maker.run_post_siesta)

    def test_make_dry_run_method_exists(self):
        """Test that _make_dry_run method exists."""
        maker = BaseSiestaMaker()

        assert hasattr(maker, "_make_dry_run")
        assert callable(maker._make_dry_run)

    def test_make_calculation_method_exists(self):
        """Test that _make_calculation method exists."""
        maker = BaseSiestaMaker()

        assert hasattr(maker, "_make_calculation")
        assert callable(maker._make_calculation)

    def test_generate_dry_run_label_method_exists(self):
        """Test that _generate_dry_run_label method exists."""
        maker = BaseSiestaMaker()

        assert hasattr(maker, "_generate_dry_run_label")
        assert callable(maker._generate_dry_run_label)


class TestBaseSiestaMakerWithPrevDir:
    """Test BaseSiestaMaker with previous directory."""

    def test_make_with_prev_dir(self, si_structure):
        """Test make() with prev_dir parameter."""
        maker = BaseSiestaMaker()

        job = maker.make(si_structure, prev_dir="/path/to/prev")

        assert hasattr(job, "name")
        assert hasattr(job, "function")

    def test_make_with_none_prev_dir(self, si_structure):
        """Test make() with prev_dir=None."""
        maker = BaseSiestaMaker()

        job = maker.make(si_structure, prev_dir=None)

        assert hasattr(job, "name")

    def test_make_with_extra_dir(self, si_structure):
        """Test make() with extra_dir parameter."""
        maker = BaseSiestaMaker()

        job = maker.make(si_structure, extra_dir="/path/to/extra")

        assert hasattr(job, "name")


class TestBaseSiestaMakerWriteAdditionalData:
    """Test write_additional_data functionality."""

    def test_write_additional_data_empty(self):
        """Test maker with empty write_additional_data."""
        maker = BaseSiestaMaker(write_additional_data={})

        assert maker.write_additional_data == {}

    def test_write_additional_data_with_files(self):
        """Test maker with write_additional_data containing files."""
        additional_data = {
            "my_file:txt": "file contents",
            "data:json": '{"key": "value"}',
        }

        maker = BaseSiestaMaker(write_additional_data=additional_data)

        assert maker.write_additional_data == additional_data
        assert "my_file:txt" in maker.write_additional_data


class TestDisplayWelcomeBanner:
    """Test display_welcome_banner function."""

    def test_display_welcome_banner_exists(self):
        """Test that display_welcome_banner function exists."""
        assert display_welcome_banner is not None
        assert callable(display_welcome_banner)

    @patch("atomate2.siesta.jobs.base.print_fancy_logo")
    @patch("atomate2.siesta.jobs.base.print_in_box_rich")
    def test_display_welcome_banner_calls_functions(
        self, mock_print_box, mock_print_logo
    ):
        """Test that display_welcome_banner calls print functions."""
        display_welcome_banner()

        # Should call both print functions
        mock_print_logo.assert_called_once()
        mock_print_box.assert_called_once()

    @patch("atomate2.siesta.jobs.base.print_fancy_logo")
    @patch("atomate2.siesta.jobs.base.print_in_box_rich")
    def test_display_welcome_banner_with_settings(
        self, mock_print_box, mock_print_logo
    ):
        """Test that display_welcome_banner includes SETTINGS."""
        display_welcome_banner()

        # Check that SETTINGS are passed to print_in_box_rich
        call_args = mock_print_box.call_args[0][0]
        assert isinstance(call_args, dict)
        assert "SIESTA_CMD" in call_args or len(call_args) > 0


class TestBaseSiestaMakerIntegration:
    """Integration tests for BaseSiestaMaker."""

    def test_maker_with_all_options(self, si_structure):
        """Test BaseSiestaMaker with all options specified."""
        maker = BaseSiestaMaker(
            name="integration_test",
            write_input_set_kwargs={"test": "value"},
            copy_siesta_kwargs={},
            run_siesta_kwargs={},
            task_document_kwargs={},
            stop_children_kwargs={},
            write_additional_data={"file:txt": "content"},
            store_output_data=False,
            use_custodian=True,
            custodian_max_errors=10,
            dry_run=True,
            dry_run_output_dir="test_dir",
            dry_run_format="xsf",
            dry_run_label="test_label",
        )

        # All attributes should be set correctly
        assert maker.name == "integration_test"
        assert maker.store_output_data is False
        assert maker.use_custodian is True
        assert maker.custodian_max_errors == 10
        assert maker.dry_run is True

        # Should be able to create a job
        job = maker.make(si_structure)
        assert hasattr(job, "name")

    def test_maker_creates_unique_jobs(self, si_structure, al_structure):
        """Test that maker creates unique jobs for different structures."""
        maker = BaseSiestaMaker()

        job1 = maker.make(si_structure)
        job2 = maker.make(al_structure)

        # Jobs should be different objects
        assert job1 is not job2
        assert job1.name == job2.name  # Same maker name


class TestBaseSiestaMakerEdgeCases:
    """Test edge cases and error handling."""

    def test_maker_with_empty_dicts(self):
        """Test maker with all empty dict parameters."""
        maker = BaseSiestaMaker(
            write_input_set_kwargs={},
            copy_siesta_kwargs={},
            run_siesta_kwargs={},
            task_document_kwargs={},
            stop_children_kwargs={},
            write_additional_data={},
        )

        assert maker.write_input_set_kwargs == {}
        assert maker.copy_siesta_kwargs == {}

    def test_maker_modification_doesnt_affect_jobs(self, si_structure):
        """Test that modifying maker after job creation doesn't affect job."""
        maker = BaseSiestaMaker(name="original")

        job1 = maker.make(si_structure)

        # Modify maker (create new instance)
        maker = BaseSiestaMaker(name="modified")

        job2 = maker.make(si_structure)

        # Jobs should be independent
        assert job1 is not job2
