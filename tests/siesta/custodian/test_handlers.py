"""Tests for SIESTA custodian error handlers.

This module tests all error handlers (SCF, Memory, Time, Numerical) for
automatic error correction during SIESTA calculations.
"""

from __future__ import annotations

from unittest.mock import patch

from atomate2.siesta.custodian.errors import ErrorType, SiestaError
from atomate2.siesta.custodian.handlers import (
    MemoryHandler,
    NumericalHandler,
    SCFConvergenceHandler,
    TimeHandler,
)


class TestSCFConvergenceHandler:
    """Tests for SCF convergence error handler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = SCFConvergenceHandler(max_attempts=10)

        assert handler.max_attempts == 10
        assert handler.max_num_corrections == 10
        assert handler.error_type == ErrorType.SCF_CONVERGENCE
        assert handler.is_terminating is True
        assert handler.is_monitor is False

    def test_initialization_default(self):
        """Test default initialization."""
        handler = SCFConvergenceHandler()

        assert handler.max_attempts == 10
        assert handler.max_num_corrections == 10

    def test_check_no_error(self, tmp_path):
        """Test check when no SCF error present."""
        # Create output file without SCF error
        output_file = tmp_path / "siesta.out"
        output_file.write_text("Job completed successfully\n")

        handler = SCFConvergenceHandler()

        with patch(
            "atomate2.siesta.custodian.handlers.scf.detect_error"
        ) as mock_detect:
            mock_detect.return_value = []
            result = handler.check(str(tmp_path))

        assert result is False

    def test_check_with_scf_error(self, tmp_path):
        """Test check when SCF error is present."""
        handler = SCFConvergenceHandler()

        # Mock detect_error to return SCF convergence error
        scf_error = SiestaError(
            error_type=ErrorType.SCF_CONVERGENCE,
            message="SCF did not converge",
        )

        with patch(
            "atomate2.siesta.custodian.handlers.scf.detect_error"
        ) as mock_detect:
            mock_detect.return_value = [scf_error]
            result = handler.check(str(tmp_path))

        assert result is True

    def test_correct_attempt_1(self, tmp_path):
        """Test first correction attempt (reduce mixing weight)."""
        # Create FDF file
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("SCF.Mixer.Weight  0.1\n")

        handler = SCFConvergenceHandler()
        handler.n_applied_corrections = 0  # Simulate first attempt

        with patch(
            "atomate2.siesta.custodian.handlers.scf.update_fdf_file"
        ) as mock_update:
            result = handler.correct(str(tmp_path))

        # Check that mixing weight reduction was attempted
        assert mock_update.called
        assert "errors" in result
        assert "actions" in result

    def test_correct_attempt_2(self, tmp_path):
        """Test second correction attempt (increase mixing history)."""
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("SCF.Mixer.History  5\n")

        handler = SCFConvergenceHandler()
        handler.n_applied_corrections = 1  # Simulate second attempt

        with patch(
            "atomate2.siesta.custodian.handlers.scf.update_fdf_file"
        ) as mock_update:
            result = handler.correct(str(tmp_path))

        assert mock_update.called
        assert "errors" in result

    def test_correct_max_attempts_exceeded(self, tmp_path):
        """Test that max attempts is enforced via max_num_corrections.

        Max-correction enforcement is delegated to custodian through the
        max_num_corrections property; the handler itself always emits an
        action and relies on custodian to stop once the limit is reached.
        """
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("SystemLabel test\n")

        handler = SCFConvergenceHandler(max_attempts=3)
        assert handler.max_num_corrections == 3

        handler.n_applied_corrections = 3  # Already at max

        result = handler.correct(str(tmp_path))

        # Handler still produces a valid correction; custodian enforces the cap
        assert result["actions"] is not None

    def test_as_dict(self):
        """Test MSONable serialization."""
        handler = SCFConvergenceHandler(max_attempts=10)

        d = handler.as_dict()

        assert d["@module"] == "atomate2.siesta.custodian.handlers.scf"
        assert d["@class"] == "SCFConvergenceHandler"
        assert d["max_attempts"] == 10

    def test_from_dict(self):
        """Test MSONable deserialization."""
        d = {
            "@module": "atomate2.siesta.custodian.handlers.scf",
            "@class": "SCFConvergenceHandler",
            "max_attempts": 10,
        }

        handler = SCFConvergenceHandler.from_dict(d)

        assert handler.max_attempts == 10
        assert handler.max_num_corrections == 10


class TestMemoryHandler:
    """Tests for memory error handler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = MemoryHandler(max_attempts=5)

        assert handler.max_attempts == 5
        assert handler.max_num_corrections == 5
        assert handler.error_type == ErrorType.MEMORY
        assert handler.is_terminating is True

    def test_initialization_default(self):
        """Test default initialization."""
        handler = MemoryHandler()

        assert handler.max_attempts == 4
        assert handler.max_num_corrections == 4

    def test_check_no_error(self, tmp_path):
        """Test check when no memory error present."""
        handler = MemoryHandler()

        with patch(
            "atomate2.siesta.custodian.handlers.memory.detect_error"
        ) as mock_detect:
            mock_detect.return_value = []
            result = handler.check(str(tmp_path))

        assert result is False

    def test_check_with_memory_error(self, tmp_path):
        """Test check when memory error is present."""
        handler = MemoryHandler()

        memory_error = SiestaError(
            error_type=ErrorType.MEMORY,
            message="Out of memory",
        )

        with patch(
            "atomate2.siesta.custodian.handlers.memory.detect_error"
        ) as mock_detect:
            mock_detect.return_value = [memory_error]
            result = handler.check(str(tmp_path))

        assert result is True

    def test_correct_increases_memory(self, tmp_path):
        """Test that correction increases memory allocation."""
        # Create FDF file with memory settings
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("PAO.BasisSize  DZP\n")

        handler = MemoryHandler(max_attempts=5)
        handler.n_applied_corrections = 0

        with patch("atomate2.siesta.custodian.handlers.memory.update_fdf_file"):
            result = handler.correct(str(tmp_path))

        # Should have attempted to update memory settings
        assert "errors" in result
        assert "actions" in result

    def test_correct_max_attempts(self, tmp_path):
        """Test that max attempts is enforced via max_num_corrections.

        Max-correction enforcement is delegated to custodian through the
        max_num_corrections property; the handler always emits an action.
        """
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("MeshCutoff 300 Ry\nPAO.BasisSize DZP\n")

        handler = MemoryHandler(max_attempts=2)
        assert handler.max_num_corrections == 2

        handler.n_applied_corrections = 2

        result = handler.correct(str(tmp_path))

        # Handler still produces a valid correction; custodian enforces the cap
        assert result["actions"] is not None

    def test_as_dict(self):
        """Test MSONable serialization."""
        handler = MemoryHandler(max_attempts=5)

        d = handler.as_dict()

        assert d["@module"] == "atomate2.siesta.custodian.handlers.memory"
        assert d["max_attempts"] == 5

    def test_from_dict(self):
        """Test MSONable deserialization."""
        d = {
            "@module": "atomate2.siesta.custodian.handlers.memory",
            "@class": "MemoryHandler",
            "max_attempts": 5,
        }

        handler = MemoryHandler.from_dict(d)

        assert handler.max_attempts == 5


class TestTimeHandler:
    """Tests for walltime limit error handler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = TimeHandler(max_attempts=3)

        assert handler.max_attempts == 3
        assert handler.max_num_corrections == 3
        assert handler.error_type == ErrorType.TIME_LIMIT
        assert handler.is_terminating is True

    def test_initialization_default(self):
        """Test default initialization."""
        handler = TimeHandler()

        assert handler.max_attempts == 2
        assert handler.max_num_corrections == 2

    def test_check_no_error(self, tmp_path):
        """Test check when no time limit error."""
        handler = TimeHandler()

        with patch(
            "atomate2.siesta.custodian.handlers.time.detect_error"
        ) as mock_detect:
            mock_detect.return_value = []
            result = handler.check(str(tmp_path))

        assert result is False

    def test_check_with_time_error(self, tmp_path):
        """Test check when time limit error is present."""
        handler = TimeHandler()

        time_error = SiestaError(
            error_type=ErrorType.TIME_LIMIT,
            message="Walltime limit reached",
        )

        with patch(
            "atomate2.siesta.custodian.handlers.time.detect_error"
        ) as mock_detect:
            mock_detect.return_value = [time_error]
            result = handler.check(str(tmp_path))

        assert result is True

    def test_correct_restarts_from_checkpoint(self, tmp_path):
        """Test that correction recovers by restarting from saved checkpoints."""
        # Create FDF file plus saved density matrix / structure checkpoints
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("SystemLabel test\n")
        (tmp_path / "siesta.DM").write_text("dummy DM content")
        (tmp_path / "siesta.XV").write_text("dummy XV content")

        handler = TimeHandler(max_attempts=2)
        handler.n_applied_corrections = 0

        with patch("atomate2.siesta.custodian.handlers.time.update_fdf_file"):
            result = handler.correct(str(tmp_path))

        # Should have produced restart actions for the saved checkpoints
        assert "errors" in result
        assert result["actions"] is not None
        assert len(result["actions"]) == 2

    def test_correct_no_fdf_file(self, tmp_path):
        """Test correction when FDF file doesn't exist."""
        handler = TimeHandler()
        handler.n_applied_corrections = 0

        result = handler.correct(str(tmp_path))

        # Should return unfixable
        assert result["actions"] is None

    def test_correct_max_attempts(self, tmp_path):
        """Test correction at max attempts."""
        handler = TimeHandler(max_attempts=1)
        handler.n_applied_corrections = 1

        result = handler.correct(str(tmp_path))

        assert result["actions"] is None

    def test_as_dict(self):
        """Test MSONable serialization."""
        handler = TimeHandler(max_attempts=3)

        d = handler.as_dict()

        assert d["@module"] == "atomate2.siesta.custodian.handlers.time"
        assert d["max_attempts"] == 3

    def test_from_dict(self):
        """Test MSONable deserialization."""
        d = {
            "@module": "atomate2.siesta.custodian.handlers.time",
            "@class": "TimeHandler",
            "max_attempts": 2,
        }

        handler = TimeHandler.from_dict(d)

        assert handler.max_attempts == 2


class TestNumericalHandler:
    """Tests for numerical precision error handler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = NumericalHandler()

        assert handler.error_type == ErrorType.NUMERICAL
        assert handler.is_terminating is True
        assert handler.max_num_corrections == 3

    def test_check_no_error(self, tmp_path):
        """Test check when no numerical error."""
        handler = NumericalHandler()

        with patch(
            "atomate2.siesta.custodian.handlers.numerical.detect_error"
        ) as mock_detect:
            mock_detect.return_value = []
            result = handler.check(str(tmp_path))

        assert result is False

    def test_check_with_numerical_error(self, tmp_path):
        """Test check when numerical error is present."""
        handler = NumericalHandler()

        numerical_error = SiestaError(
            error_type=ErrorType.NUMERICAL,
            message="Numerical instability detected",
        )

        with patch(
            "atomate2.siesta.custodian.handlers.numerical.detect_error"
        ) as mock_detect:
            mock_detect.return_value = [numerical_error]
            result = handler.check(str(tmp_path))

        assert result is True

    def test_correct_increases_precision(self, tmp_path):
        """Test that correction increases numerical precision."""
        # Create FDF file
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("Mesh.Cutoff  200 Ry\n")

        handler = NumericalHandler()
        handler.n_applied_corrections = 0

        with patch(
            "atomate2.siesta.custodian.handlers.numerical.update_fdf_file"
        ) as mock_update:  # noqa: F841
            result = handler.correct(str(tmp_path))

        # Should have attempted to update precision settings
        assert "errors" in result
        assert "actions" in result

    def test_correct_max_attempts(self, tmp_path):
        """Test that max attempts is enforced via max_num_corrections.

        Max-correction enforcement is delegated to custodian through the
        max_num_corrections property; the handler always emits an action.
        """
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("Mesh.Cutoff  200 Ry\n")

        handler = NumericalHandler(max_attempts=1)
        assert handler.max_num_corrections == 1

        handler.n_applied_corrections = 1

        result = handler.correct(str(tmp_path))

        # Handler still produces a valid correction; custodian enforces the cap
        assert result["actions"] is not None

    def test_as_dict(self):
        """Test MSONable serialization."""
        handler = NumericalHandler(max_attempts=3)

        d = handler.as_dict()

        assert d["@module"] == "atomate2.siesta.custodian.handlers.numerical"
        assert d["@class"] == "NumericalHandler"

    def test_from_dict(self):
        """Test MSONable deserialization."""
        d = {
            "@module": "atomate2.siesta.custodian.handlers.numerical",
            "@class": "NumericalHandler",
            "max_attempts": 3,
        }

        handler = NumericalHandler.from_dict(d)

        assert handler.max_num_corrections == 3


class TestHandlerIntegration:
    """Integration tests for multiple handlers."""

    def test_all_handlers_have_error_types(self):
        """Test that all handlers have proper error types."""
        handlers = [
            SCFConvergenceHandler(),
            MemoryHandler(),
            TimeHandler(),
            NumericalHandler(),
        ]

        for handler in handlers:
            assert hasattr(handler, "error_type")
            assert isinstance(handler.error_type, ErrorType)

    def test_all_handlers_have_check_method(self):
        """Test that all handlers implement check method."""
        handlers = [
            SCFConvergenceHandler(),
            MemoryHandler(),
            TimeHandler(),
            NumericalHandler(),
        ]

        for handler in handlers:
            assert hasattr(handler, "check")
            assert callable(handler.check)

    def test_all_handlers_have_correct_method(self):
        """Test that all handlers implement correct method."""
        handlers = [
            SCFConvergenceHandler(),
            MemoryHandler(),
            TimeHandler(),
            NumericalHandler(),
        ]

        for handler in handlers:
            assert hasattr(handler, "correct")
            assert callable(handler.correct)

    def test_all_handlers_are_terminating(self):
        """Test that all handlers are terminating."""
        handlers = [
            SCFConvergenceHandler(),
            MemoryHandler(),
            TimeHandler(),
            NumericalHandler(),
        ]

        for handler in handlers:
            assert handler.is_terminating is True

    def test_all_handlers_are_msonable(self):
        """Test that all handlers support MSONable serialization."""
        handlers = [
            SCFConvergenceHandler(max_attempts=10),
            MemoryHandler(max_attempts=4),
            TimeHandler(max_attempts=2),
            NumericalHandler(max_attempts=3),
        ]

        for handler in handlers:
            # Test as_dict
            d = handler.as_dict()
            assert "@module" in d
            assert "@class" in d

            # Test from_dict round-trip
            handler_class = type(handler)
            restored = handler_class.from_dict(d)
            assert type(restored) is type(handler)

    def test_handler_max_corrections_property(self):
        """Test that all handlers respect max_num_corrections."""
        handlers = [
            (SCFConvergenceHandler(max_attempts=5), 5),
            (MemoryHandler(max_attempts=3), 3),
            (TimeHandler(max_attempts=2), 2),
            (NumericalHandler(max_attempts=4), 4),
        ]

        for handler, expected_max in handlers:
            assert handler.max_num_corrections == expected_max
