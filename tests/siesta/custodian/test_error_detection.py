"""Test script for SIESTA custodian error detection.

This script demonstrates how the error detection system works
with simple test cases.
"""

import tempfile
from pathlib import Path

import pytest

from atomate2.siesta.custodian.errors import (
    MEMORY_PATTERNS,
    SCF_CONVERGENCE_PATTERNS,
    TIME_LIMIT_PATTERNS,
    ErrorType,
    check_for_errors,
    detect_error,
    get_error_type,
)
from atomate2.siesta.custodian.handlers import (
    MemoryHandler,
    SCFConvergenceHandler,
    TimeHandler,
)


class TestErrorPatterns:
    """Test error pattern matching."""

    def test_scf_convergence_pattern(self):
        """Test SCF convergence error detection."""
        # Test the actual SIESTA error message
        content = (
            "SCF_NOT_CONV: SCF did not converge in maximum number of steps (required)."
        )
        assert SCF_CONVERGENCE_PATTERNS.matches(content)

        # Test alternative messages
        content2 = "SCF cycle not converged after 500 iterations"
        assert SCF_CONVERGENCE_PATTERNS.matches(content2)

    def test_memory_pattern(self):
        """Test memory error detection."""
        content = "Out of memory in allocation"
        assert MEMORY_PATTERNS.matches(content)

        content2 = "re_alloc: allocation failed"
        assert MEMORY_PATTERNS.matches(content2)

    def test_time_limit_pattern(self):
        """Test time limit error detection."""
        content = "CANCELLED AT 2025-10-08 DUE TO TIME LIMIT"
        assert TIME_LIMIT_PATTERNS.matches(content)


class TestErrorDetection:
    """Test error detection from files."""

    def test_detect_scf_error(self, tmp_path):
        """Test detecting SCF convergence error from file."""
        # Create test output file
        output_file = tmp_path / "siesta.out"
        output_file.write_text(
            """
            siesta: iscf   Eharris(eV)      E_KS(eV)   FreeEng(eV)   dDmax  Ef(eV)
            siesta:    1     -214.3254     -213.8956     -213.8956  1.5234 -3.4567
            siesta:  100     -214.0656     -214.0658     -214.0658  0.0010 -3.5893

            SCF_NOT_CONV: SCF did not converge in maximum number of steps (required).

            Timer: Elapsed wall time (sec) =      125.345
            """
        )

        # Detect errors
        errors = detect_error(tmp_path)
        assert len(errors) > 0
        assert errors[0].error_type == ErrorType.SCF_CONVERGENCE

        # Test helper functions
        assert check_for_errors(tmp_path) is True
        assert get_error_type(tmp_path) == ErrorType.SCF_CONVERGENCE

    def test_detect_memory_error(self, tmp_path):
        """Test detecting memory error from file."""
        output_file = tmp_path / "siesta.out"
        output_file.write_text(
            """
            Allocation would exceed memory limits
            Out of memory in dhscf
            siesta: Program stopping
            """
        )

        errors = detect_error(tmp_path)
        assert len(errors) > 0
        assert errors[0].error_type == ErrorType.MEMORY

    def test_detect_time_limit_error(self, tmp_path):
        """Test detecting time limit error from scheduler output."""
        # Time limit errors appear in slurm.out
        slurm_file = tmp_path / "slurm.out"
        slurm_file.write_text(
            """
            slurmstepd: error: *** JOB 12345 ON node001 CANCELLED AT 2025-10-08T12:00:00 DUE TO TIME LIMIT ***
            """
        )

        errors = detect_error(tmp_path)
        assert len(errors) > 0
        assert errors[0].error_type == ErrorType.TIME_LIMIT

    def test_no_errors(self, tmp_path):
        """Test case with no errors."""
        output_file = tmp_path / "siesta.out"
        output_file.write_text(
            """
            SCF Convergence by DM criterion: T
            siesta: Final energy (eV):
            siesta:  Band Struct. =    -123.456789
            siesta:  Kinetic     =     456.789012
            siesta:  Total       =    -234.567890

            siesta: Program's energy decomposition (eV):
            siesta: Eions   =       1234.567890
            siesta: Ena     =        234.567890
            siesta: Ekin    =        456.789012
            siesta: Total   =       -234.567890

            siesta: Program stopping
            """
        )

        errors = detect_error(tmp_path)
        assert len(errors) == 0
        assert check_for_errors(tmp_path) is False
        assert get_error_type(tmp_path) is None


class TestErrorHandlers:
    """Test error handler functionality."""

    def test_scf_handler_check(self, tmp_path):
        """Test SCF handler error checking."""
        # Create output with SCF error
        output_file = tmp_path / "siesta.out"
        output_file.write_text(
            "SCF_NOT_CONV: SCF did not converge in maximum number of steps."
        )

        handler = SCFConvergenceHandler()
        # check() returns bool, not ErrorType
        has_error = handler.check(tmp_path)
        assert has_error is True

        # Verify error type using detect_error()
        errors = detect_error(tmp_path)
        assert errors[0].error_type == ErrorType.SCF_CONVERGENCE

    def test_scf_handler_corrections(self, tmp_path):
        """Test SCF handler correction strategies."""
        # Create minimal siesta.fdf file
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("SystemLabel test\n")

        handler = SCFConvergenceHandler()

        # Level 1 correction
        result = handler.correct(tmp_path)
        assert "errors" in result
        assert "actions" in result
        assert "Level 1" in result["actions"][0]
        # Custodian tracks corrections automatically
        handler.n_applied_corrections = 1

        # Level 2 correction
        result = handler.correct(tmp_path)
        assert "Level 2" in result["actions"][0]
        handler.n_applied_corrections = 2

        # Level 3 correction (with kick)
        result = handler.correct(tmp_path)
        assert "Level 3" in result["actions"][0]
        assert "kick" in result["actions"][0].lower()
        handler.n_applied_corrections = 3

    def test_memory_handler_corrections(self, tmp_path):
        """Test memory handler correction strategies."""
        # Create minimal siesta.fdf
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text(
            """
            MeshCutoff 300 Ry
            PAO.BasisSize DZP
            """
        )

        handler = MemoryHandler()

        # Level 1: Reduce diag memory
        result = handler.correct(tmp_path)
        assert "errors" in result
        assert "actions" in result
        assert "Level 1" in result["actions"][0]
        handler.n_applied_corrections = 1

        # Level 2: Add ParallelOverK
        result = handler.correct(tmp_path)
        assert "Level 2" in result["actions"][0]
        assert "ParallelOverK" in result["actions"][0]
        handler.n_applied_corrections = 2

        # Level 3: Reduce mesh cutoff
        result = handler.correct(tmp_path)
        assert "Level 3" in result["actions"][0]
        assert "mesh cutoff" in result["actions"][0].lower()
        handler.n_applied_corrections = 3

    def test_time_handler_with_checkpoint(self, tmp_path):
        """Test time handler with existing checkpoint files."""
        # Create checkpoint files and siesta.fdf
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("SystemLabel test\n")
        dm_file = tmp_path / "siesta.DM"
        xv_file = tmp_path / "siesta.XV"
        dm_file.write_text("dummy DM content")
        xv_file.write_text("dummy XV content")

        handler = TimeHandler()
        result = handler.correct(tmp_path)

        assert "errors" in result
        assert "actions" in result
        assert result["actions"] is not None
        assert len(result["actions"]) == 2  # DM and XV restart
        assert "density matrix" in result["actions"][0].lower()
        assert "structure" in result["actions"][1].lower()

    def test_time_handler_without_checkpoint(self, tmp_path):
        """Test time handler without checkpoint files."""
        handler = TimeHandler()
        result = handler.correct(tmp_path)

        # Should return actions: None when unfixable
        assert "errors" in result
        assert "actions" in result
        assert result["actions"] is None  # Cannot fix without checkpoint files

    def test_handler_max_attempts(self):
        """Test handler max attempts configuration."""
        # Test custom max_attempts
        handler = SCFConvergenceHandler(max_attempts=3)
        assert handler.max_attempts == 3
        assert handler.max_num_corrections == 3

        # Test default max_attempts
        handler_default = SCFConvergenceHandler()
        assert handler_default.max_attempts == 10
        assert handler_default.max_num_corrections == 10

        # Test that n_applied_corrections starts at 0
        # (custodian initializes this automatically)
        handler_new = SCFConvergenceHandler()
        assert (
            not hasattr(handler_new, "n_applied_corrections")
            or handler_new.n_applied_corrections == 0
        )


def test_demo_workflow():
    """Demonstration of complete error detection and correction workflow."""
    print("\n" + "=" * 70)
    print("SIESTA Custodian Error Detection Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Scenario 1: SCF Convergence Failure
        print("\nScenario 1: SCF Convergence Failure")
        print("-" * 70)

        output_file = tmp_path / "siesta.out"
        output_file.write_text(
            "SCF_NOT_CONV: SCF did not converge in maximum number of steps (required)."
        )

        errors = detect_error(tmp_path)
        print(f"Errors detected: {len(errors)}")
        for error in errors:
            print(f"  - Type: {error.error_type.value}")
            print(f"  - Severity: {error.severity.value}")
            print(f"  - Description: {error.description}")

        # Apply handler
        handler = SCFConvergenceHandler()
        if handler.check(tmp_path):
            print("\nApplying corrections:")
            for attempt in range(3):
                result = handler.correct(tmp_path)
                print(f"\n  Attempt {attempt + 1}:")
                print(f"    Errors: {result['errors']}")
                print(f"    Actions: {result['actions']}")
                # Manually increment for demo
                handler.n_applied_corrections = attempt + 1

        # Scenario 2: Memory Error
        print("\n\nScenario 2: Memory Error")
        print("-" * 70)

        output_file.write_text("Out of memory in dhscf\nAllocation failed")
        fdf_file = tmp_path / "siesta.fdf"
        fdf_file.write_text("MeshCutoff 400 Ry\nPAO.BasisSize TZP")

        errors = detect_error(tmp_path)
        print(f"Errors detected: {len(errors)}")
        for error in errors:
            print(f"  - Type: {error.error_type.value}")

        handler = MemoryHandler()
        if handler.check(tmp_path):
            print("\nApplying corrections:")
            result = handler.correct(tmp_path)
            print(f"    Errors: {result['errors']}")
            print(f"    Actions: {result['actions']}")

        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)


if __name__ == "__main__":
    # Run the demo
    test_demo_workflow()

    # Run pytest tests
    print("\n\nRunning pytest tests...\n")
    pytest.main([__file__, "-v"])
