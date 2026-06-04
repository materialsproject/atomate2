"""
Tests for core SIESTA flows (workflow composition and chaining).

These tests validate:
- DifferentBasisSCFFlowMaker workflow (multiple basis calculations)
- DifferentBasisFlowMaker workflow (progressive relaxations)
- DifferentBasisRelaxFlowMaker (fixed/variable cell combinations)
- Flow composition and job chaining
- Parameter strategies
- Serialization
"""

import pytest
from jobflow import Flow

from atomate2.siesta.flows.core import (
    DifferentBasisSCFFlowMaker,
    DifferentBasisFlowMaker,
    DifferentBasisRelaxFlowMaker,
)
from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker


class TestDifferentBasisSCF:
    """Tests for DifferentBasisSCFFlowMaker workflow."""

    def test_default_different_basis_scf(self):
        """Test creation of default DifferentBasisSCFFlowMaker."""
        maker = DifferentBasisSCFFlowMaker()

        assert maker.name == "Different basis scf"
        assert isinstance(maker.static_maker, StaticMaker)
        assert maker.strategy == "standard"

    def test_different_basis_scf_strategies(self):
        """Test different parameter strategies."""
        strategies = ["standard", "advanced", "legacy"]

        for strategy in strategies:
            maker = DifferentBasisSCFFlowMaker(strategy=strategy)
            assert maker.strategy == strategy

    def test_different_basis_scf_make_flow(self, si_structure):
        """Test that DifferentBasisSCFFlowMaker creates a valid flow."""
        maker = DifferentBasisSCFFlowMaker()
        flow = maker.make(si_structure)

        # Check flow structure
        assert isinstance(flow, Flow)
        assert flow.name == "Different basis scf"
        assert len(flow) > 0  # Should have jobs

        # Should create jobs for all basis sizes (26 basis sets)
        assert len(flow) == 26

    def test_different_basis_scf_job_names(self, si_structure):
        """Test that jobs have correct names with basis suffixes."""
        maker = DifferentBasisSCFFlowMaker()
        flow = maker.make(si_structure)

        # Check that job names include basis size
        basis_sizes = ["SZ", "DZ", "TZ", "DZP", "TZP"]
        job_names = [job.name for job in flow]

        for basis in basis_sizes:
            # At least one job should have this basis in its name
            assert any(basis in name for name in job_names)

    def test_different_basis_scf_standard_strategy(self, si_structure):
        """Test standard strategy parameters."""
        maker = DifferentBasisSCFFlowMaker(strategy="standard")

        # Get parameters for any basis
        params = maker._get_basis_params("DZP")

        assert "PAO.EnergyShift" in params
        assert "PAO.SplitNorm" in params
        assert params["PAO.EnergyShift"] == "0.01 Ry"
        assert params["PAO.SplitNorm"] == 0.15

    def test_different_basis_scf_advanced_strategy(self, si_structure):
        """Test advanced strategy with basis-specific parameters."""
        maker = DifferentBasisSCFFlowMaker(strategy="advanced")

        # Check SZ parameters
        sz_params = maker._get_basis_params("SZ")
        assert sz_params["PAO.EnergyShift"] == "0.02 Ry"
        assert sz_params["PAO.SplitNorm"] == 0.15

        # Check DZ parameters
        dz_params = maker._get_basis_params("DZ")
        assert dz_params["PAO.EnergyShift"] == "0.01 Ry"
        assert dz_params["PAO.SplitNorm"] == 0.20

        # Check TZ parameters
        tz_params = maker._get_basis_params("TZ")
        assert tz_params["PAO.EnergyShift"] == "0.005 Ry"
        assert tz_params["PAO.SplitNorm"] == 0.25

    def test_different_basis_scf_legacy_strategy(self, si_structure):
        """Test legacy strategy."""
        maker = DifferentBasisSCFFlowMaker(strategy="legacy")
        flow = maker.make(si_structure)

        # Legacy should still create valid flow
        assert isinstance(flow, Flow)
        assert len(flow) == 26

    def test_different_basis_scf_get_basis_sizes(self):
        """Test _get_basis_sizes returns expected list."""
        maker = DifferentBasisSCFFlowMaker()
        basis_sizes = maker._get_basis_sizes()

        # Should have 26 basis sizes
        assert len(basis_sizes) == 26

        # Check for key basis sets
        expected = ["SZ", "DZ", "TZ", "DZP", "TZP", "STANDARD", "MINIMAL"]
        for basis in expected:
            assert basis in basis_sizes

    def test_different_basis_scf_with_custom_maker(self, si_structure):
        """Test DifferentBasisSCFFlowMaker with custom StaticMaker."""
        from atomate2.siesta.sets.core import StaticSetGenerator

        custom_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"a2s_kpts": [6, 6, 6]})
        )

        maker = DifferentBasisSCFFlowMaker(static_maker=custom_maker)
        flow = maker.make(si_structure)

        assert len(flow) == 26
        assert maker.static_maker == custom_maker

    def test_different_basis_scf_prev_dir(self, si_structure, tmp_path):
        """Test DifferentBasisSCFFlowMaker with previous directory."""
        maker = DifferentBasisSCFFlowMaker()

        prev_dir = tmp_path / "previous"
        prev_dir.mkdir()

        flow = maker.make(si_structure, prev_dir=str(prev_dir))
        assert isinstance(flow, Flow)

    def test_different_basis_scf_serialization(self):
        """Test DifferentBasisSCFFlowMaker serialization."""
        maker = DifferentBasisSCFFlowMaker(strategy="advanced")

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "DifferentBasisSCFFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, DifferentBasisSCFFlowMaker)
        assert maker_restored.strategy == "advanced"


class TestDifferentBasis:
    """Tests for DifferentBasisFlowMaker workflow (progressive relaxations)."""

    def test_default_different_basis(self):
        """Test creation of default DifferentBasisFlowMaker."""
        maker = DifferentBasisFlowMaker()

        assert maker.name == "Different basis"
        assert isinstance(maker.relax_maker_sz, RelaxMaker)
        assert isinstance(maker.relax_maker_szp, RelaxMaker)
        assert isinstance(maker.relax_maker_dz, RelaxMaker)
        assert isinstance(maker.relax_maker_dzp, RelaxMaker)
        assert isinstance(maker.relax_maker_tz, RelaxMaker)
        assert isinstance(maker.relax_maker_tzp, RelaxMaker)

    def test_different_basis_make_flow(self, si_structure):
        """Test that DifferentBasisFlowMaker creates a valid flow."""
        maker = DifferentBasisFlowMaker()
        flow = maker.make(si_structure)

        # Check flow structure
        assert isinstance(flow, Flow)
        assert flow.name == "Different basis"

        # Should create 6 relaxation jobs (SZ, SZP, DZ, DZP, TZ, TZP)
        assert len(flow) == 6

    def test_different_basis_job_names(self, si_structure):
        """Test that jobs have correct basis suffixes."""
        maker = DifferentBasisFlowMaker()
        flow = maker.make(si_structure)

        # Check job names include basis
        job_names = [job.name for job in flow]

        expected_suffixes = ["-SZ", "-SZP", "-DZ", "-DZP", "-TZ", "-TZP"]
        for suffix in expected_suffixes:
            assert any(suffix in name for name in job_names)

    def test_different_basis_with_custom_makers(self, si_structure):
        """Test DifferentBasisFlowMaker with custom RelaxMakers."""
        from atomate2.siesta.sets.core import RelaxSetGenerator

        custom_maker = RelaxMaker(
            input_set_generator=RelaxSetGenerator(
                user_params={"a2s_kpts": [4, 4, 4]}, basis_set_size="DZP"
            )
        )

        maker = DifferentBasisFlowMaker(relax_maker_dzp=custom_maker)
        flow = maker.make(si_structure)

        assert len(flow) == 6
        assert maker.relax_maker_dzp == custom_maker

    def test_different_basis_prev_dir(self, si_structure, tmp_path):
        """Test DifferentBasisFlowMaker with previous directory."""
        maker = DifferentBasisFlowMaker()

        prev_dir = tmp_path / "previous"
        prev_dir.mkdir()

        flow = maker.make(si_structure, prev_dir=str(prev_dir))
        assert isinstance(flow, Flow)
        assert len(flow) == 6

    def test_different_basis_serialization(self):
        """Test DifferentBasisFlowMaker serialization."""
        maker = DifferentBasisFlowMaker()

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "DifferentBasisFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, DifferentBasisFlowMaker)


class TestDifferentBasisRelaxMaker:
    """Tests for DifferentBasisRelaxFlowMaker (fixed/variable cell combinations)."""

    def test_default_different_basis_relax_maker(self):
        """Test creation of default DifferentBasisRelaxFlowMaker."""
        maker = DifferentBasisRelaxFlowMaker()

        assert maker.name == "Different basis relaxation"
        assert isinstance(maker.relax_maker_sz_fixed, RelaxMaker)
        assert isinstance(maker.relax_maker_sz_variable, RelaxMaker)
        assert isinstance(maker.relax_maker_dz_fixed, RelaxMaker)
        assert isinstance(maker.relax_maker_dz_variable, RelaxMaker)

    def test_different_basis_relax_maker_make_flow(self, si_structure):
        """Test that DifferentBasisRelaxFlowMaker creates a valid flow."""
        maker = DifferentBasisRelaxFlowMaker()
        flow = maker.make(si_structure)

        # Check flow structure
        assert isinstance(flow, Flow)
        assert flow.name == "Different basis relaxation"

        # Should create 4 relaxation jobs
        # (SZ fixed, SZ variable, DZ fixed, DZ variable)
        assert len(flow) == 4

    def test_different_basis_relax_maker_job_names(self, si_structure):
        """Test that jobs have correct suffixes."""
        maker = DifferentBasisRelaxFlowMaker()
        flow = maker.make(si_structure)

        job_names = [job.name for job in flow]

        # Check for expected name patterns
        expected_patterns = [
            "-SZ-fixed-cell",
            "-SZ-variable-cell",
            "-DZ-fixed-cell",
            "-DZ-variable-cell",
        ]

        for pattern in expected_patterns:
            assert any(pattern in name for name in job_names)

    def test_different_basis_relax_maker_with_custom_makers(self, si_structure):
        """Test DifferentBasisRelaxFlowMaker with custom makers."""
        custom_maker = RelaxMaker.fixed_cell_relaxation(
            user_params={"a2s_kpts": [6, 6, 6], "PAO.BasisSize": "DZP"}
        )

        maker = DifferentBasisRelaxFlowMaker(relax_maker_dz_fixed=custom_maker)
        flow = maker.make(si_structure)

        assert len(flow) == 4
        assert maker.relax_maker_dz_fixed == custom_maker

    def test_different_basis_relax_maker_prev_dir(self, si_structure, tmp_path):
        """Test DifferentBasisRelaxFlowMaker with previous directory."""
        maker = DifferentBasisRelaxFlowMaker()

        prev_dir = tmp_path / "previous"
        prev_dir.mkdir()

        flow = maker.make(si_structure, prev_dir=str(prev_dir))
        assert isinstance(flow, Flow)
        assert len(flow) == 4

    def test_different_basis_relax_maker_serialization(self):
        """Test DifferentBasisRelaxFlowMaker serialization."""
        maker = DifferentBasisRelaxFlowMaker()

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "DifferentBasisRelaxFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, DifferentBasisRelaxFlowMaker)


class TestFlowComposition:
    """Integration tests for flow composition and chaining."""

    def test_flow_with_multiple_structures(self, si_structure, al_structure):
        """Test that flows work with different structures."""
        maker = DifferentBasisSCFFlowMaker()

        flow1 = maker.make(si_structure)
        flow2 = maker.make(al_structure)

        # Both should be valid
        assert isinstance(flow1, Flow)
        assert isinstance(flow2, Flow)
        assert len(flow1) == len(flow2)  # Same number of basis sets

    def test_all_flow_makers_create_valid_flows(self, si_structure):
        """Test that all core flow makers can create valid flows."""
        makers = [
            DifferentBasisSCFFlowMaker(),
            DifferentBasisSCFFlowMaker(strategy="advanced"),
            DifferentBasisSCFFlowMaker(strategy="legacy"),
            DifferentBasisFlowMaker(),
            DifferentBasisRelaxFlowMaker(),
        ]

        for maker in makers:
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)
            assert hasattr(flow, "name")
            assert len(flow) > 0

    def test_flow_output_references(self, si_structure):
        """Test that flows have proper output handling."""
        makers = [
            DifferentBasisSCFFlowMaker(),
            DifferentBasisFlowMaker(),
            DifferentBasisRelaxFlowMaker(),
        ]

        for maker in makers:
            flow = maker.make(si_structure)
            # Flow should be iterable (has jobs)
            jobs = list(flow)
            assert len(jobs) > 0

            # Each job should have a name
            for job in jobs:
                assert hasattr(job, "name")
                assert hasattr(job, "function")

    def test_different_basis_scf_all_strategies(self, si_structure):
        """Test DifferentBasisSCFFlowMaker with all strategies."""
        strategies = ["standard", "advanced", "legacy"]

        for strategy in strategies:
            maker = DifferentBasisSCFFlowMaker(strategy=strategy)
            flow = maker.make(si_structure)

            assert isinstance(flow, Flow)
            assert len(flow) == 26  # All have 26 basis sets

    def test_flow_with_custom_name(self, si_structure):
        """Test creating flows with custom names."""
        maker1 = DifferentBasisSCFFlowMaker(name="My Custom SCF Flow")
        maker2 = DifferentBasisFlowMaker(name="My Custom Relax Flow")

        flow1 = maker1.make(si_structure)
        flow2 = maker2.make(si_structure)

        assert flow1.name == "My Custom SCF Flow"
        assert flow2.name == "My Custom Relax Flow"


class TestBackwardCompatibility:
    """Tests for backward compatibility functions."""

    def test_different_basis_scf_advance_deprecated(self, si_structure):
        """Test deprecated DifferentBasisSCFAdvance function."""
        from atomate2.siesta.flows.core import DifferentBasisSCFAdvance

        # Should raise deprecation warning
        with pytest.warns(DeprecationWarning, match="deprecated"):
            maker = DifferentBasisSCFAdvance()

        # But should still work
        assert isinstance(maker, DifferentBasisSCFFlowMaker)
        assert maker.strategy == "advanced"

    def test_different_basis_scf_old_deprecated(self, si_structure):
        """Test deprecated DifferentBasisSCFOld function."""
        from atomate2.siesta.flows.core import DifferentBasisSCFOld

        # Should raise deprecation warning
        with pytest.warns(DeprecationWarning, match="deprecated"):
            maker = DifferentBasisSCFOld()

        # But should still work
        assert isinstance(maker, DifferentBasisSCFFlowMaker)
        assert maker.strategy == "legacy"


class TestFlowEdgeCases:
    """Test edge cases and error handling."""

    def test_flow_with_none_prev_dir(self, si_structure):
        """Test flows with None as prev_dir."""
        makers = [
            DifferentBasisSCFFlowMaker(),
            DifferentBasisFlowMaker(),
            DifferentBasisRelaxFlowMaker(),
        ]

        for maker in makers:
            flow = maker.make(si_structure, prev_dir=None)
            assert isinstance(flow, Flow)

    def test_flow_maker_modification_doesnt_affect_flows(self, si_structure):
        """Test that modifying maker after flow creation doesn't affect flow."""
        maker = DifferentBasisSCFFlowMaker(strategy="standard")
        flow1 = maker.make(si_structure)

        # Modify maker (create new with different strategy)
        maker = DifferentBasisSCFFlowMaker(strategy="advanced")
        flow2 = maker.make(si_structure)

        # Flows should be independent
        assert flow1 is not flow2
        assert len(flow1) == len(flow2)  # Both should have 25 jobs

    def test_multiple_flows_from_same_maker(self, si_structure, al_structure):
        """Test creating multiple flows from the same maker."""
        maker = DifferentBasisSCFFlowMaker()

        flow1 = maker.make(si_structure)
        flow2 = maker.make(al_structure)

        # Flows should be independent
        assert flow1 is not flow2
        assert len(flow1) == len(flow2)


# ============================================================================
# Comprehensive Core Workflows Testing
# Pattern: dry-run, inheritance, edge cases
# ============================================================================


class TestCoreDryRun:
    """Tests for dry-run mode support in core workflows."""

    def test_different_basis_scf_with_dry_run_enabled(self, si_structure):
        """Test DifferentBasisSCFFlowMaker with dry_run=True."""
        maker = DifferentBasisSCFFlowMaker(dry_run=True)
        flow = maker.make(si_structure)

        # Verify dry_run is set
        assert maker.dry_run is True
        assert isinstance(flow, Flow)

    def test_different_basis_scf_dry_run_default_false(self):
        """Test that DifferentBasisSCFFlowMaker dry_run defaults to False."""
        maker = DifferentBasisSCFFlowMaker()
        assert maker.dry_run is False

    def test_different_basis_with_dry_run_enabled(self, si_structure):
        """Test DifferentBasisFlowMaker with dry_run=True."""
        maker = DifferentBasisFlowMaker(dry_run=True)
        flow = maker.make(si_structure)

        # Verify dry_run is set
        assert maker.dry_run is True
        assert isinstance(flow, Flow)

    def test_different_basis_dry_run_default_false(self):
        """Test that DifferentBasisFlowMaker dry_run defaults to False."""
        maker = DifferentBasisFlowMaker()
        assert maker.dry_run is False

    def test_different_basis_relax_maker_with_dry_run_enabled(self, si_structure):
        """Test DifferentBasisRelaxFlowMaker with dry_run=True."""
        maker = DifferentBasisRelaxFlowMaker(dry_run=True)
        flow = maker.make(si_structure)

        # Verify dry_run is set
        assert maker.dry_run is True
        assert isinstance(flow, Flow)

    def test_different_basis_relax_maker_dry_run_default_false(self):
        """Test that DifferentBasisRelaxFlowMaker dry_run defaults to False."""
        maker = DifferentBasisRelaxFlowMaker()
        assert maker.dry_run is False


class TestCoreInheritance:
    """Tests for BaseSiestaFlowMaker inheritance."""

    def test_different_basis_scf_inherits_from_base_siesta_flow_maker(self):
        """Test that DifferentBasisSCFFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = DifferentBasisSCFFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_different_basis_scf_has_dry_run_attribute(self):
        """Test that DifferentBasisSCFFlowMaker has dry_run attribute."""
        maker = DifferentBasisSCFFlowMaker()
        assert hasattr(maker, "dry_run")

    def test_different_basis_inherits_from_base_siesta_flow_maker(self):
        """Test that DifferentBasisFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = DifferentBasisFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_different_basis_has_dry_run_attribute(self):
        """Test that DifferentBasisFlowMaker has dry_run attribute."""
        maker = DifferentBasisFlowMaker()
        assert hasattr(maker, "dry_run")

    def test_different_basis_relax_maker_inherits_from_base_siesta_flow_maker(self):
        """Test that DifferentBasisRelaxFlowMaker inherits from BaseSiestaFlowMaker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        maker = DifferentBasisRelaxFlowMaker()
        assert isinstance(maker, BaseSiestaFlowMaker)

    def test_different_basis_relax_maker_has_dry_run_attribute(self):
        """Test that DifferentBasisRelaxFlowMaker has dry_run attribute."""
        maker = DifferentBasisRelaxFlowMaker()
        assert hasattr(maker, "dry_run")


class TestCoreEdgeCasesExtended:
    """Extended edge cases for core workflows."""

    def test_different_basis_scf_unknown_basis_in_advanced_strategy(self):
        """Test advanced strategy with unknown basis returns default parameters."""
        maker = DifferentBasisSCFFlowMaker(strategy="advanced")

        # Test with unknown basis name
        params = maker._get_basis_params("UNKNOWN_BASIS")

        # Should return default parameters
        assert "PAO.EnergyShift" in params
        assert "PAO.SplitNorm" in params
        assert params["PAO.EnergyShift"] == "0.01 Ry"
        assert params["PAO.SplitNorm"] == 0.15

    def test_different_basis_scf_legacy_strategy_returns_empty_params(self):
        """Test that legacy strategy returns empty parameters."""
        maker = DifferentBasisSCFFlowMaker(strategy="legacy")

        # Legacy strategy doesn't use parameters
        params = maker._get_basis_params("DZP")
        assert params == {}
